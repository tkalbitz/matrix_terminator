/*
 * matrix_generator.c
 *
 *  Created on: Sep 22, 2011
 *      Author: tkalbitz
 */

extern "C"
{
#include "matrix_generator.h"
}

#include "mat_lib_info.h"
#include "evo_error.h"
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <ctype.h>
#include <errno.h>

#include <sys/wait.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "ccpso2/pso.h"
#include "ccpso2/pso_rating.h"
#include "ccpso2/pso_setup.h"
#include "ccpso2/pso_param_s.h"

#include "ya_malloc.h"

void update_lbest(struct pso_instance& inst, struct param_s& ps)
{
	const dim3 blocks(BLOCKS, inst.dim.particles);
	const dim3 threads(inst.dim.matrix_width, inst.dim.matrix_height);

	for(int c = 0; c < ps.s_count; c++) {
		pso_calc_res<<<blocks, threads>>>(inst, ps.s, c);
		CUDA_CALL(cudaGetLastError());
		cudaThreadSynchronize();
		CUDA_CALL(cudaGetLastError());

		pso_evaluation_lbest<<<BLOCKS, PARTICLE_COUNT>>>(inst, ps.s, c * PARTICLE_COUNT);
		CUDA_CALL(cudaGetLastError());
		cudaThreadSynchronize();
		CUDA_CALL(cudaGetLastError());
	}
}

int pso_run(const int     instance,
	    const int     cycles,
	    double* const result)
{
	struct pso_info_t* const pso_info = pso_get(instance);
	if(pso_info == NULL)
		return E_INVALID_INST;

	struct pso_instance *inst = pso_info->inst;
	struct pso_instance *dev_inst;
	int *dev_rules;

	dev_inst = pso_inst_create_dev_inst(inst, &dev_rules);

	setup_global_particle_kernel<<<1, 320>>>(dev_inst);
	CUDA_CALL(cudaGetLastError());
	setup_particle_kernel<<<1, 320>>>(dev_inst);
	CUDA_CALL(cudaGetLastError());

	setup_rating<<<1, 512>>>(dev_inst);
	CUDA_CALL(cudaGetLastError());

	setup_col_permut<<<1, 512>>>(inst->col_permut,
			             inst->width_per_line * BLOCKS,
			             inst->width_per_line);
	CUDA_CALL(cudaGetLastError());

	// Prepare
	struct param_s ps;
	param_s_init(*inst, ps);

	const int width = inst->dim.blocks;
	double * const rating = (double*)ya_malloc(width * sizeof(double));

	int rounds = -1;
	int block = 0;

	for(int i = 0; i < cycles; i++) {
		update_lbest(*inst, ps);
		param_s_update(*inst, ps);

		pso_neighbor_best<<<BLOCKS, PARTICLE_COUNT>>>(*inst, ps.s);
		CUDA_CALL(cudaGetLastError());
		cudaThreadSynchronize();
		CUDA_CALL(cudaGetLastError());

		pso_swarm_step_ccpso2<<<BLOCKS, 64>>>(*inst, ps.s);
		CUDA_CALL(cudaGetLastError());
		cudaThreadSynchronize();
		CUDA_CALL(cudaGetLastError());

		CUDA_CALL(cudaMemcpy(rating, inst->gbrat, width * sizeof(double),
						cudaMemcpyDeviceToHost));

		if(rating[0] == 0.) {
			rounds = i;
			block = i;
			i = cycles;
		}
	}

	const int cwidth = inst->dim.blocks *
		           inst->width_per_line *
		           sizeof(double);

	CUDA_CALL(cudaMemcpy(result, inst->particle_gbest, cwidth,
			cudaMemcpyDeviceToHost));


	param_s_destroy(ps);
	cudaFree(dev_inst);
	cudaFree(dev_rules);
	return rounds;
}
