/*
 * pso_instance.cu
 *
 *  Created on: Sep 28, 2011
 *      Author: tkalbitz
 */

#include "assert.h"

#include "pso_config.h"
#include "pso_instance.h"
#include "pso_setup.h"

int get_pso_threads(const struct pso_instance * const inst)
{
	return min(512, inst->width_per_line);
}

void init_rnd_generator(struct pso_instance *inst, int seed)
{
	curandState *rnd_states;
	const int count = get_pso_threads(inst);

	//TODO: How many?
	CUDA_CALL(cudaMalloc(&rnd_states,
			     count * BLOCKS * PARTICLE_COUNT * sizeof(curandState)));

	const dim3 blocks(BLOCKS, inst->dim.particles);
	setup_rnd_kernel<<<blocks, count>>>(rnd_states, seed);
	CUDA_CALL(cudaGetLastError());
	cudaThreadSynchronize();
	inst->rnd_states = rnd_states;
}

void set_num_matrices(struct pso_instance* inst)
{
	int m = INT_MIN;
	for(size_t i = 0; i < inst->rules_len; i++)
		m = max(m, inst->rules[i]);

	inst->num_matrices = m + 1; /* matrices are zero based */
}

void alloc_particle_matrix(struct pso_instance *inst)
{
	assert(inst->num_matrices != 0);

	const size_t len = inst->total * sizeof(double);

	const size_t gbest_len = inst->width_per_line *
				 inst->dim.blocks * sizeof(double);

	CUDA_CALL(cudaMalloc(&(inst->s), BLOCKS * sizeof(int)));
	CUDA_CALL(cudaMalloc(&(inst->particle),       len));
	CUDA_CALL(cudaMalloc(&(inst->particle_lbest), len));
	CUDA_CALL(cudaMalloc(&(inst->particle_gbest), gbest_len));
	CUDA_CALL(cudaMalloc(&(inst->rat_tmp), gbest_len * inst->dim.particles));
	CUDA_CALL(cudaMalloc(&(inst->col_permut), inst->width_per_line * BLOCKS * sizeof(int)));
}

void alloc_result_matrix(struct pso_instance *inst)
{
	const size_t len = inst->dim.particles * inst->dim.matrix_width *
			   inst->dim.matrix_height * inst->dim.blocks *
			   sizeof(double);

	CUDA_CALL(cudaMalloc(&(inst->res), len));
}

void alloc_rating(struct pso_instance *inst)
{
	size_t len = inst->width_per_line * inst->dim.particles *
		     inst->dim.blocks;

	CUDA_CALL(cudaMalloc(&(inst->prat) ,     len * sizeof(double)));
	CUDA_CALL(cudaMalloc(&(inst->lbrat),     len * sizeof(double)));
	CUDA_CALL(cudaMalloc(&(inst->lbest_idx), len * sizeof(int)));
	CUDA_CALL(cudaMalloc(&(inst->gbrat), inst->width_per_line * inst->dim.blocks * sizeof(double)));
	CUDA_CALL(cudaMalloc(&(inst->gb_best), inst->dim.blocks * sizeof(double)));
	CUDA_CALL(cudaMalloc(&(inst->gb_old) , inst->dim.blocks * sizeof(double)));
}

void pso_inst_init(struct pso_instance* const inst, int matrix_width)
{
	inst->dim.blocks    = BLOCKS;
	inst->dim.particles = PARTICLE_COUNT;
	inst->dim.matrix_width  = matrix_width;
	inst->dim.matrix_height = matrix_width;

	inst->res_block = 0;
	inst->res_parent = 0;
	inst->res_child_block = 0;
	inst->res_child_idx = 0;

	set_num_matrices(inst);

	inst->width_per_inst = inst->num_matrices *    /* there are n matrices needed for the rules */
			       inst->dim.matrix_width; /* each one has a fixed width */

	inst->width_per_matrix = inst->dim.matrix_width * inst->dim.matrix_height;
	inst->width_per_line = inst->width_per_inst * inst->dim.matrix_height;
	inst->total = inst->width_per_line * inst->dim.particles * inst->dim.blocks;
	alloc_particle_matrix(inst);
	alloc_result_matrix(inst);
	alloc_rating(inst);
	init_rnd_generator(inst, (int)time(0));
}

void pso_inst_cleanup(struct pso_instance * const inst,
		  struct pso_instance * const dev_inst)
{
	if(dev_inst != NULL)
		cudaFree(dev_inst);

	cudaFree(inst->rnd_states);
	cudaFree(inst->col_permut);
	cudaFree(inst->particle);
	cudaFree(inst->particle_gbest);
	cudaFree(inst->particle_lbest);
	cudaFree(inst->res);
	cudaFree(inst->prat);
	cudaFree(inst->lbrat);
	cudaFree(inst->lbest_idx);
	cudaFree(inst->gb_best);
	cudaFree(inst->gb_old);
	cudaFree(inst->rat_tmp);
	cudaFree(inst->s);
}

struct pso_instance* pso_inst_create_dev_inst(struct pso_instance *inst,
					      int** dev_rules)
{
	struct pso_instance *dev_inst;
	int *rules = inst->rules;
	int *tmp_dev_rules;
	CUDA_CALL(cudaMalloc(&tmp_dev_rules, inst->rules_len * sizeof(int)));
	CUDA_CALL(cudaMemcpy(tmp_dev_rules,  rules, inst->rules_len * sizeof(int),
					cudaMemcpyHostToDevice));

	inst->rules = tmp_dev_rules;
	CUDA_CALL(cudaMalloc(&dev_inst, sizeof(*dev_inst)));
	CUDA_CALL(cudaMemcpy(dev_inst, inst, sizeof(*dev_inst),
					cudaMemcpyHostToDevice));
//	inst->rules = rules;

	if(dev_rules != NULL)
		*dev_rules = tmp_dev_rules;

	return dev_inst;
}

void pso_inst_copy_dev_to_host(struct pso_instance * const dev,
			      struct pso_instance * const host)
{
	int *rules = host->rules;
	CUDA_CALL(cudaMemcpy(host, dev, sizeof(*dev), cudaMemcpyDeviceToHost));
	host->rules = rules;
}

