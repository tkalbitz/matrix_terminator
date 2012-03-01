/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
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

#include "custom/c_rating.h"
#include "custom/c_setup.h"
#include "custom/c_instance.h"

#include "ya_malloc.h"

static void copy_result(struct c_instance& inst, float* dest, int block, int bpos)
{
	const int width = inst.itotal * sizeof(float);
	float* global_cpy = (float*)ya_malloc(width);
	memset(global_cpy, 1, width);

	CUDA_CALL(cudaMemcpy(global_cpy, inst.instances, width,
			     cudaMemcpyDeviceToHost));

	int block_offset = inst.width_per_inst * inst.icount * block;
	float* ptr = global_cpy + block_offset + bpos * inst.width_per_inst;

	memcpy(dest, ptr, inst.width_per_inst * sizeof(*dest));
	free(global_cpy);
}

int c_run(const int     instance,
	  const int     cycles,
	  const int     asteps,
	  float* const result)
{
	struct c_info_t* const c_info = c_get(instance);
	if(c_info == NULL)
		return E_INVALID_INST;

	struct c_instance* host_inst = c_info->inst;
	struct c_instance inst;

	inst = *host_inst;
	inst.rules = c_create_dev_rules(inst);

	int3* stack;
	unsigned int* top;
	const size_t slen = BLOCKS * inst.rules_count * inst.width_per_matrix;
	CUDA_CALL(cudaMalloc(&stack, inst.num_matrices * slen * sizeof(*stack)));
	CUDA_CALL(cudaMalloc(&top, BLOCKS * sizeof(*top)));

	dim3 threads(inst.mdim, inst.mdim);
	dim3 blocks(BLOCKS);

	setup_best_kernel<<<1, BLOCKS>>>(inst);
	CUDA_CALL(cudaGetLastError());

	setup_instances_kernel<<<1, 320>>>(inst);
	CUDA_CALL(cudaGetLastError());

	patch_matrix_kernel<<<BLOCKS, inst.mdim>>>(inst);
	CUDA_CALL(cudaGetLastError());

	setup_rating(inst);

	float* rating = (float*)ya_malloc(inst.icount * sizeof(float));
	int* best_idx = (int*)ya_malloc(BLOCKS * sizeof(best_idx));

	int rounds = INT_MAX;
	int block = 0; int pos = 0;

	for(unsigned long i = 0; i < cycles; i++) {
		start_astep(inst, stack, top, asteps);

		if(i % 100 == 0) {
			CUDA_CALL(cudaMemcpy(rating, inst.best, BLOCKS * sizeof(*rating), cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaMemcpy(best_idx, inst.best_idx, BLOCKS * sizeof(*best_idx), cudaMemcpyDeviceToHost));
			pos   = best_idx[0];

			for(int j = 0; j < BLOCKS; j++) {
				if(rating[j] == 0.) {
					rounds = i;
					block = j;
					i = cycles;
					pos = best_idx[j];
					break;
				}
			}
		}
	}

	free(rating);
	free(best_idx);
	cudaFree(inst.rules);
	cudaFree(stack);

	if(rounds != INT_MAX) {
		copy_result(inst, result, block, pos);
	}

	return rounds;
}
