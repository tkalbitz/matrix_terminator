/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#include <assert.h>
#include <math.h>
#include <time.h>

#include "c_instance.h"
#include "c_setup.h"

void init_rnd_generator(struct c_instance& inst, int seed)
{
	curandState *rnd_states;

	int count = 320;

	CUDA_CALL(cudaMalloc(&rnd_states, count * BLOCKS *
			sizeof(curandState)));

	inst.rnd_states = rnd_states;

	setup_c_rnd_kernel<<<1, count>>>(inst, seed);
	CUDA_CALL(cudaGetLastError());
}

void set_num_matrices(struct c_instance& inst)
{
	int m = INT_MIN;
	for(size_t i = 0; i < inst.rules_len; i++)
		m = max(m, inst.rules[i]);

	inst.num_matrices = m + 1; /* matrices are zero based */
}

void alloc_instance_mem(struct c_instance& inst)
{
	assert(inst.num_matrices != 0);

	const size_t ilen = inst.itotal * sizeof(float);

	CUDA_CALL(cudaMalloc(&(inst.instances),  ilen));
	CUDA_CALL(cudaMalloc(&(inst.best), BLOCKS * sizeof(*inst.best)));
	CUDA_CALL(cudaMalloc(&(inst.best_idx), BLOCKS * sizeof(*inst.best_idx)));

	const size_t ratlen = BLOCKS * inst.icount * sizeof(*inst.rating);
	CUDA_CALL(cudaMalloc(&(inst.rating), ratlen));
}

void c_inst_init(struct c_instance& inst, int matrix_width)
{
	inst.mdim = matrix_width;
	set_num_matrices(inst);

	inst.width_per_matrix = inst.mdim * inst.mdim;
	inst.width_per_inst = inst.num_matrices * inst.mdim * inst.mdim;

	inst.itotal = inst.width_per_inst * inst.icount * BLOCKS;

	alloc_instance_mem(inst);
	init_rnd_generator(inst, (int)time(0));
}

void c_inst_cleanup(struct c_instance& inst)
{
	cudaFree(inst.rnd_states);
	cudaFree(inst.instances);
	cudaFree(inst.rating);
	cudaFree(inst.best);
	cudaFree(inst.best_idx);
}

int* c_create_dev_rules(struct c_instance& inst)
{
	int *rules = inst.rules;
	int *tmp_dev_rules;
	CUDA_CALL(cudaMalloc(&tmp_dev_rules, inst.rules_len * sizeof(int)));
	CUDA_CALL(cudaMemcpy(tmp_dev_rules,  rules, inst.rules_len * sizeof(int),
			     cudaMemcpyHostToDevice));

	return tmp_dev_rules;
}
