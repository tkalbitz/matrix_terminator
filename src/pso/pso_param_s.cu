/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <curand_kernel.h>

#include "pso_param_s.h"
#include "pso_memory.h"
#include "ya_malloc.h"

extern void update_lbest(struct pso_instance& inst, struct param_s& ps);

void param_s_init(struct pso_instance& inst, struct param_s& ps)
{
	int s_set[] = {1, 2, 5, 10, 25};

	ps.s_set_len = 5;
	ps.s = 2;
	ps.s_count = inst.width_per_line / ps.s;

	ps.old_rat = (double*)ya_malloc(BLOCKS * sizeof(double));
	ps.s_set   = (int*)ya_malloc(ps.s_set_len * sizeof(*ps.s_set));

	memcpy(ps.s_set, s_set, ps.s_set_len*sizeof(*s_set));
	for(int i = 0; i < BLOCKS; i++) {
		ps.old_rat[i] = FLT_MAX;
	}
}

void param_s_destroy(struct param_s& ps)
{
	free(ps.s_set);
	free(ps.old_rat);
}

__global__ void reset_lbest(struct pso_instance inst) {
	const int len = inst.width_per_line * PARAM_COUNT * BLOCKS;

	for(int i = tx; i < len; i += blockDim.x) {
		inst.lbrat[i] = FLT_MAX;
	}
}

__global__ void permutate_columns(struct pso_instance inst) {
	const int len = inst.width_per_line;
	int* cols = inst.col_permut + len * tx;
	int tmp, r1, r2;

	const int id = get_thread_id();
	curandState rnd = inst.rnd_states[id];

	for(int i = 0; i < len; i++) {
		r1 = curand(&rnd) % len;
		r2 = curand(&rnd) % len;

		tmp  = cols[r1];
		cols[r1] = cols[r2];
		cols[r2]  = tmp;
	}

	inst.rnd_states[id] = rnd;
}

void print_col_permut(struct pso_instance& inst)
{
	const int width = inst.width_per_line * sizeof(int);
	int* col = (int*)ya_malloc(width);
	CUDA_CALL(cudaMemcpy(col, inst.col_permut, width, cudaMemcpyDeviceToHost));

	printf("col perm:");
	for(int i = 0; i < inst.width_per_line; i++) {
		printf("%d ", col[i]);
	}
	printf("\n");
}

void param_s_update(struct pso_instance& inst, struct param_s& ps)
{
	const int width = BLOCKS * sizeof(double);
	double *new_rat = (double*)ya_malloc(width);

	CUDA_CALL(cudaMemcpy(new_rat, inst.gbrat, width, cudaMemcpyDeviceToHost));

	const double eps = 0.00001;

	//TODO: More than one block
	if(abs(new_rat[0] - ps.old_rat[0]) < eps) {
		CUDA_CALL(cudaMemcpy(inst.particle, inst.particle_lbest,
				     inst.total * sizeof(double),
				     cudaMemcpyDeviceToDevice));
		reset_lbest<<<512, BLOCKS>>>(inst);

		ps.s = ps.s_set[rand() % ps.s_set_len];
		ps.s_count = inst.width_per_line / ps.s;
		permutate_columns<<<BLOCKS, 1>>>(inst);
		update_lbest(inst, ps);
	}

//	printf("s:%d %f %f\n", ps.s, new_rat[0], ps.old_rat[0]);
	memcpy(ps.old_rat, new_rat, width);
}
