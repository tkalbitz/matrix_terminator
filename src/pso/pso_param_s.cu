/*
 * pso_param_s.cu
 *
 *  Created on: Jan 27, 2012
 *      Author: tkalbitz
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

#include "pso_param_s.h"
#include "ya_malloc.h"

extern void update_lbest(struct pso_instance& inst, struct param_s& ps);

void param_s_init(struct pso_instance& inst, struct param_s& ps)
{
	int s_set[] = {1, 2, 5, 10, 25, 50};

	ps.s_set_len = 6;
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

void param_s_update(struct pso_instance& inst, struct param_s& ps)
{
	const int width = BLOCKS * sizeof(double);
	double *new_rat = (double*)ya_malloc(width);

	CUDA_CALL(cudaMemcpy(new_rat, inst.gbrat, width, cudaMemcpyDeviceToHost));

	const double eps = 0.00001;

	//TODO: More than one block
	if(abs(new_rat[0] - ps.old_rat[0]) < eps) {
		ps.s = ps.s_set[rand() % ps.s_set_len];
		ps.s_count = inst.width_per_line / ps.s;
		printf("set s:%d %f %f\n", ps.s, new_rat[0], ps.old_rat[0]);
		reset_lbest<<<512, 1>>>(inst);
		update_lbest(inst, ps);
	}

	printf("s:%d %f %f\n", ps.s, new_rat[0], ps.old_rat[0]);
	memcpy(ps.old_rat, new_rat, width);
}
