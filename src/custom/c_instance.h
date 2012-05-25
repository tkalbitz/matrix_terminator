/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef C_INSTANCE_H_
#define C_INSTANCE_H_

#include <stdio.h>
#include <stdint.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "c_config.h"

#define PARAM_COUNT    3

struct c_instance
{
	int num_matrices;         /* number of matrices in rules */
	int mdim;                 /* dimension of the matrix */
	float parent_max;         /* maximum of a matrix position */

	int icount;               /* number of instances */

	int itotal;               /* number of bytes for all instances */

	int width_per_inst;       /* number elements for one instance */
	int width_per_matrix;     /* number elements for one matrix */

	float* instances;         /* all instances */
	float* rating;            /* rating of the instances */

	float* best;              /* best rating of the different blocks */
	int*   best_idx;          /* where is the best located */

	curandState *rnd_states;  /* random number generator states */

	float delta;              /* numbers in the matrices are multiple the amount */
	float eps;
	int*   rules;             /* rules that must be matched */
	size_t rules_len;         /* number of elements in rules */
	size_t rules_count;       /* number of rules */

	uint8_t match:1,	  /* match all rules or any of them */
		cond_left:3,	  /* left condition */
	        cond_right:3,	  /* right condition */
		reserved:1;
};

void c_inst_init(struct c_instance& inst, int blocks, int matrix_width);
void c_inst_cleanup(struct c_instance& inst);
int* c_create_dev_rules(struct c_instance& inst);
#endif /* C_INSTANCE_H_ */
