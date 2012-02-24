/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef PSO_INSTANCE_H_
#define PSO_INSTANCE_H_

#include <stdio.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { cudaError_t xxs = (x); \
	if((xxs) != cudaSuccess) { \
		fprintf(stderr, "Error '%s' at %s:%d\n", cudaGetErrorString(xxs),__FILE__,__LINE__); \
		exit(EXIT_FAILURE);}} while(0)

#define tx (threadIdx.x)
#define ty (threadIdx.y)
#define bx (blockIdx.x)

#define MATCH_ALL 0
#define MATCH_ANY 1

#define COND_UPPER_LEFT  0
#define COND_UPPER_RIGHT 1
#define COND_UPPER_LEFT_LOWER_RIGHT 2

struct pso_dimension
{
	int blocks;
	int particles;
	int matrix_width;
	int matrix_height;
};

#define MUL_SEP       -1
#define MUL_MARK      -2

#define PARAM_COUNT    3

struct pso_instance
{
	struct pso_dimension dim;     /* dimension of the matrix */
	double* particle;
	double* particle_lbest;
	double* particle_gbest;
	double* res;
	double* prat;
	double* lbrat;
	double* gbrat;
	double* rat_tmp;

	curandState *rnd_states;      /* random number generator states */

	int num_matrices;          /* number of matrices of the problem */
	int width_per_inst;        /* how many elements are stored for each thread */
	int width_per_line;        /* how many elements are stored for problem */
	int width_per_matrix;      /* how many elements per matrix */
	int total;		   /* complete length of all elements */
	int* s;                    /* current group size */

	double delta;                 /* numbers in the matrices are multiple the amount */
	int*   rules;                 /* rules that must be matched */
	size_t rules_len;             /* number of elements in rules */
	size_t rules_count;           /* number of rules */

	double* gb_best;               /* rating of the global best particles */
	double* gb_old;                /* rating of the global best particles */
	int* col_permut;              /* permutation of the columns */
	int* lbest_idx;               /* permutation of the columns */


	unsigned int res_block;       /* in which block is the result */
	unsigned int res_parent;      /* which parent is the result */

	unsigned int res_child_block; /* in which block is the result */
	unsigned int res_child_idx;   /* which parent is the result */

	double parent_max;

	uint8_t match:1,	/* match all rules or any of them */
		cond_left:3,	/* left condition */
	        cond_right:3,	/* right condition */
		reserved:1;
};


void pso_inst_init(struct pso_instance* const inst, int matrix_width);
void pso_inst_cleanup(struct pso_instance * const inst,
		      struct pso_instance * const dev_inst);

struct pso_instance*
pso_inst_create_dev_inst(struct pso_instance *inst, int** dev_rules);

void pso_inst_copy_dev_to_host(struct pso_instance * const dev,
				struct pso_instance * const host);

#endif /* PSO_INSTANCE_H_ */
