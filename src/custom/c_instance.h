/*
 * c_instance.h
 *
 *  Created on: Feb 8, 2012
 *      Author: tkalbitz
 */

#ifndef C_INSTANCE_H_
#define C_INSTANCE_H_

#include <stdio.h>
#include <stdint.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "c_config.h"

#define CUDA_CALL(x) do { cudaError_t xxs = (x); \
	if((xxs) != cudaSuccess) { \
		fprintf(stderr, "Error '%s' at %s:%d\n", cudaGetErrorString(xxs),__FILE__,__LINE__); \
		exit(EXIT_FAILURE);}} while(0)

#define tx (threadIdx.x)
#define ty (threadIdx.y)
#define bx (blockIdx.x)
#define by (blockIdx.y)

#define MATCH_ALL 0
#define MATCH_ANY 1

#define COND_UPPER_LEFT  0
#define COND_UPPER_RIGHT 1
#define COND_UPPER_LEFT_LOWER_RIGHT 2

#define MUL_SEP       -1

#define PARAM_COUNT    3

struct c_instance
{
	int num_matrices;         /* number of matrices of the problem */
	int mdim;                 /* dimension of the matrix */
	double parent_max;        /* maximum of a matrix position */

	int icount;               /* number of instances */

	int itotal;               /* number of bytes for all instances */
	int stotal;               /* number of bytes for all search instances */

	int width_per_inst;       /* number elements for one instance */
	int width_per_matrix;     /* number elements for one matrix */

	double* instances;        /* all instances */
	double* rating;           /* rating of the instances */

	double* sinstances;       /* search space instances */
	double* srating;          /* rating of search space instances */

	double* tmp;              /* copy of the current element */
	double* tmprat;           /* copy of the current element */

	double* res;              /* tmp space for the rating function */
	double* best;             /* best rating of the different blocks */
	int*    best_idx;         /* where is the best located */

	curandState *rnd_states;  /* random number generator states */

	double delta;             /* numbers in the matrices are multiple the amount */
	int*   rules;             /* rules that must be matched */
	size_t rules_len;         /* number of elements in rules */
	size_t rules_count;       /* number of rules */

	uint8_t match:1,	 /* match all rules or any of them */
		cond_left:3,	 /* left condition */
	        cond_right:3,	 /* right condition */
		reserved:1;
};

void c_inst_init(struct c_instance& inst, int matrix_width);
void c_inst_cleanup(struct c_instance& inst,
		    struct c_instance* dev_inst);

struct c_instance*
c_inst_create_dev_inst(struct c_instance& inst, int** dev_rules);
#endif /* C_INSTANCE_H_ */
