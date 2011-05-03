/*
 * instance.h
 *
 *  Created on: Apr 8, 2011
 *      Author: tkalbitz
 */

#ifndef INSTANCE_H_
#define INSTANCE_H_

#include <stdio.h>
#include <stdint.h>

#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { cudaError_t xxs = (x); \
	if((xxs) != cudaSuccess) { \
		fprintf(stderr, "Error '%s' at %s:%d\n", cudaGetErrorString(xxs),__FILE__,__LINE__); \
		exit(EXIT_FAILURE);}} while(0)

#define MATCH_ALL 0
#define MATCH_ANY 1

#define COND_UPPER_LEFT  0
#define COND_UPPER_RIGHT 1
#define COND_UPPER_LEFT_LOWER_RIGHT 2

struct dimension 
{
	int blocks;
	int childs;
	int parents;
	int matrix_width;
	int matrix_height;
};

#define MUL_SEP  -1
#define MUL_STOP -2

struct instance {
	struct dimension dim;	   /* dimension of the matrix */
	cudaPitchedPtr dev_parent; /* device memory for all parents */
	cudaPitchedPtr dev_child;  /* device memory for all childs */ 
	cudaPitchedPtr dev_res;    /* result of the evaluation */
	cudaPitchedPtr dev_crat;   /* rating of the childs */
	cudaPitchedPtr dev_prat;   /* rating of the parents */
	curandState *rnd_states;   /* random number generator states */

	int num_matrices;          /* number of matrices of the problem */
	size_t width_per_inst;     /* how many elements are stored for each thread */

	cudaExtent dev_parent_ext; /* extent for parents */
	cudaExtent dev_child_ext;  /* extent for childs */
	cudaExtent dev_res_ext;    /* extend for result */
	cudaExtent dev_crat_ext;   /* extend for child rating */
	cudaExtent dev_prat_ext;   /* extend for parent rating */

	float delta;		/* numbers in the matrices are multiple the amount */
	int*   rules;		/* rules that must be matched */
	size_t rules_len;       /* number of elements in rules */
	size_t rule_count;	/* how many rules exists? */

	int rounds;             /* how many rounds we worked on that problem? */
	int res_block;		/* in which block is the result */
	int res_parent;		/* which parent is the result */

	uint8_t	match:1,	/* match all rules or any of them */
		cond_left:3,	/* left condition */
	        cond_right:3,	/* right condition */
		reserved:1;
};

#endif /* INSTANCE_H_ */
