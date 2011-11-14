#ifndef INSTANCE_H_
#define INSTANCE_H_

#include <stdio.h>
#include <stdint.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "config.h"

#define CUDA_CALL(x) do { cudaError_t xxs = (x); \
	if((xxs) != cudaSuccess) { \
		fprintf(stderr, "Error '%s' at %s:%d\n", cudaGetErrorString(xxs),__FILE__,__LINE__); \
		exit(EXIT_FAILURE);}} while(0)

#define tx (threadIdx.x)
#define ty (threadIdx.y)

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

#define MUL_SEP       -1

struct instance {
	struct dimension dim;	   /* dimension of the matrix */
	cudaPitchedPtr dev_parent; /* device memory for all parents */
	cudaPitchedPtr dev_child;  /* device memory for all childs */ 
	cudaPitchedPtr dev_res;    /* result of the evaluation */
	cudaPitchedPtr dev_crat;   /* rating of the childs */
	cudaPitchedPtr dev_prat;   /* rating of the parents */
	cudaPitchedPtr dev_debug;  /* debug infos */

	curandState *rnd_states;   /* random number generator states */

	cudaPitchedPtr dev_sparam; /* strategy param */
	cudaExtent dev_sparam_ext; /* extend for parent strategy param */

	cudaPitchedPtr dev_psparam; /* parent strategy param */
	cudaExtent dev_psparam_ext; /* extend for parent strategy param */


	int num_matrices;          /* number of matrices of the problem */
	size_t width_per_inst;     /* how many elements are stored for each thread */

	cudaExtent dev_parent_ext; /* extent for parents */
	cudaExtent dev_child_ext;  /* extent for childs */
	cudaExtent dev_res_ext;    /* extend for result */
	cudaExtent dev_crat_ext;   /* extend for child rating */
	cudaExtent dev_prat_ext;   /* extend for parent rating */
	cudaExtent dev_debug_ext;  /* extend for debug infos */

	double delta;		/* numbers in the matrices are multiple the amount */
	int*   rules;		/* rules that must be matched */
	size_t rules_len;       /* number of elements in rules */
	size_t rules_count;     /* number of rules */

	unsigned int res_block; /* in which block is the result */
	unsigned int res_parent;/* which parent is the result */

	unsigned int res_child_block; /* in which block is the result */
	unsigned int res_child_idx;   /* which parent is the result */

	double parent_max;

	uint8_t match:1,	/* match all rules or any of them */
		cond_left:3,	/* left condition */
	        cond_right:3,	/* right condition */
		reserved:1;
};

void inst_init(struct instance* const inst, int matrix_width);
void inst_cleanup(struct instance * const inst,
		      struct instance * const dev_inst);

struct instance* inst_create_dev_inst(struct instance *inst, int** dev_rules);
void inst_copy_dev_to_host(struct instance * const dev,
		           struct instance * const host);
int get_evo_threads(const struct instance * const inst);

#endif /* INSTANCE_H_ */
