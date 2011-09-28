/*
 * pso_instance.h
 *
 *  Created on: Sep 28, 2011
 *      Author: tkalbitz
 */

#ifndef PSO_INSTANCE_H_
#define PSO_INSTANCE_H_

#include <stdio.h>
#include <stdint.h>

#include <cuda.h>
#include <curand_kernel.h>

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

struct pso_dimension
{
	int blocks;
	int particles;
	int matrix_width;
	int matrix_height;
};

#define MUL_SEP       -1

#define PARAM_COUNT    3

struct pso_instance
{
	struct pso_dimension dim;     /* dimension of the matrix */
	cudaPitchedPtr dev_particle;  /* device memory for all particles */
	cudaPitchedPtr dev_particle_lbest;  /* device memory for all particles */
	cudaPitchedPtr dev_particle_gbest;  /* device memory for all particles */
	cudaPitchedPtr dev_params;    /* device memory for all childs */
	cudaPitchedPtr dev_res;       /* result of the evaluation */
	cudaPitchedPtr dev_prat;      /* rating of the particles */

	curandState *rnd_states;      /* random number generator states */

	int num_matrices;             /* number of matrices of the problem */
	size_t width_per_inst;        /* how many elements are stored for each thread */

	cudaExtent dev_particle_ext;       /* extent for parents */
	cudaExtent dev_particle_lbest_ext; /* extent for parents */
	cudaExtent dev_particle_gbest_ext; /* extent for parents */
	cudaExtent dev_params_ext;          /* extent for parameter */
	cudaExtent dev_res_ext;             /* extend for result */
	cudaExtent dev_prat_ext;            /* extend for particle rating */

	double delta;                 /* numbers in the matrices are multiple the amount */
	int*   rules;                 /* rules that must be matched */
	size_t rules_len;             /* number of elements in rules */
	size_t rules_count;           /* number of rules */

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
