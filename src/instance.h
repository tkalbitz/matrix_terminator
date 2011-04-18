/*
 * instance.h
 *
 *  Created on: Apr 8, 2011
 *      Author: tkalbitz
 */

#ifndef INSTANCE_H_
#define INSTANCE_H_

#include <stdint.h>
#include <cuda.h>

#define MATCH_ALL 0
#define MATCH_ANY 1

#define COND_UPPER_LEFT  0
#define COND_UPPER_RIGHT 1
#define COND_UPPER_LEFT_LOWER_RIGHT 2

struct dimension 
{
	int blocks;
	int threads;
	int childs;
	int matrix_width;
	int matrix_height;
};

struct instance {
	struct dimension dim;	   /* dimension of the matrix */
	cudaPitchedPtr dev_parent; /* device memory for all parents */
	cudaPitchedPtr dev_child;  /* device memory for all childs */ 
	curandState *rnd_states;   /* random number generator states */

	cudaExtent dev_parent_ext; /* extent for parents */
	cudaExtent dev_child_ext;  /* extent for childs */

	double delta;		/* numbers in the matrices are multiple the amount */
	char*  rules;		/* rules that must be matched */

	uint8_t	match:1,	/* match all rules or any of them */
		cond_left:3,	/* left condition */
	        cond_right:3,	/* right condition */
		reserved:1;
};

#endif /* INSTANCE_H_ */
