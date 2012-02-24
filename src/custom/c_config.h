/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef C_CONFIG_H_
#define C_CONFIG_H_

#define PARENT_MAX 10.

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

#ifdef __CDT_PARSER__
#ifndef CDT_WORKAROUND
#define CDT_WORKAROUND
	#define BLOCKS 1
	#define MATRIX_WIDTH 5

	struct xxx_block {
		int x, y;
	};

	struct xxx_block blockIdx;
	struct xxx_block threadIdx;
	struct xxx_block blockDim;
	struct xxx_block gridDim;

	#define __syncthreads() /* */
	#define time(a) 0
	#define srand(a) /* */
	#define max(a, b) (a)
	#define min(a, b) (b)
	#define __dmul_rn(a, b) (a)
	#define __fmul_rn(a, b) (a)
	#define __dadd_rn(a, b) (b)
	#define __fadd_rn(a, b) (b)
	#define __double2uint_rn(a) (a)
	#define __float2uint_rn(a) (a)
	#define atomicAdd(x, y) 1
	#define atomicExch(x, y) 1

	#define no_argument		0
	#define required_argument	1
	#define optional_argument	2

	#define getopt_long(a, b, c, d, e) 1

#endif
#endif


#endif /* C_CONFIG_H_ */
