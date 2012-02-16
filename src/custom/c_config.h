/*
 * c_config.h
 *
 *  Created on: Feb 8, 2012
 *      Author: tkalbitz
 */

#ifndef C_CONFIG_H_
#define C_CONFIG_H_

#define PARENT_MAX 10.

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
