#include <limits.h>
#include <float.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "instance.h"

#include "evo_rating.h"
#include "evo_memory.cu"

__shared__ int MHEIGHT;
__shared__ int MWIDTH;

__shared__ double res[MATRIX_HEIGHT][MATRIX_WIDTH];
__shared__ double shrd_rating;
__shared__ double matrix_form;

__device__ inline void eval_set_res_matrix_to_zero()
{
	res[threadIdx.y][threadIdx.x] = 0.;
}

__device__ inline void eval_set_res_matrix_to_identity()
{
	if(threadIdx.x != threadIdx.y) {
		res[threadIdx.y][threadIdx.x] = 0.;
	} else {
		res[threadIdx.y][threadIdx.x] = 1.;
	}
}

__device__ inline void eval_copy_matrix_to_res(const struct instance * const inst,
					       struct memory * const mem,
		    	    	    	       const int cmatrix)
{
	const int cstart = mem->c_zero + cmatrix * MWIDTH;
	res[ty][tx] = C_ROW(ty)[cstart + tx];
}

__device__ void eval_mul_inplace(const struct instance * const inst,
				 struct memory         * const mem,
				 const int cmatrix)
{
	const int rows = MHEIGHT;
	const int cstart = mem->c_zero  + cmatrix * MWIDTH;

	double y, t;
	double c = 0;
	double sum = 0;

	/* result rows */
	#pragma unroll
	for(int i = 0; i < rows; i++) {
		y = __dmul_rn(res[ty][i], C_ROW(i)[cstart + tx]) - c;
		t = __dadd_rn(sum, y);
		c = (t - sum) - y;
		sum = t;
	}

	__syncthreads();
	res[ty][tx] = sum;
	__syncthreads();
}

__device__ const int* eval_interpret_rule(const struct instance * const inst,
				    	  struct memory		* const mem,
				    	  const int             * rule)
{
	if(*rule == MUL_SEP || *rule == MUL_MARK)
		return rule;

	/*
	 * all multiplications are inplace,
	 * so we copy the first matrix to our result
	 */
	eval_copy_matrix_to_res(inst, mem, *rule);
	rule++;

	__syncthreads();

	for(; *rule != MUL_SEP && *rule != MUL_MARK; rule++) {
		eval_mul_inplace(inst, mem, *rule);
	}

	return rule;
}

__device__ void evo_result_rating(const struct instance * const inst,
				  struct memory         * const mem,
				  const int		        rule_type)
{
	const int rows = MHEIGHT - 1;
	const int cols = MWIDTH  - 1;
	double rating = 0.;

	const double penalty = 1e9;

        if(rule_type != MUL_MARK && ty == 0 && tx == 0) {
                switch(inst->cond_right) {
                case COND_UPPER_LEFT:
                        if((R_ROW(0)[mem->r_zero] - res[0][0]) < 1.f)
                                rating += penalty;
                        break;
                case COND_UPPER_RIGHT:
                        if((R_ROW(0)[mem->r_zero + cols] - res[0][cols]) < 1.f)
                                rating += penalty;
                        break;
                case COND_UPPER_LEFT_LOWER_RIGHT:
                        if((R_ROW(0)[mem->r_zero] - res[0][0]) < 1.f)
                                rating += penalty;

                        if((R_ROW(rows)[mem->r_zero + cols] - res[rows][cols]) < 1.f)
                                rating += penalty;
                        break;
                default:
                        rating += 2*penalty;
                        break;
                }

                if(rule_type == MUL_SEP && inst->match == MATCH_ANY) {
                        if(rating == 0.)
                                matrix_form = 0.;

                        rating = 0.;
                }
        }
	__syncthreads();
	// keep only negative numbers
	res[ty][tx] = fabs(min(R_ROW(ty)[mem->r_zero + tx] - res[ty][tx], 0.));
	res[ty][tx] = __dmul_rn(res[ty][tx], res[ty][tx]);
	__syncthreads();

	double c = 0.0;
	double y, t;
	double sum;

	//only lines are processed
	if(tx == 0) {
		sum = res[ty][0];

		for(int i = 1; i < MWIDTH; i++) {
			y = res[ty][i] - c;
			t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}

		res[ty][0] = sum;
	}
	__syncthreads();

	if(tx == 0 && ty == 0) {
		for(int i = 0; i < MHEIGHT; i++) {
			y = res[i][0] - c;
			t = rating + y;
			c = (t - rating) - y;
			rating = t;
		}

		shrd_rating += sqrtf(rating);
	}
	__syncthreads();
}

__device__ void evo_init_mem2(const struct instance* const inst,
			      struct memory * const mem)
{
	evo_init_mem(inst, mem);
	/*
	 * each block represent one child which has a
	 * defined pos in the matrix
	 */
	mem->c_zero = inst->width_per_inst * blockIdx.y;
	mem->c_end  = inst->width_per_inst * (blockIdx.y + 1);

	mem->r_zero = blockIdx.y * MWIDTH;
	mem->r_end  = mem->r_zero + MWIDTH;
}

__global__ void evo_calc_res(struct instance * const inst)
{
	__shared__ struct memory res_mem;

	const int* end = inst->rules + inst->rules_len - 1;
	const int* rules = inst->rules;
	struct memory * const mem = &res_mem;

	if(tx == 0 && ty == 0) {
		MHEIGHT = inst->dim.matrix_height;
		MWIDTH  = inst->dim.matrix_width;
		shrd_rating = 0.;
		matrix_form = 1e9;
		evo_init_mem2(inst, &res_mem);
	}

	__syncthreads();
	uint8_t cur_rule = 0;

	do {
		eval_set_res_matrix_to_identity();
                __syncthreads();

                const int rule_type = *rules;
		rules++;
		rules = eval_interpret_rule(inst , mem, rules);

                __syncthreads();
                CR_ROW(ty)[mem->r_zero + tx] = res[ty][tx];
                MDEBUG(inst, cur_rule, 0, res[ty][tx]);
		eval_set_res_matrix_to_identity();
                __syncthreads();

		rules++;
		rules = eval_interpret_rule(inst , mem, rules);
                MDEBUG(inst, cur_rule, 1, res[ty][tx]);
		__syncthreads();
		evo_result_rating(inst, mem, rule_type);
		cur_rule++;
		__syncthreads();
	} while(rules != end);

	if(tx == 0 && ty == 0) {
		if(inst->match == MATCH_ANY)
			shrd_rating += matrix_form;

		res_mem.c_rat[2 * blockIdx.y]     = shrd_rating;
		res_mem.c_rat[2 * blockIdx.y + 1] = blockIdx.y;
	}
	__syncthreads();
}
