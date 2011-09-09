#include <limits.h>
#include <float.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "instance.h"

#include "evo_rating.h"
#include "evo_memory.cu"

__shared__ double res[2][MATRIX_HEIGHT][MATRIX_WIDTH];
__shared__ int MHEIGHT;
__shared__ int MWIDTH;


__device__ inline void eval_set_res_matrix_to_zero()
{
	res[0][threadIdx.y][threadIdx.x] = 0.;
	res[1][threadIdx.y][threadIdx.x] = 0.;
}

__device__ inline void eval_set_res_matrix_to_identity()
{
	if(threadIdx.x != threadIdx.y) {
		res[0][threadIdx.y][threadIdx.x] = 0.;
		res[1][threadIdx.y][threadIdx.x] = 0.;
	} else {
		res[0][threadIdx.y][threadIdx.x] = 1.;
		res[1][threadIdx.y][threadIdx.x] = 1.;
	}
}

__device__ inline void eval_copy_matrix_to_res(const struct instance * const inst,
					       struct memory * const mem,
		    	    	    	       const int cmatrix,
		    	    	    	       const int rmatrix)
{
	const int cstart = mem->c_zero + cmatrix * MWIDTH;

	res[rmatrix][ty][tx] = C_ROW(ty)[cstart + tx];
}

__device__ void eval_mul_inplace(const struct instance * const inst,
				 struct memory         * const mem,
				 const int cmatrix,
				 const int rmatrix)
{
	const int rows = MHEIGHT;
	const int cstart = mem->c_zero  + cmatrix * MWIDTH;

	double tmp = 0;

	/* result rows */
	#pragma unroll
	for(int i = 0; i < rows; i++) {
		tmp += res[rmatrix][ty][i] * C_ROW(i)[cstart + tx];
	}

	__syncthreads();
	res[rmatrix][ty][tx] = tmp;
	__syncthreads();
}

__device__ const int* eval_interpret_rule(const struct instance * const inst,
				    	  struct memory		* const mem,
				    	  const int* rule,
				    	  const int  rmatrix)
{
	if(*rule == MUL_SEP)
		return rule;

	/*
	 * all multiplications are inplace,
	 * so we copy the first matrix to our result
	 */
	eval_copy_matrix_to_res(inst, mem, *rule, rmatrix);
	rule++;

	__syncthreads();

	for(; *rule != MUL_SEP; rule++) {
		eval_mul_inplace(inst, mem, *rule, rmatrix);
	}

	return rule;
}

__shared__ struct memory res_mem;
__shared__ double shrd_rating;
__shared__ double matrix_form;

//__device__ double get_max_value(const struct instance * const inst)
//{
//	double my_max;
//	__shared__ double max_value[MATRIX_HEIGHT];
//
//	if(threadIdx.x == 0) {
//		my_max = res[0][threadIdx.y][0];
//		for(int i = 1; i < MWIDTH; i++) {
//			my_max = max(my_max, res[0][threadIdx.y][i]);
//		}
//		max_value[threadIdx.y] = my_max;
//
//		__syncthreads();
//		if(threadIdx.y == 0) {
//			for(int k = 1; k < MHEIGHT; k++) {
//				max_value[0] = max(max_value[0], max_value[k]);
//			}
//		}
//		__syncthreads();
//	}
//
//	return max_value[0];
//}

__device__ void evo_result_rating(const struct instance * const inst,
				  struct memory         * const mem)
{
	const int rows = MHEIGHT - 1;
	const int cols = MWIDTH  - 1;
	double rating = 0.;

	const double penalty = 1e9;

	if(ty == 0 && tx == 0) {
		switch(inst->cond_right) {
		case COND_UPPER_LEFT:
			if((res[0][0][0] - res[1][0][0]) < 1.f)
				rating += penalty;
			break;
		case COND_UPPER_RIGHT:
			if((res[0][0][cols] - res[1][0][cols]) < 1.f)
				rating += penalty;
			break;
		case COND_UPPER_LEFT_LOWER_RIGHT:
			if((res[0][0][0] - res[1][0][0]) < 1.f)
				rating += penalty;

			if((res[0][rows][cols] - res[1][rows][cols]) < 1.f)
				rating += penalty;
			break;
		default:
			rating += 2*penalty;
			break;
		}

		if(inst->match == MATCH_ANY) {
			if(rating == 0.)
				matrix_form = 0.;

			rating = 0.;
		}
	}

	__syncthreads();
	// keep only negative numbers
	res[0][ty][tx] = fabs(min(res[0][ty][tx] - res[1][ty][tx], 0.));
	res[0][ty][tx] *= res[0][ty][tx];
//	double max_value = get_max_value();
//	max_value = (max_value == 0 ? 1 : max_value); // div. by zero is evil...
//	res[0][ty][tx] /= max_value;
	__syncthreads();

	//only lines are processed
	if(tx != 0)
		return;

	double c = 0.0;
	double y, t;
	double sum = res[0][ty][0];

	for(int i = 1; i < MWIDTH; i++) {
		y = res[0][ty][i] - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}

	res[0][ty][0] = sum;

	if(ty != 0)
		return;

	for(int i = 0; i < MHEIGHT; i++) {
		y = res[0][i][0] - c;
		t = rating + y;
		c = (t - rating) - y;
		rating = t;

	}

	shrd_rating += sqrtf(rating);
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

#ifdef DEBUG
	mem->r_zero1 = blockIdx.y * 2 * MWIDTH;
	mem->r_end1  = mem->r_zero1 + MWIDTH;
	mem->r_zero2 = mem->r_zero1 + MWIDTH;
	mem->r_end2  = mem->r_zero2 + MWIDTH;
#endif
}

__global__ void evo_calc_res(struct instance * const inst)
{
	const int* end = inst->rules + inst->rules_len - 1;
	const int* rules = inst->rules;

	if(tx == 0 && ty == 0) {
		evo_init_mem2(inst, &res_mem);
		shrd_rating = 0.;
		matrix_form = 1e9;
		MHEIGHT = inst->dim.matrix_height;
		MWIDTH  = inst->dim.matrix_width;
	}

	__syncthreads();
	uint8_t cur_rule = 0;

	do {
		eval_set_res_matrix_to_identity();

		rules++;
		rules = eval_interpret_rule(inst , &res_mem, rules, 0);

		rules++;
		rules = eval_interpret_rule(inst , &res_mem, rules, 1);

		evo_result_rating(inst, &res_mem);
		__syncthreads();

		#ifdef DEBUG
		if(shrd_rating == 0.) {
			struct memory *mem = &res_mem;
			for(int i = 0; i < inst->num_matrices; i++) {
				R_ROW(ty)[tx + i * MWIDTH] =
						C_ROW(ty)[res_mem.c_zero + i * MWIDTH + tx];
			}

			inst->res_child_block = blockIdx.x;
			inst->res_child_idx   = blockIdx.y;
		}
		#endif

		cur_rule++;
		__syncthreads();
	} while(rules != end);

	__syncthreads();

	if(tx == 0 && ty == 0) {
		if(inst->match == MATCH_ANY)
			shrd_rating += matrix_form;

		res_mem.c_rat[2 * blockIdx.y]     = shrd_rating;
		res_mem.c_rat[2 * blockIdx.y + 1] = blockIdx.y;
	}
}
