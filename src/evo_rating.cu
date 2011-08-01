#include <limits.h>
#include <float.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "instance.h"

#include "evo_rating.h"
#include "evo_memory.cu"

__shared__ double res[2][MATRIX_HEIGHT][MATRIX_WIDTH];

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

__device__ inline void eval_copy_matrix_to_res(struct memory * const mem,
		    	    	    	       const int cmatrix,
		    	    	    	       const int rmatrix)
{
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int cstart = mem->c_zero + cmatrix * MATRIX_WIDTH;

	res[rmatrix][ty][tx] = C_ROW(ty)[cstart + tx];
}

__device__ void eval_mul_inplace(const struct instance * const inst,
				 struct memory         * const mem,
				 const int cmatrix,
				 const int rmatrix)
{
	const int rows = MATRIX_HEIGHT;
	const int cstart = mem->c_zero  + cmatrix * inst->dim.matrix_width;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

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
	eval_copy_matrix_to_res(mem, *rule, rmatrix);
	rule++;

	__syncthreads();

	for(; *rule != MUL_SEP; rule++) {
		eval_mul_inplace(inst, mem, *rule, rmatrix);
	}

	return rule;
}

__shared__ struct memory res_mem;
__shared__ double shrd_rating;

__device__ double get_max_value()
{
	double my_max;
	__shared__ double max_value[MATRIX_HEIGHT];

	if(threadIdx.x == 0) {
		my_max = res[0][threadIdx.y][0];
		for(int i = 1; i < MATRIX_WIDTH; i++) {
			my_max = max(my_max, res[0][threadIdx.y][i]);
		}
		max_value[threadIdx.y] = my_max;

		__syncthreads();
		if(threadIdx.y == 0) {
			for(int k = 1; k < MATRIX_HEIGHT; k++) {
				max_value[0] = max(max_value[0], max_value[k]);
			}
		}
		__syncthreads();
	}

	return max_value[0];
}

__device__ void evo_result_rating(const struct instance * const inst,
				  struct memory         * const mem)
{
	const int rows = MATRIX_HEIGHT - 1;
	const int cols = MATRIX_WIDTH  - 1;
	double rating = 0.;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

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
	}

	__syncthreads();
	// keep only negative numbers
	res[0][ty][tx] = fabs(min(res[0][ty][tx] - res[1][ty][tx], 0.));

//	double max_value = get_max_value();
//	max_value = (max_value == 0 ? 1 : max_value); // div. by zero is evil...
//	res[0][ty][tx] /= max_value;
	__syncthreads();

	//only lines are processed
	if(tx != 0)
		return;

	for(int i = 1; i < MATRIX_WIDTH; i++) {
		res[0][ty][0] += res[0][ty][i];
	}

	if(ty != 0)
		return;

	for(int i = 0; i < MATRIX_HEIGHT; i++) {
		rating += res[0][i][0];
	}

	shrd_rating += rating;
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
	mem->r_zero1 = blockIdx.y * 2 * inst->dim.matrix_width;
	mem->r_end1  = mem->r_zero1 + inst->dim.matrix_width;
	mem->r_zero2 = mem->r_zero1 + inst->dim.matrix_width;
	mem->r_end2  = mem->r_zero2 + inst->dim.matrix_width;
#endif
}

__global__ void evo_calc_res(struct instance * const inst)
{
	const int* end = inst->rules + inst->rules_len - 1;
	const int* rules = inst->rules;

	char* const r_dev_ptr = (char*)inst->dev_rules.ptr;
        const size_t r_pitch = inst->dev_rules.pitch;
        const size_t r_slice_pitch = r_pitch * inst->dim.childs * inst->dim.parents;
        char* const r_slice = r_dev_ptr + blockIdx.x /* z */ * r_slice_pitch;
        uint8_t* const active_rules = (uint8_t*) (r_slice + blockIdx.y * r_pitch);

	if(threadIdx.x == 0 && threadIdx.y == 0) {
		evo_init_mem2(inst, &res_mem);
		shrd_rating = 0.;
	}

	__syncthreads();
	uint8_t cur_rule = 0;

	do {
		/* ignore matched rules */	
		if(inst->match == MATCH_ANY && !active_rules[cur_rule]) {
			rules++;
			while(*rules != MUL_SEP) {
				rules++;
			}
			rules++;
			while(*rules != MUL_SEP) {
				rules++;
			}
			cur_rule++;
			__syncthreads();
			continue;
		}

		eval_set_res_matrix_to_identity();

		rules++;
		rules = eval_interpret_rule(inst , &res_mem, rules, 0);

		rules++;
		rules = eval_interpret_rule(inst , &res_mem, rules, 1);

		const double old_rating = shrd_rating;
		evo_result_rating(inst, &res_mem);
		__syncthreads();

		if(inst->match == MATCH_ANY && old_rating == shrd_rating) {
			active_rules[cur_rule] = 0;
		}

		#ifdef DEBUG
		if(shrd_rating == 0.) {
			struct memory *mem = &res_mem;
			for(int i = 0; i < inst->num_matrices; i++) {
				R_ROW(ty)[tx + i * MATRIX_WIDTH] =
						C_ROW(ty)[res_mem.c_zero + i * MATRIX_WIDTH + tx];
			}

			inst->res_child_block = blockIdx.x;
			inst->res_child_idx   = blockIdx.y;
		}
		#endif

		cur_rule++;
		__syncthreads();
	} while(rules != end);

	__syncthreads();

	if(threadIdx.x == 0 && threadIdx.y == 0) {
		res_mem.c_rat[2 * blockIdx.y]     = shrd_rating;
		res_mem.c_rat[2 * blockIdx.y + 1] = blockIdx.y;
	}
}
