#include <limits.h>
#include <float.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "config.h"
#include "instance.h"

__shared__ double res[2][MATRIX_HEIGHT][MATRIX_WIDTH];
//__shared__ double child[MATRIX_HEIGHT][MATRIX_WIDTH];

__device__ void eval_set_res_matrix_to_zero(struct instance *inst,
					    struct memory   *mem)
{
	res[0][threadIdx.y][threadIdx.x] = 0.;
	res[1][threadIdx.y][threadIdx.x] = 0.;
}

__device__ void eval_copy_matrix_to_res(struct instance *inst,
		    	    	    	struct memory *mem,
		    	    	    	const int cmatrix,
		    	    	    	const int rmatrix)
{
	const int cstart = mem->c_zero + cmatrix * MATRIX_WIDTH;
	res[rmatrix][threadIdx.y][threadIdx.x] = C_ROW(threadIdx.y)[cstart + threadIdx.x];
}

__device__ void eval_mul_inplace(struct instance *inst,
				 struct memory *mem,
				 const int cmatrix,
				 const int rmatrix)
{
	const int rows = MATRIX_HEIGHT;
	const int cstart = mem->c_zero  + cmatrix * inst->dim.matrix_width;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	//child[ty][tx] = C_ROW(ty)[cstart + tx];
	//__syncthreads();

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

__device__ int* eval_interpret_rule(struct instance *inst,
				    struct memory   *mem,
				    int*   rule,
				    const int rmatrix)
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

__device__ double get_max_value(struct instance *inst,
		  	  	struct memory   *mem)
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

__device__ void evo_result_rating(struct instance *inst,
				  struct memory   *mem)
{
	const int rows = MATRIX_HEIGHT - 1;
	const int cols = MATRIX_WIDTH  - 1;
	double rating = 0.;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	if(ty == 0 && tx == 0) {
		if(inst->cond_right == COND_UPPER_LEFT) {
			if((res[0][0][0] - res[1][0][0]) < 1.f)
				rating += 0.5;
		} else if(inst->cond_right == COND_UPPER_RIGHT) {
			if((res[0][0][cols] - res[1][0][cols]) < 1.f)
				rating += 0.5;
		} else if(inst->cond_right == COND_UPPER_LEFT_LOWER_RIGHT) {
			if((res[0][0][0] - res[1][0][0]) < 1.f)
				rating += 0.5;

			if((res[0][rows][cols] - res[1][rows][cols]) < 1.f)
				rating += 0.5;
		} else {
			rating += 5;
		}
	}

	__syncthreads();

	// keep only negative numbers
	res[0][ty][tx] = fabs(min(res[0][ty][tx] - res[1][ty][tx], 0.));

	double max_value = get_max_value(inst, mem);
	max_value = (max_value == 0.0 ? 1 : max_value); // div. by zero is evil...
	res[0][ty][tx] /= max_value;

	//only lines are processed
	if(tx != 0)
		return;

//	res[0][ty][0] *= res[0][ty][0];
	for(int i = 1; i < MATRIX_WIDTH; i++) {
		res[0][ty][0] += res[0][ty][i]; // * res[0][ty][i];
	}

	if(ty != 0)
		return;

	for(int i = 0; i < MATRIX_HEIGHT; i++) {
		rating += res[0][i][0];
	}

//	rating = sqrt(rating);

	if(inst->match == MATCH_ALL) {
		shrd_rating += rating;
	} else {
		shrd_rating = min(shrd_rating, rating);
	}
}

__device__ void evo_init_mem2(struct instance* inst, struct memory *mem)
{
	char* p_dev_ptr = (char*)inst->dev_parent.ptr;
	size_t p_pitch = inst->dev_parent.pitch;
	size_t p_slice_pitch = p_pitch * inst->dim.matrix_height;
	char* p_slice = p_dev_ptr + blockIdx.x /* z */ * p_slice_pitch;

	char* c_dev_ptr = (char*)inst->dev_child.ptr;
	size_t c_pitch = inst->dev_child.pitch;
	size_t c_slice_pitch = c_pitch * inst->dim.matrix_height;
	char* c_slice = c_dev_ptr + blockIdx.x /* z */ * c_slice_pitch;

	char* r_dev_ptr = (char*)inst->dev_res.ptr;
	size_t r_pitch = inst->dev_res.pitch;
	size_t r_slice_pitch = r_pitch * inst->dim.matrix_height;
	char* r_slice = r_dev_ptr + blockIdx.x /* z */ * r_slice_pitch;

	mem->p_pitch = p_pitch;
	mem->p_slice = p_slice;
	mem->c_pitch = c_pitch;
	mem->c_slice = c_slice;
	mem->r_pitch = r_pitch;
	mem->r_slice = r_slice;

	/*
	 * each thread represent one child which has a
	 * defined pos in the matrix
	 */
	mem->c_zero = inst->width_per_inst * blockIdx.y;
	mem->c_end  = inst->width_per_inst * (blockIdx.y + 1);

	mem->r_zero1 = blockIdx.y * 2 * inst->dim.matrix_width;
	mem->r_end1  = mem->r_zero1 + inst->dim.matrix_width;
	mem->r_zero2 = mem->r_zero1 + inst->dim.matrix_width;
	mem->r_end2  = mem->r_zero2 + inst->dim.matrix_width;

	char* t_dev_ptr = (char*)inst->dev_crat.ptr;
	size_t t_pitch = inst->dev_crat.pitch;
	size_t t_slice_pitch = t_pitch * 1;
	char* t_slice = t_dev_ptr + blockIdx.x /* z */ * t_slice_pitch;
	mem->c_rat = (double*) (t_slice + 0 * t_pitch);

	t_dev_ptr = (char*)inst->dev_prat.ptr;
	t_pitch = inst->dev_prat.pitch;
	t_slice_pitch = t_pitch * 1;
	t_slice = t_dev_ptr + blockIdx.x /* z */ * t_slice_pitch;
	mem->p_rat = (double*) (t_slice + 0 * t_pitch);
}

__global__ void evo_calc_res(struct instance *inst)
{
	const int* end = inst->rules + inst->rules_len - 1;
	int* rules = inst->rules;

	if(threadIdx.x == 0 && threadIdx.y == 0) {
		evo_init_mem2(inst, &res_mem);

		shrd_rating = 0.;
		if(inst->match == MATCH_ANY) {
			shrd_rating = FLT_MAX;
		}
	}

	__syncthreads();

	do {
		eval_set_res_matrix_to_zero(inst, &res_mem);
		__syncthreads();

		rules++;
		rules = eval_interpret_rule(inst , &res_mem, rules, 0);

		rules++;
		rules = eval_interpret_rule(inst , &res_mem, rules, 1);

		evo_result_rating(inst, &res_mem);
		__syncthreads();
	} while(rules != end);

	/* copy to test the result */
//	CR_ROW(threadIdx.y)[res_mem.r_zero1 + threadIdx.x] = res[0][threadIdx.y][threadIdx.x];
//	CR_ROW(threadIdx.y)[res_mem.r_zero2 + threadIdx.x] = res[1][threadIdx.y][threadIdx.x];

	__syncthreads();

	if(threadIdx.x == 0 && threadIdx.y == 0) {
		res_mem.c_rat[2 * blockIdx.y]     = shrd_rating;
		res_mem.c_rat[2 * blockIdx.y + 1] = blockIdx.y;
	}
}
