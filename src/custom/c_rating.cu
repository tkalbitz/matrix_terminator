#include <limits.h>
#include <float.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "c_rating.h"
#include "c_instance.h"

#define TIDX(cy, cx) (blockIdx.x * inst.width_per_matrix * blockDim.y + \
                      blockIdx.y * inst.width_per_matrix + \
                      cy * inst.mdim + cx)
#define TRES(cy, cx) inst.res[TIDX(cy, cx)]

#define RIDX(cy, cx) ((cy) * MWIDTH + (cx))
#define RES(cy, cx)  res[RIDX(cy, cx)]

__shared__ int MWIDTH;

__shared__ double res[MATRIX_WIDTH * MATRIX_WIDTH];
__shared__ double shrd_rating;
__shared__ double matrix_form;

__device__ inline void eval_set_res_matrix_to_zero()
{
	RES(ty, tx) = 0.;
}

__device__ inline void eval_set_res_matrix_to_identity()
{
	if(tx != ty) {
		RES(ty, tx) = 0.;
	} else {
		RES(ty, tx) = 1.;
	}
}

__device__ inline void eval_copy_matrix_to_res(const struct c_instance& inst,
		    	    	    	       const double           * matrix)
{
	RES(ty, tx) = matrix[RIDX(ty, tx)];
}

__device__ void eval_mul_inplace(const struct c_instance& inst,
				 const double* matrix)
{
	const int rows = MWIDTH;

	double y, t;
	double c = 0;
	double sum = 0;

	/* result rows */
	for(int i = 0; i < rows; i++) {
		y = __dmul_rn(RES(ty, i), matrix[RIDX(i, tx)]) - c;
		t = __dadd_rn(sum, y);
		c = (t - sum) - y;
		sum = t;
	}

	__syncthreads();
	RES(ty, tx) = sum;
	__syncthreads();
}

__device__ const int* eval_interpret_rule(const struct c_instance& inst,
				    	  const int              * rule,
				    	  const double		 * ind)
{
	if(*rule == MUL_SEP)
		return rule;

	/*
	 * all multiplications are inplace,
	 * so we copy the first matrix to our result
	 */
	const double* matrix = ind + (*rule) * inst.width_per_matrix;
	eval_copy_matrix_to_res(inst, matrix);
	rule++;

	__syncthreads();

	for(; *rule != MUL_SEP; rule++) {
		matrix = ind + (*rule) * inst.width_per_matrix;
		eval_mul_inplace(inst, matrix);
	}

	return rule;
}

__device__ void c_result_rating(const struct c_instance& inst)
{
	double rating = 0.;

        if(ty == 0 && tx == 0) {
        	const double penalty = 1e6;
        	const int rows = MWIDTH - 1;

                switch(inst.cond_right) {
                case COND_UPPER_LEFT:
                        if((TRES(0, 0) - RES(0, 0)) < 1.f)
                                rating += penalty;
                        break;
                case COND_UPPER_RIGHT:
                        if((TRES(0, rows) - RES(0, rows)) < 1.f)
                                rating += penalty;
                        break;
                case COND_UPPER_LEFT_LOWER_RIGHT:
                        if((TRES(0, 0) - RES(0, 0)) < 1.f)
                                rating += penalty;

                        if((TRES(rows, rows) - RES(rows, rows)) < 1.f)
                                rating += penalty;
                        break;
                default:
                        rating += 2*penalty;
                        break;
                }

                if(inst.match == MATCH_ANY) {
                        if(rating == 0.)
                                matrix_form = 0.;

                        rating = 0.;
                }
        }
	__syncthreads();
	// keep only negative numbers
	if(min(TRES(ty, tx) - (RES(ty, tx)), 0.) == 0.)
		RES(ty, tx) = 0;
	else
		RES(ty, tx) = (RES(ty, tx) + 1) / (TRES(ty, tx) + 1);

//	RES(ty, tx) = fabs(min(TRES(ty, tx) - (RES(ty, tx)), 0.));
//	RES(ty, tx) = __dmul_rn(RES(ty, tx), RES(ty, tx));
	__syncthreads();

	double c = 0.0;
	double y, t;
	double sum;

	//only lines are processed
	if(tx == 0) {
		sum = 0.;

		for(int i = 0; i < MWIDTH; i++) {
			y = RES(ty, i) - c;
			t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}

		RES(ty, 0) = sum;
	}
	__syncthreads();

	if(tx == 0 && ty == 0) {
		for(int i = 0; i < MWIDTH; i++) {
			y = RES(i, 0) - c;
			t = rating + y;
			c = (t - rating) - y;
			rating = t;
		}

		shrd_rating += rating;
	}
	__syncthreads();
}

__device__ double c_calc_res(const struct c_instance& inst,
		             const double* const ind)
{
	const int* end = inst.rules + inst.rules_len - 1;
	const int* rules = inst.rules;

	if(tx == 0 && ty == 0) {
		MWIDTH = inst.mdim;
		shrd_rating = 0.;
		matrix_form = 1e9;
	}

	__syncthreads();

	do {
		eval_set_res_matrix_to_identity();

		rules++;
		rules = eval_interpret_rule(inst , rules, ind);

		__syncthreads();
		TRES(ty, tx) = RES(ty, tx);
		__syncthreads();
		eval_set_res_matrix_to_identity();
		__syncthreads();

		rules++;
		rules = eval_interpret_rule(inst , rules, ind);
		__syncthreads();

		c_result_rating(inst);
		__syncthreads();

		__syncthreads();
	} while(rules != end);

	__syncthreads();

	if(tx == 0 && ty == 0) {
		if(inst.match == MATCH_ANY)
			shrd_rating += matrix_form;
	}

	__syncthreads();
	return shrd_rating;
}

__global__ void setup_rating(struct c_instance inst, int yoff)
{
	const int idx = (blockIdx.x) * inst.icount + (blockIdx.y + yoff);
	const double* indv = inst.instances + idx * inst.width_per_inst;
	inst.rating[idx] = c_calc_res(inst, indv);
}

__global__ void copy_parent_kernel(struct c_instance inst)
{
	__shared__ int parent;
	if(tx == 0 && ty == 0) {
		parent = curand(&(inst.rnd_states[blockIdx.x])) % inst.icount;
		parent = (blockIdx.x * inst.icount + parent) *
				inst.width_per_inst;
	}
	__syncthreads();
	double* src = inst.instances + parent;
	double* dest = inst.tmp + blockIdx.x * inst.width_per_inst;

	for(int i = tx; i < inst.width_per_inst; i += blockDim.x) {
		dest[i] = src[i];
	}
}

__global__ void mutate_kernel(struct c_instance inst)
{
	const int pos = bx * inst.scount + by;

	double* src = inst.tmp + bx * inst.width_per_inst;
	double* dest = inst.sinstances + pos * inst.width_per_inst;

	for(int i = tx; i < inst.width_per_inst; i += blockDim.x) {
		dest[i] = src[i];
	}

	__syncthreads();

	if(tx == 0 && ty == 0) {
		const int mpos = curand(&(inst.rnd_states[pos])) %
				inst.width_per_inst;
		dest[mpos] = max(dest[mpos] - inst.delta, 0.);
	}
}

__global__ void rate_mutated_kernel(struct c_instance inst)
{
	const int pos = bx * inst.scount + by;
	double* ind = inst.sinstances + pos * inst.width_per_inst;
	inst.srating[pos] = c_calc_res(inst, ind);
}

