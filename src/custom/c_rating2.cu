#include <limits.h>
#include <float.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "c_rating.h"
#include "c_instance.h"


#define RIDX(cy, cx) ((cy) * mdim + (cx))
#define RES(cy, cx)  res[RIDX(cy, cx)]
#define TRES(cy, cx) slhs[RIDX(cy, cx)]

__shared__ double sind[2 * MATRIX_WIDTH * MATRIX_WIDTH];

__shared__ double slhs[MATRIX_WIDTH * MATRIX_WIDTH];
__shared__ double res[MATRIX_WIDTH * MATRIX_WIDTH];

__shared__ volatile double shrd_rating;
__shared__ double matrix_form;

template<int mdim>
__device__ void eval_set_res_matrix_to_identity(const struct c_instance& inst)
{
	if(tx != ty) {
		RES(ty, tx) = 0.;
	} else {
		RES(ty, tx) = 1.;
	}
}

template<int mdim>
__device__ inline void eval_copy_matrix_to_res(const struct c_instance& inst,
		    	    	    	       const int matrix)
{
	const int tid = RIDX(ty, tx);
	const int mat = matrix * mdim * mdim;
	res[tid] = sind[mat + tid];
}

template<int mdim>
__device__ void  eval_mul_inplace(const struct c_instance& inst, const int matrix)
{
	double y, t;
	double c = 0;
	double sum = 0;

	const int mat = matrix * mdim * mdim;

	/* result rows */
	for(int i = 0; i < mdim; i++) {
		y = __dmul_rn(RES(ty, i), sind[mat + RIDX(i, tx)]) - c;
		t = __dadd_rn(sum, y);
		c = (t - sum) - y;
		sum = t;
	}

	__syncthreads();
	RES(ty, tx) = sum;
	__syncthreads();
}

template<int mdim>
__device__ const int* eval_interpret_rule(const struct c_instance& inst,
				    	  const int              * rule)
{
	if(*rule == MUL_SEP)
		return rule;

	/*
	 * all multiplications are inplace,
	 * so we copy the first matrix to our result
	 */
	eval_copy_matrix_to_res<mdim>(inst, *rule);
	rule++;

	__syncthreads();

	for(; *rule != MUL_SEP; rule++) {
		eval_mul_inplace<mdim>(inst, *rule);
	}

	return rule;
}

template<int mdim>
__device__ void c_result_rating(const struct c_instance& inst)
{
	double rating = 0.;

        if(ty == 0 && tx == 0) {
        	const double penalty = 1e6;
        	const int rows = mdim - 1;

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
//	if(min(TRES(ty, tx) - (RES(ty, tx)), 0.) == 0.)
//		RES(ty, tx) = 0;
//	else
//		RES(ty, tx) = (RES(ty, tx) + 1) / (TRES(ty, tx) + 1);

	const double a =  RES(ty, tx);
	const double b = TRES(ty, tx);

//	RES(ty, tx) = a > b ? (a * a - b * b) : 0.;
	RES(ty, tx) = fabs(min(b - a, 0.));
//	RES(ty, tx) = __dmul_rn(RES(ty, tx), RES(ty, tx));

	__syncthreads();

	double c = 0.0;
	double y, t;
	double sum;

	//only lines are processed
	if(tx == 0) {
		sum = 0.;

		for(int i = 0; i < mdim; i++) {
			y = RES(ty, i) - c;
			t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}

		RES(ty, 0) = sum;
	}
	__syncthreads();

	c = 0.0;
	if(tx == 0 && ty == 0) {
		for(int i = 0; i < mdim; i++) {
			y = RES(i, 0) - c;
			t = rating + y;
			c = (t - rating) - y;
			rating = t;
		}

		shrd_rating += rating;
	}
	__syncthreads();
}

template<int mdim>
__device__ void c_calc_res(const struct c_instance& inst)
{
	const int* end = inst.rules + inst.rules_len - 1;
	const int* rules = inst.rules;

	if(tx == 0 && ty == 0) {
		shrd_rating = 0.;
		matrix_form = 1e9;
	}

	__syncthreads();

	do {
		eval_set_res_matrix_to_identity<mdim>(inst);

		rules++;
		rules = eval_interpret_rule<mdim>(inst, rules);

		__syncthreads();
		TRES(ty, tx) = RES(ty, tx);
		__syncthreads();
		eval_set_res_matrix_to_identity<mdim>(inst);
		__syncthreads();

		rules++;
		rules = eval_interpret_rule<mdim>(inst, rules);
		__syncthreads();

		c_result_rating<mdim>(inst);
		__syncthreads();
	} while(rules != end);

	__syncthreads();

	if(tx == 0 && ty == 0) {
		if(inst.match == MATCH_ANY)
			shrd_rating += matrix_form;
	}
}

template<int mnum, int mdim>
__global__ void all_in_one_kernel(struct c_instance inst, const int lucky)
{
	const int bbx = blockIdx.x;

	/* mutation */
	double* const indv = inst.tmp + bbx * inst.width_per_inst;

	__shared__ double old_rat;
	double old_val;
	int    mut_pos;

	old_rat = inst.tmprat[bbx];

	/* copy data to cache the ind */
	for(int i = ty * mdim + tx; i < mnum*mdim*mdim; i += mdim*mdim)
		sind[i] = indv[i];

	__syncthreads();

	for(int steps = 0; steps < lucky; steps++) {

		if(tx == 0 && ty == 0) {
			const int mat = curand(&(inst.rnd_states[bbx])) % mnum;
			const int row = curand(&(inst.rnd_states[bbx])) % (mdim -1);
			const int col = 1 + curand(&(inst.rnd_states[bbx])) % (mdim -1);
			mut_pos = mat*mdim*mdim + row * mdim + col;
			old_val = sind[mut_pos];
			sind[mut_pos] = max(old_val - inst.delta, 0.);
		}
		__syncthreads();

		/* rating of mutated kernel */
		c_calc_res<mdim>(inst);
		__syncthreads();

		/* copy back */
		if(tx == 0 && ty == 0) {
			const int luck = curand(&inst.rnd_states[bbx]) % lucky;

			if(shrd_rating > old_rat && luck) {
				sind[mut_pos] = old_val;
			} else {
				old_rat = shrd_rating;
			}
		}
	}

	if(old_rat == inst.tmprat[bbx])
		return;

	/* copy cache back */
	for(int i = ty * mdim + tx; i < mnum*mdim*mdim; i += mdim*mdim)
		indv[i] = sind[i];

	if(tx == 0 && ty == 0)
		inst.tmprat[bbx] = old_rat;
}
