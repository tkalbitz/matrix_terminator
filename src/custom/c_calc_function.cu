/*
 * c_calc_function.cu
 *
 *  Created on: Feb 16, 2012
 *      Author: tkalbitz
 */

#include <cuda.h>
#include <curand_kernel.h>

#include "c_config.h"

#define RIDX(cy, cx) ((cy) * mdim + (cx))
#define RES(cy, cx)  res[RIDX(cy, cx)]
#define TRES(cy, cx) slhs[RIDX(cy, cx)]

/* cached individuum */
extern __shared__ float sind[];

/* cached lhs of the result */
__shared__ float* slhs;

/* accu and cached rhs */
__shared__ float* res;

__shared__ volatile float shrd_rating;
__shared__ float matrix_form;

__shared__ float old_rat;
__shared__ curandState rnd;

/* cached rules */
__shared__ int srules[100];
__shared__ int* rend;

template<int mdim>
__device__ void eval_set_res_matrix_to_identity()
{
	if(tx != ty) {
		RES(ty, tx) = 0.;
	} else {
		RES(ty, tx) = 1.;
	}
}

template<int mdim>
__device__ inline void eval_copy_matrix_to_res(const int matrix)
{
	const int tid = RIDX(ty, tx);
	const int mat = matrix * mdim * mdim;
	res[tid] = sind[mat + tid];
}

template<int mdim>
__device__ void  eval_mul_inplace(const int matrix)
{
	#ifdef KAHAN
		float y, t;
		float c = 0;
	#endif
	float sum = 0;

	const int mat = matrix * mdim * mdim;

	/* result rows */
	for(int i = 0; i < mdim; i++) {
		#ifdef KAHAN
			y = __fmul_rn(RES(ty, i), sind[mat + RIDX(i, tx)]) - c;
			t = __fadd_rn(sum, y);
			c = (t - sum) - y;
			sum = t;
		#else
			sum += __fmul_rn(RES(ty, i), sind[mat + RIDX(i, tx)]);
		#endif
	}

	__syncthreads();
	RES(ty, tx) = sum;
	__syncthreads();
}

template<int mdim>
__device__ const int* eval_interpret_rule(const int* rule)
{
	if(*rule == MUL_SEP)
		return rule;

	/*
	 * all multiplications are inplace,
	 * so we copy the first matrix to our result
	 */
	eval_copy_matrix_to_res<mdim>(*rule);
	rule++;

	__syncthreads();

	for(; *rule != MUL_SEP; rule++) {
		eval_mul_inplace<mdim>(*rule);
	}

	return rule;
}

template<int mdim, int mcond>
__device__ void c_result_rating(int match)
{
	float rating = 0.;

        if(ty == 0 && tx == 0) {
        	const float penalty = 1e6;
        	const int rows = mdim - 1;

                switch(mcond) {
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

                if(match == MATCH_ANY) {
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

	const float a =  RES(ty, tx);
	const float b = TRES(ty, tx);

//	RES(ty, tx) = a > b ? (a * a - b * b) : 0.;
	RES(ty, tx) = fabs(min(b - a, 0.));
	RES(ty, tx) = __fmul_rn(RES(ty, tx), RES(ty, tx));

	__syncthreads();

	#ifdef KAHAN
		float c = 0.f;
		float y, t;
	#endif
	float sum;


	//only lines are processed
	if(tx == 0) {
		sum = 0.f;
		for(int i = 0; i < mdim; i++) {
			#ifdef KAHAN
				y = RES(ty, i) - c;
				t = sum + y;
				c = (t - sum) - y;
				sum = t;
			#else
				sum += RES(ty, i);
			#endif
		}

		RES(ty, 0) = sum;
	}
	__syncthreads();

	if(tx == 0 && ty == 0) {
		for(int i = 0; i < mdim; i++) {
			#ifdef KAHAN
				y = RES(i, 0) - c;
				t = rating + y;
				c = (t - rating) - y;
				rating = t;
			#else
				rating += RES(i, 0);
			#endif
		}

		shrd_rating += rating;
	}
	__syncthreads();
}

template<int mdim, int mcond>
__device__ void c_calc_res(int match)
{
	const int* rules = srules;

	if(tx == 0 && ty == 0) {
		shrd_rating = 0.;
		matrix_form = 1e9;
	}

	__syncthreads();

	do {
		eval_set_res_matrix_to_identity<mdim>();

		rules++;
		rules = eval_interpret_rule<mdim>(rules);

		__syncthreads();
		TRES(ty, tx) = RES(ty, tx);
		__syncthreads();
		eval_set_res_matrix_to_identity<mdim>();
		__syncthreads();

		rules++;
		rules = eval_interpret_rule<mdim>(rules);
		__syncthreads();

		c_result_rating<mdim, mcond>(match);
		__syncthreads();
	} while(rules != rend);

	__syncthreads();

	if(tx == 0 && ty == 0) {
		if(match == MATCH_ANY)
			shrd_rating += matrix_form;
	}
}
