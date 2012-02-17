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

//__shared__ float* rdest;

__shared__ volatile float shrd_rating;
__shared__ float matrix_form;

/* cached rules */
__shared__ int srules[100];
__shared__ int* rend;

template<int mdim>
__device__ const int* eval_interpret_rule(const int* rule, float* rdest)
{
	if(*rule == MUL_SEP) {
		//set matrix to identity matrix
		rdest[RIDX(ty, tx)] = (tx != ty) ? 0.f : 1.f;
		return rule;
	}

	int mat = *rule * mdim * mdim;
	rule++;

	/* there is only one matrix? this is our result */
	if(*rule == MUL_SEP) {
		const int tid = RIDX(ty, tx);
		rdest[tid] = sind[mat + tid];
		return;
	}

	/*
	 * store in rdest the first result, so there we save one store
	 * per matrix element
	 */
	float sum = 0.f;
	const int mat2 = *rule * mdim * mdim;

	for(int i = 0; i < mdim; i++) {
		sum = sum + (sind[mat + RIDX(ty, i)] * sind[mat2 + RIDX(i, tx)]);
	}

	__syncthreads();
	rdest[RIDX(ty, tx)] = sum;
	__syncthreads();
	rule++;

	/* multiply rest of the rule */
	for(; *rule != MUL_SEP; rule++) {
		sum = 0.f;
		mat = *rule * mdim * mdim;

		/* result rows */
		for(int i = 0; i < mdim; i++) {
			sum = sum + (rdest[RIDX(ty, i)] * sind[mat + RIDX(i, tx)]);
		}

		__syncthreads();
		rdest[RIDX(ty, tx)] = sum;
		__syncthreads();
	}

	return rule;
}

template<int mdim, int mcond>
__device__ void c_result_rating(int match)
{
	float rating = 0.;

        if(ty == 0 && tx == 0) {
        	const float penalty = 1e6f;
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
                        if(rating == 0.f)
                                matrix_form = 0.f;

                        rating = 0.f;
                }
        }
	__syncthreads();

	const float a =  RES(ty, tx);
	const float b = TRES(ty, tx);

	const float r = a - b;
	const float f = (b == 0.f ? 1000.f : 1.f );

	RES(ty, tx) = a > b ? (f * (r * r)) : 0.f;
//	RES(ty, tx) = fabs(min(b - a, 0.));
//	RES(ty, tx) = __fmul_rn(RES(ty, tx), RES(ty, tx));

	__syncthreads();

	//only lines are processed
	if(tx == 0) {
		float sum = RES(ty, 0);
		for(int i = 1; i < mdim; i++) {
			sum += RES(ty, i);
		}

		RES(ty, 0) = sum;
	}
	__syncthreads();

	if(tx == 0 && ty == 0) {
		for(int i = 0; i < mdim; i++) {
			rating += RES(i, 0);
		}

		shrd_rating += rating;
	}
}

template<int mdim, int mcond>
__device__ void c_calc_res(int match)
{
	const int* rules = srules;

	if(tx == 0 && ty == 0) {
		shrd_rating = 0.f;
		matrix_form = 1e9f;
	}

	__syncthreads();

	do {
		rules++;
		rules = eval_interpret_rule<mdim>(rules, slhs);

		rules++;
		rules = eval_interpret_rule<mdim>(rules, res);
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
