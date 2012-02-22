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

/* cached individuum */
extern __shared__ float sind[];

/* accu and cached rhs */
__shared__ float* res;

//__shared__ float* rdest;

__shared__ volatile float shrd_rating;
__shared__ float matrix_form;

/* cached rules */
__shared__ int srules[100];
__shared__ int* rend;

template<int mdim>
__device__ const int* eval_interpret_rule(const int* rule, float* rres)
{
	if(*rule == MUL_SEP) {
		//set matrix to identity matrix
		*rres = (tx != ty) ? 0.f : 1.f;
		return rule;
	}

	int mat = *rule * mdim * mdim;
	rule++;

	/* there is only one matrix? this is our result */
	if(*rule == MUL_SEP) {
		const int tid = RIDX(ty, tx);
		*rres = sind[mat + tid];
		return rule;
	}

	/*
	 * calculate the first result
	 */
	const int mat2 = *rule * mdim * mdim;

	*rres = sind[mat + RIDX(ty, 0)] * sind[mat2 + RIDX(0, tx)];
	for(int i = 1; i < mdim; i++) {
		*rres += sind[mat + RIDX(ty, i)] * sind[mat2 + RIDX(i, tx)];
	}
	rule++;

	/* no further rule */
	if(*rule == MUL_SEP) {
		return rule;
	}

	res[RIDX(ty, tx)] = *rres;
	__syncthreads();

	const int* arule = rule + 1;

	/* multiply rest of the rule */
	for(; *arule != MUL_SEP; arule++, rule++) {
		mat = *rule * mdim * mdim;

		/* result rows */
		*rres = res[RIDX(ty, 0)] * sind[mat + RIDX(0, tx)];
		for(int i = 1; i < mdim; i++) {
			*rres += res[RIDX(ty, i)] * sind[mat + RIDX(i, tx)];
		}

		__syncthreads();
		res[RIDX(ty, tx)] = *rres;
		__syncthreads();
	}

	mat = *rule * mdim * mdim;

	/* result rows */
	*rres = res[RIDX(ty, 0)] * sind[mat + RIDX(0, tx)];
	for(int i = 1; i < mdim; i++) {
		*rres += res[RIDX(ty, i)] * sind[mat + RIDX(i, tx)];
	}

	return (rule + 1);
}

template<int mdim, int mcond>
__device__ void c_result_rating(int match, const float eps, const float lhs, const float rhs)
{
	float rating = 0.f;
	const float penalty = 1e6f;
	const int rows = mdim - 1;

	if(mcond == COND_UPPER_LEFT) {
		if(tx == 0 && ty == 0) {
			if((lhs - rhs) < 1.f)
				rating = penalty;

			if(match == MATCH_ANY) {
					if(rating == 0.f)
						matrix_form = 0.f;

					rating = 0.f;
			}
		}
	}

	if(mcond == COND_UPPER_RIGHT)
		if(tx == rows && ty == 0) {
			if((lhs - rhs) < 1.f)
				rating = penalty;

			if(match == MATCH_ANY) {
					if(rating == 0.f)
						matrix_form = 0.f;

					rating = 0.f;
			}
		}

	if(mcond == COND_UPPER_LEFT_LOWER_RIGHT) {
		if(tx == rows && ty == rows)
			if((lhs - rhs) < 1.f)
				rating = penalty;

		if(tx == 0 && ty == 0)
			if((lhs - rhs) < 1.f)
				rating = penalty;

		if(match == MATCH_ANY) {
				if(rating == 0.f)
					matrix_form = 0.f;

				rating = 0.f;
		}

	}

	float r = lhs - rhs;

	if(fabs(r) < eps)
		r = 0.f;

	const float f = (lhs == 0.f ? 1000.f : 1.f );

	RES(ty, tx) = (rhs > lhs ? (f * (r * r)) : 0.f) + rating;
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
		float sum = RES(0, 0);
		for(int i = 1; i < mdim; i++) {
			sum += RES(i, 0);
		}

		shrd_rating += sum;
	}
}

template<int mdim, int mcond>
__device__ void c_calc_res(const int match, const float eps)
{
	const int* rules = srules;

	if(tx == 0 && ty == 0) {
		shrd_rating = 0.f;
		matrix_form = 1e9f;
	}

	float lhs;
	float rhs;

	do {
		rules++;
		rules = eval_interpret_rule<mdim>(rules, &lhs);
		__syncthreads();

		rules++;
		rules = eval_interpret_rule<mdim>(rules, &rhs);
		__syncthreads();

		c_result_rating<mdim, mcond>(match, eps, lhs, rhs);
		__syncthreads();
	} while(rules != rend);

	if(tx == 0 && ty == 0) {
		if(match == MATCH_ANY)
			shrd_rating += matrix_form;
	}
}
