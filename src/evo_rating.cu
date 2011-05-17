#include <limits.h>
#include <float.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "config.h"
#include "instance.h"

__device__ void eval_set_res_matrix_to_zero(struct instance *inst,
					    struct memory   *mem)
{
	int rows = inst->dim.matrix_height;
	int start = mem->r_zero1;

	double *row0;

	row0 = R_ROW(0) + start;

	const int width = 2*inst->dim.matrix_width;
	for(int c = 0; c < width; c++) {
		row0[c] = 0.f;
	}

	for(int r = 1; r < rows; r++) {
		double_memcpy(&(R_ROW(r)[start]), row0, width);
	}
}

__device__ void eval_copy_matrix_to_res(struct instance *inst,
		    	    	    	struct memory *mem,
		    	    	    	const int cmatrix,
		    	    	    	const int rmatrix)
{
	const int rows = MATRIX_HEIGHT;

	const int cstart = mem->c_zero  + cmatrix * MATRIX_WIDTH;
	const int rstart = mem->r_zero1 + rmatrix * MATRIX_WIDTH;

	for(int r = 0; r < rows; r++) {
		double_memcpy(&(R_ROW(r)[rstart]),
			      &(C_ROW(r)[cstart]),
			      inst->dim.matrix_width);
	}
}

__device__ void eval_mul_inplace(struct instance *inst,
				 struct memory *mem,
				 const int cmatrix,
				 const int rmatrix)
{
	const int rows = MATRIX_HEIGHT;

	const int cstart = mem->c_zero  + cmatrix * inst->dim.matrix_width;
	const int rstart = mem->r_zero1 + rmatrix * inst->dim.matrix_width;

	double row[MUL_ROW_LEN];

	/* result rows */
	for(int rridx = 0; rridx < rows; rridx++) {
		double* const rrow = &(R_ROW(rridx)[rstart]);

		/* copy current line so we can work inplace */
		double_memcpy(row, rrow, inst->dim.matrix_width);

		/* child column */
		for(int ccidx = 0; ccidx < rows; ccidx++) {
			double tmp = 0.f;

			/* child row */
			for(int cridx = 0; cridx < rows; cridx++) {
				const double* const crow = C_ROW(cridx);
				tmp += row[cridx] * crow[cstart + ccidx];
			}

			rrow[ccidx] = tmp;
		}
	}
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

	for(; *rule != MUL_SEP; rule++) {
		eval_mul_inplace(inst, mem, *rule, rmatrix);
	}

	return rule;
}

__device__ double evo_result_rating(struct instance *inst,
				   struct memory   *mem)
{
	const int rows = MATRIX_HEIGHT;
	const int cols = MATRIX_WIDTH;
	const int first = mem->r_zero1;
	const int sec   = mem->r_zero2;
	double* row;
	double rating = 0.f;

	for(int r = 0; r < rows; r++) {
		row = R_ROW(r);
		for(int c = 0; c < cols; c++) {
			rating += fabs(min(row[first + c] - row[sec + c], 0.f));
		}
	}

	row = R_ROW(0);
	if(inst->cond_right == COND_UPPER_LEFT) {
		if((row[first] - row[sec]) < 1.f)
			rating += 1e5;
	} else if(inst->cond_right == COND_UPPER_RIGHT) {
		if((row[first + cols - 1] - row[sec + cols - 1]) < 1.f)
			rating += 1e5;
	} else if(inst->cond_right == COND_UPPER_LEFT_LOWER_RIGHT) {
		if((row[first] - row[sec]) < 1.f)
			rating += 1e5;

		row = R_ROW(rows-1);
		if((row[first + cols - 1] - row[sec + cols - 1]) < 1.f)
			rating += 1e5;
	} else {
		rating += 1e10;
	}

	return rating;
}

__device__ double evo_calc_res(struct instance *inst,
			      struct memory   *mem)
{
	const int* end = inst->rules + inst->rules_len - 1;
	int* rules = inst->rules;
	double rating = 0.f;

	if(inst->match == MATCH_ANY) {
		rating = FLT_MAX;
	}

	do {
		eval_set_res_matrix_to_zero(inst, mem);

		rules++;
		rules = eval_interpret_rule(inst , mem, rules, 0);

		rules++;
		rules = eval_interpret_rule(inst , mem, rules, 1);

		if(inst->match == MATCH_ALL) {
			rating += evo_result_rating(inst, mem);
		} else {
			rating = min(rating, evo_result_rating(inst, mem));
		}
	} while(rules != end);

	return rating;
}
