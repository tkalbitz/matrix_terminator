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
	int cols = 2 * inst->dim.matrix_width;

	float *row;

	for(int r = 0; r < rows; r++) {
		row = R_ROW(r);
		for(int c = 0; c < cols; c++) {
			row[c] = 0.f;
		}
	}
}

__device__ void eval_copy_matrix_to_res(struct instance *inst,
		    	    	    	struct memory *mem,
		    	    	    	int cmatrix,
		    	    	    	int rmatrix)
{
	int rows = inst->dim.matrix_height;

	int cstart = mem->c_zero +
		     cmatrix * inst->dim.matrix_width;
	int rstart = mem->r_zero1 +
		     rmatrix * inst->dim.matrix_width;

	for(int r = 0; r < rows; r++) {
		memcpy(&(R_ROW(r)[rstart]),
		       &(C_ROW(r)[cstart]),
		       inst->dim.matrix_width * sizeof(float));
	}
}

__device__ void eval_mul_inplace(struct instance *inst,
				 struct memory *mem,
				 int cmatrix,
				 int rmatrix)
{
	int rows = inst->dim.matrix_height;

	int cstart = mem->c_zero  + cmatrix * inst->dim.matrix_width;
	int rstart = mem->r_zero1 + rmatrix * inst->dim.matrix_width;

	float *rrow;
	float row[MUL_ROW_LEN];

	/* result rows */
	for(int rridx = 0; rridx < rows; rridx++) {
		rrow = R_ROW(rridx);

		/* copy current line so we can work inplace */
		memcpy(row, &(rrow[rstart]), inst->dim.matrix_width * sizeof(float));

		/* child column */
		for(int ccidx = 0; ccidx < rows; ccidx++) {
			int pos = rstart + ccidx;
			rrow[pos] = 0.f;

			/* child row */
			for(int cridx = 0; cridx < rows; cridx++) {
				rrow[pos] += row[cridx] * C_ROW(cridx)[cstart + ccidx];
			}
		}
	}
}

__device__ int* eval_interpret_rule(struct instance *inst,
				    struct memory   *mem,
				    int*   rule,
				    int    rmatrix)
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

__device__ float evo_result_rating(struct instance *inst,
				   struct memory   *mem)
{
	int rows = inst->dim.matrix_height;
	int cols = inst->dim.matrix_width;
	int first = mem->r_zero1;
	int sec = mem->r_zero2;
	float* row;
	float rating = 0.f;

	for(int r = 0; r < rows; r++) {
		row = R_ROW(r);
		for(int c = 0; c < cols; c++) {
			rating += fabs(max(row[first + c] - row[sec + c], 0.f));
		}
	}

	row = R_ROW(0);
	if(inst->cond_right == COND_UPPER_LEFT) {
		if((row[first] - row[sec]) < 0)
			rating += 1e5;
	} else if(inst->cond_right == COND_UPPER_RIGHT) {
		if((row[sec - 1] - row[sec + cols - 1]) < 0)
			rating += 1e5;
	} else if(inst->cond_right == COND_UPPER_LEFT_LOWER_RIGHT) {
		if((row[first] - row[sec]) < 0)
			rating += 1e5;

		row = R_ROW(rows-1);
		if((row[sec-1] - row[sec + cols - 1]) < 0)
			rating += 1e5;
	}

	return rating;
}

__device__ float evo_calc_res(struct instance *inst,
			      struct memory   *mem)
{
	int* end = inst->rules + inst->rules_len - 1;
	int* rules = inst->rules;
	float rating = 0.f;

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
