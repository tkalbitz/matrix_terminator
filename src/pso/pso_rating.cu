#include <limits.h>
#include <float.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "pso_rating.h"
#include "pso_memory.h"
#include "pso_config.h"


#define RBLOCK           (blockIdx.x * inst.width_per_line * inst.dim.particles + blockIdx.y * inst.width_per_line)
#define RMAT(mat)        (RBLOCK + (mat) * inst.width_per_matrix)
#define RELEM(mat, cy, cx) (RMAT(mat) + (cy) * inst.dim.matrix_width + (cx))
#define RE(mat, cy, cx)   (inst.rat_tmp[RELEM(mat, cy, cx)])

#define TIDX(cy, cx) (blockIdx.x * inst.width_per_matrix * inst.dim.particles + blockIdx.y * inst.width_per_matrix + cy * inst.dim.matrix_width + cx)
#define TRES(cy, cx) inst.res[TIDX(cy, cx)]
#define RES(cy, cx) res[((cy) * blockDim.y + (cx))]

__shared__ int MHEIGHT;
__shared__ int MWIDTH;

__shared__ double res[MATRIX_HEIGHT * MATRIX_WIDTH];
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

__device__ inline void eval_copy_matrix_to_res(const struct pso_instance& inst,
		    	    	    	       const int cmatrix)
{
	RES(ty, tx) = inst.rat_tmp[RELEM(cmatrix, tx, ty)];
}

__device__ void eval_mul_inplace(const struct pso_instance& inst,
				 const int cmatrix)
{
	const int rows = MHEIGHT;

	double y, t;
	double c = 0;
	double sum = 0;

	/* result rows */
	for(int i = 0; i < rows; i++) {
		y = __dmul_rn(RES(ty, i), RE(cmatrix, i, tx)) - c;
		t = __dadd_rn(sum, y);
		c = (t - sum) - y;
		sum = t;
	}

	__syncthreads();
	RES(ty, tx) = sum;
	__syncthreads();
}

__device__ const int* eval_interpret_rule(const struct pso_instance& inst,
				    	  const int                 * rule)
{
	if(*rule == MUL_SEP)
		return rule;

	/*
	 * all multiplications are inplace,
	 * so we copy the first matrix to our result
	 */
	eval_copy_matrix_to_res(inst, *rule);
	rule++;

	__syncthreads();

	for(; *rule != MUL_SEP; rule++) {
		eval_mul_inplace(inst, *rule);
	}

	return rule;
}

__device__ void pso_result_rating(const struct pso_instance& inst)
{
	const int rows = MHEIGHT - 1;
	const int cols = MWIDTH  - 1;
	double rating = 0.;

	const double penalty = 1e9;

        if(ty == 0 && tx == 0) {
                switch(inst.cond_right) {
                case COND_UPPER_LEFT:
                        if((TRES(0, 0) - RES(0, 0)) < 1.f)
                                rating += penalty;
                        break;
                case COND_UPPER_RIGHT:
                        if((TRES(0, cols) - RES(0, cols)) < 1.f)
                                rating += penalty;
                        break;
                case COND_UPPER_LEFT_LOWER_RIGHT:
                        if((TRES(0, 0) - RES(0, 0)) < 1.f)
                                rating += penalty;

                        if((TRES(rows, cols) - RES(rows, cols)) < 1.f)
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
	RES(ty, tx) = fabs(min(TRES(ty, tx) - RES(ty, tx), 0.));
	RES(ty, tx) = __dmul_rn(RES(ty, tx), RES(ty, tx));
	__syncthreads();

	double c = 0.0;
	double y, t;
	double sum;

	//only lines are processed
	if(tx == 0) {
		sum = RES(ty, 0);

		for(int i = 1; i < MWIDTH; i++) {
			y = RES(ty, i) - c;
			t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}

		RES(ty, 0) = sum;
	}
	__syncthreads();

	if(tx == 0 && ty == 0) {
		for(int i = 0; i < MHEIGHT; i++) {
			y = RES(i, 0) - c;
			t = rating + y;
			c = (t - rating) - y;
			rating = t;
		}

		shrd_rating += sqrtf(rating);
	}
	__syncthreads();
}

__device__ void prepare_tmp_matrix(struct pso_instance& inst,
		                   const int s, const int cur)
{
	int i;
	int rpos = RELEM(0, ty, tx);
	int gpos = blockIdx.x * inst.width_per_line +
		   ty * inst.dim.matrix_width + tx;

	const int add = inst.width_per_matrix;

	for(i = 0; i < inst.num_matrices; i++) {
		inst.rat_tmp[rpos] = inst.particle_gbest[gpos];
		gpos += add;
		rpos += add;
	}
	__syncthreads();

	const int end = cur * s + s;
	for(i = cur * s + tx; i < end; i += blockDim.x) {
		const int perm_idx = inst.col_permut[blockIdx.x * inst.width_per_line + i];
		const int dest_idx = RMAT(0) + perm_idx;
		inst.rat_tmp[dest_idx] = inst.particle[ELEM_IDX(perm_idx)];
	}

}

__global__ void pso_calc_res(struct pso_instance inst, const int s, const int cur)
{
	const int* end = inst.rules + inst.rules_len - 1;
	const int* rules = inst.rules;

	if(tx == 0 && ty == 0) {
		MHEIGHT = inst.dim.matrix_height;
		MWIDTH  = inst.dim.matrix_width;
		shrd_rating = 0.;
		matrix_form = 1e9;
	}

	__syncthreads();
	prepare_tmp_matrix(inst, s, cur);
	__syncthreads();

	uint8_t cur_rule = 0;

	do {
		eval_set_res_matrix_to_identity();

		rules++;
		rules = eval_interpret_rule(inst , rules);

		__syncthreads();
		TRES(ty, tx) = RES(ty, tx);
		eval_set_res_matrix_to_identity();
		__syncthreads();

		rules++;
		rules = eval_interpret_rule(inst , rules);
		__syncthreads();

		pso_result_rating(inst);
		__syncthreads();

		cur_rule++;
		__syncthreads();
	} while(rules != end);

	__syncthreads();

	if(tx == 0 && ty == 0) {
		if(inst.match == MATCH_ANY)
			shrd_rating += matrix_form;

		const int s_count = inst.width_per_line / s;
		const int idx = PARTICLE_COUNT * blockIdx.x * s_count +
				cur * s_count +
				blockIdx.y;
		inst.prat[idx] = shrd_rating;
	}
}

