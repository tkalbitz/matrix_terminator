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

#define RIDX(cy, cx) ((cy) * inst.mdim + (cx))
#define RES(cy, cx)  res[RIDX(cy, cx)]

__shared__ float res[MATRIX_WIDTH * MATRIX_WIDTH];
__shared__ float tmpmat[MATRIX_WIDTH * MATRIX_WIDTH];
__shared__ float shrd_rating;
__shared__ float matrix_form;

__device__ inline void
eval_set_res_matrix_to_identity(const struct c_instance& inst)
{
	if(tx != ty) {
		RES(ty, tx) = 0.;
	} else {
		RES(ty, tx) = 1.;
	}
}

__device__ inline void eval_copy_matrix_to_res(const struct c_instance& inst,
		    	    	    	       const float *  const    matrix,
		    	    	    	       float * const r)
{
	const int tid = RIDX(ty, tx);
	r[tid] = matrix[tid];
}

__device__ void eval_mul_inplace(const struct c_instance& inst)
{
	float y, t;
	float c = 0;
	float sum = 0;

	/* result rows */
	for(int i = 0; i < inst.mdim; i++) {
		y = __fmul_rn(RES(ty, i), tmpmat[RIDX(i, tx)]) - c;
		t = __fadd_rn(sum, y);
		c = (t - sum) - y;
		sum = t;
	}

	__syncthreads();
	RES(ty, tx) = sum;
	__syncthreads();
}

__device__ const int* eval_interpret_rule(const struct c_instance& inst,
				    	  const int              * rule,
				    	  const float		 * ind)
{
	if(*rule == MUL_SEP)
		return rule;

	/*
	 * all multiplications are inplace,
	 * so we copy the first matrix to our result
	 */
	const float* matrix = ind + (*rule) * inst.width_per_matrix;
	eval_copy_matrix_to_res(inst, matrix, res);
	rule++;

	__syncthreads();

	for(; *rule != MUL_SEP; rule++) {
		matrix = ind + (*rule) * inst.width_per_matrix;
		eval_copy_matrix_to_res(inst, matrix, tmpmat);
		eval_mul_inplace(inst);
	}

	return rule;
}

__device__ void c_result_rating(const struct c_instance& inst)
{
	float rating = 0.;

        if(ty == 0 && tx == 0) {
        	const float penalty = 1e6;
        	const int rows = inst.mdim - 1;

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

	const float a =  RES(ty, tx);
	const float b = TRES(ty, tx);

	RES(ty, tx) = a > b ? (a * a - b * b) : 0.;
//	RES(ty, tx) = fabs(min(b - a, 0.));
//	RES(ty, tx) = __fmul_rn(RES(ty, tx), RES(ty, tx));

	__syncthreads();

	float c = 0.0;
	float y, t;
	float sum;

	//only lines are processed
	if(tx == 0) {
		sum = 0.;

		for(int i = 0; i < inst.mdim; i++) {
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
		for(int i = 0; i < inst.mdim; i++) {
			y = RES(i, 0) - c;
			t = rating + y;
			c = (t - rating) - y;
			rating = t;
		}

		shrd_rating += rating;
	}
	__syncthreads();
}

__device__ float c_calc_res(const struct c_instance& inst,
		             const float* const ind)
{
	const int* end = inst.rules + inst.rules_len - 1;
	const int* rules = inst.rules;

	if(tx == 0 && ty == 0) {
		shrd_rating = 0.;
		matrix_form = 1e9;
	}

	__syncthreads();

	do {
		eval_set_res_matrix_to_identity(inst);

		rules++;
		rules = eval_interpret_rule(inst, rules, ind);

		__syncthreads();
		TRES(ty, tx) = RES(ty, tx);
		__syncthreads();
		eval_set_res_matrix_to_identity(inst);
		__syncthreads();

		rules++;
		rules = eval_interpret_rule(inst, rules, ind);
		__syncthreads();

		c_result_rating(inst);
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

__global__ void calc_res(struct c_instance inst, float* ind, float* dest)
{
	float res = c_calc_res(inst, ind);

	if(tx == 0 && ty == 0)
		*dest = shrd_rating;
}

__global__ void calc_tmp_res(struct c_instance inst)
{
	float* ind = inst.tmp + bx * inst.width_per_inst;
	float res = c_calc_res(inst, ind);

	if(tx == 0 && ty == 0)
		inst.tmprat[bx] = shrd_rating;
}

__global__ void setup_rating(struct c_instance inst, int yoff)
{
	const int idx = (blockIdx.x) * inst.icount + (blockIdx.y + yoff);
	const float* indv = inst.instances + idx * inst.width_per_inst;
	float rat = c_calc_res(inst, indv);

	if(tx == 0 && ty == 0)
		inst.rating[idx] = rat;
}

__global__ void copy_parent_kernel(struct c_instance inst)
{

	__shared__ int parent;
	if(tx == 0 && ty == 0) {
		float* const rat = inst.rating + bx * inst.icount;

		parent = curand(&(inst.rnd_states[blockIdx.x])) % inst.icount;
		parent = (blockIdx.x * inst.icount + parent) *
				inst.width_per_inst;
	}
	__syncthreads();
	float* src = inst.instances + parent;
	float* dest = inst.tmp + blockIdx.x * inst.width_per_inst;

	for(int i = tx; i < inst.width_per_inst; i += blockDim.x) {
		dest[i] = src[i];
	}
}

//__global__ void all_in_one_kernel(struct c_instance inst, const int lucky)
//{
//	const int bbx = blockIdx.x;
//
//	/* mutation */
//	float* const indv = inst.tmp + bbx * inst.width_per_inst;
//
//	float old_rat = inst.tmprat[bbx];
//	float old_val;
//	int    mut_pos;
//
//	for(int steps = 0; steps < lucky; steps++) {
//
//		if(tx == 0 && ty == 0) {
//			const int mat = curand(&(inst.rnd_states[bbx])) % inst.num_matrices;
//			const int row = curand(&(inst.rnd_states[bbx])) % (inst.mdim -1);
//			const int col = 1 + curand(&(inst.rnd_states[bbx])) % (inst.mdim -1);
//			mut_pos = mat*inst.mdim*inst.mdim + row * inst.mdim + col;
//			old_val = indv[mut_pos];
//			indv[mut_pos] = max(old_val - inst.delta, 0.);
//		}
//		__syncthreads();
//
//		/* rating of mutated kernel */
//		c_calc_res(inst, indv);
//		__syncthreads();
//
//		/* copy back */
//		if(tx == 0 && ty == 0) {
//			const int luck = curand(&inst.rnd_states[bbx]) % lucky;
//
//			if(shrd_rating > old_rat && luck) {
//				indv[mut_pos] = old_val;
//			} else {
//				old_rat = shrd_rating;
//			}
//		}
//		__syncthreads();
//	}
//
//	if(tx == 0 && ty == 0)
//		inst.tmprat[bbx] = old_rat;
//}

__global__ void mutate_kernel(struct c_instance inst)
{
	float* src = inst.tmp + bx * inst.width_per_inst;
	float* dest = inst.sinstances + bx * inst.width_per_inst;

	for(int i = tx; i < inst.width_per_inst; i += blockDim.x) {
		dest[i] = src[i];
	}

	__syncthreads();

	if(tx == 0 && ty == 0) {
		const int mat = curand(&(inst.rnd_states[bx])) % inst.num_matrices;
		const int row = curand(&(inst.rnd_states[bx])) % (inst.mdim -1);
		const int col = 1 + curand(&(inst.rnd_states[bx])) % (inst.mdim -1);
		const int mpos = mat*inst.mdim*inst.mdim + row*inst.mdim + col;
		dest[mpos] = max(dest[mpos] - inst.delta, 0.);
	}
}

__global__ void rate_mutated_kernel(struct c_instance inst)
{
	const float* ind = inst.sinstances + bx * inst.width_per_inst;
	const float rat = c_calc_res(inst, ind);

	if(tx == 0 && ty == 0)
		inst.srating[bx] = shrd_rating;
}

__global__ void copy_to_child_kernel(struct c_instance inst)
{
	__shared__ int child;
	const int bbx = bx;
	float* const rat = inst.rating + bbx * inst.icount;

	if(tx == 0 && ty == 0) {
		child = curand(&(inst.rnd_states[bbx])) % inst.icount;

		float trat = inst.tmprat[bbx];
		if(trat < rat[child]) {
			if(trat < inst.best[bbx]) {
				inst.best[bbx] = trat;
				inst.best_idx[bbx] = child;
			}

			rat[child] = trat;
			child = (bbx * inst.icount + child) *
				inst.width_per_inst;
		} else {
			child = -1;
		}
	}
	__syncthreads();

	if(child == -1)
		return;

	float* src  = inst.tmp + bbx * inst.width_per_inst;
	float* dest = inst.instances + child;

	for(int i = tx; i < inst.width_per_inst; i += blockDim.x) {
		dest[i] = src[i];
	}
	__syncthreads();
}

__global__ void copy_to_tmp_kernel(struct c_instance inst, int lucky)
{
	const int bbx = bx;
	__shared__ int luck;

	if(tx == 0 && ty == 0)
		luck = curand(&inst.rnd_states[bbx]) % lucky;

	if(inst.srating[bbx] > inst.tmprat[bbx] && luck)
		return;

	inst.tmprat[bbx] = inst.srating[bbx];

	float* src  = inst.sinstances + bbx * inst.width_per_inst;
	float* dest = inst.tmp + bbx * inst.width_per_inst;

	for(int i = tx; i < inst.width_per_inst; i += blockDim.x) {
		dest[i] = src[i];
	}
}

/* Here be dragons: this is not pretty but works. */
__global__ void path_mutate_kernel_p1(struct c_instance inst,
		                      int3* stack, unsigned int* top)
{
	const int* end = inst.rules + inst.rules_len - 1;
	const int* rules = inst.rules;
	const float* ind = inst.tmp + bx * inst.width_per_inst;

	int pos;
	int cur_rule = 0;
	int3 entry;

	stack += bx * inst.rules_count * inst.width_per_matrix * 2;
	top += bx;

	if(tx == 0 && ty == 0) {
		atomicExch(top, 0);

		entry.x = 0;
		entry.y = 0;
		entry.z = 0;
		stack[0] = entry;
	}

	__syncthreads();

	const int rows = inst.mdim - 1;
	int special = 0;

	if(inst.cond_right == COND_UPPER_LEFT && ty == 0 && tx == 0)
		special = 1;
	if(inst.cond_right == COND_UPPER_RIGHT && ty == 0 && tx == rows)
		special = 1;
	if(inst.cond_right == COND_UPPER_LEFT_LOWER_RIGHT &&
		((ty == 0 && tx == 0) || (ty == rows && tx == rows)))
		special = 1;

	do {
		eval_set_res_matrix_to_identity(inst);

		rules++;
		rules = eval_interpret_rule(inst , rules, ind);

		__syncthreads();
		TRES(ty, tx) = RES(ty, tx);
		__syncthreads();
		eval_set_res_matrix_to_identity(inst);
		__syncthreads();

		rules++;
		rules = eval_interpret_rule(inst , rules, ind);
		__syncthreads();

		entry.x = tx;
		entry.y = ty;
		entry.z = cur_rule;
		const float lhs = TRES(ty, tx);
		const float rhs = RES(ty, tx);

		const int ok = special ? ((lhs - rhs) >= 1.) : lhs >= rhs;
		if(!ok) {
			pos = atomicAdd(top, 1);
			stack[pos] = entry;
		}

		cur_rule++;
		__syncthreads();
	} while(rules != end);
}

__global__ void path_mutate_kernel_p2(struct c_instance inst, int3* stack,
		unsigned int* top)
{
	const int tid = bx;
	const int* rules = inst.rules;
	float* ind = inst.tmp + tid * inst.width_per_inst;

	int cur_rule = 0;

	stack += tid * inst.rules_count * inst.width_per_matrix * 2;
	top += tid;

	curandState rnd = inst.rnd_states[tid];

	const int chosen = (*top < 2 ? 0 : curand(&rnd) % *top);
	int3 entry = stack[chosen];
	int l = entry.y;
	int r = entry.x;
	int goal;

	/* at least go to the first entry */
	rules++;

	/* we have to jump to the rule for that entry */
	while(cur_rule != entry.z) {
		while(*rules != MUL_SEP)
			rules++;

		rules++;

		while(*rules != MUL_SEP)
			rules++;

		rules++;
		cur_rule++;
	}

	/* put new weights on the path */
	for(; *rules != MUL_SEP; rules++) {
		goal = *(rules+1) < 0 ? r : 1 + curand(&rnd) % (inst.mdim - 2);
		float* pos = ind + (*rules) * inst.width_per_matrix +
				 l * inst.mdim + goal;
		*pos = max(*pos + inst.delta, 1.);
		l = goal;
	}

	inst.rnd_states[tid] = rnd;
}
