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
//	if(min(TRES(ty, tx) - (RES(ty, tx)), 0.) == 0.)
//		RES(ty, tx) = 0;
//	else
//		RES(ty, tx) = (RES(ty, tx) + 1) / (TRES(ty, tx) + 1);

	RES(ty, tx) = fabs(min(TRES(ty, tx) - RES(ty, tx), 0.));
//	RES(ty, tx) = __dmul_rn(RES(ty, tx), RES(ty, tx));
	__syncthreads();

	double c = 0.0;
	double y, t;
	double sum;

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

__global__ void calc_res(struct c_instance inst, double* ind, double* dest)
{
	double res = c_calc_res(inst, ind);

	if(tx == 0 && ty == 0)
		*dest = shrd_rating;
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

__device__ void c_ensure_constraints(const struct c_instance & inst,
				     double* const elems)
{
	const int matrices = inst.num_matrices;
	int x;

	if(inst.cond_left == COND_UPPER_LEFT) {
		for(x = 0; x < matrices; x++) {
			const int matrix = x * inst.width_per_matrix;

			if(elems[matrix] < 1.0)
				elems[matrix] = 1.;
		}
	} else if(inst.cond_left == COND_UPPER_RIGHT) {
		for(x = 0; x < matrices; x++) {
			const int matrix = x * inst.width_per_matrix;
			if(elems[matrix] < 1.0)
				elems[matrix] = 1.;
		}
	} else if(inst.cond_left == COND_UPPER_LEFT_LOWER_RIGHT) {
		for(x = 0; x < matrices; x++) {
			const int matrix1 = x * inst.width_per_matrix;
			const int matrix2 = (x + 1) * inst.width_per_matrix - 1;

			if(elems[matrix1] < 1.0)
				elems[matrix1] = 1.;

			if(elems[matrix2] < 1.0)
				elems[matrix2] = 1;
		}
	} else {
		/*
		 * This should be recognized ;) It's only a 1.3 card
		 *  so there is no printf :/
		 */
		for(int i = 0; i < inst.width_per_matrix; i++) {
			elems[i] = 1337;
		}
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
		c_ensure_constraints(inst, dest);
	}
}

__global__ void rate_mutated_kernel(struct c_instance inst)
{
	const int pos = bx * inst.scount + by;
	double* ind = inst.sinstances + pos * inst.width_per_inst;
	double rat = c_calc_res(inst, ind);

	if(tx == 0 && ty == 0)
		inst.srating[pos] = rat;
}

#define PRED_STEP(s) if(tid < (s)) { \
			if(rat[tid] > rat[tid + (s)]) { \
				rat[tid]  = rat[tid + (s)]; \
				sidx[tid] = tid + (s); \
			} else sidx[tid] = tid; }

__global__ void reduce_rating_kernel(struct c_instance inst)
{
	const int tid = tx;
	double* const rat   = inst.srating  + bx * inst.scount;
	int* const sidx = inst.srat_idx + bx * inst.scount;

	/* only the thread.x dim is used */
	if (inst.scount >= 1024) { PRED_STEP(512);  __syncthreads(); }
	if (inst.scount >= 512)  { PRED_STEP(256);  __syncthreads(); }
	if (inst.scount >= 256)  { PRED_STEP(128);  __syncthreads(); }
	if (inst.scount >= 128)  { PRED_STEP(64);   __syncthreads(); }
	if (inst.scount >=  64)  { PRED_STEP(32);   __syncthreads(); }
	if (inst.scount >=  32)  { PRED_STEP(16);   __syncthreads(); }
	if (inst.scount >=  16)  { PRED_STEP( 8);   __syncthreads(); }
	if (inst.scount >=   8)  { PRED_STEP( 4);   __syncthreads(); }
	if (inst.scount >=   4)  { PRED_STEP( 2);   __syncthreads(); }
	if (inst.scount >=   2)  { PRED_STEP( 1);   __syncthreads(); }
}

__global__ void copy_to_child_kernel(struct c_instance inst)
{
	__shared__ int child;
	const int bbx = bx;
	double* const srat    = inst.srating   + bbx * inst.scount;
	int*    const sratidx = inst.srat_idx  + bbx * inst.scount;
	double* const rat     = inst.rating  + bbx * inst.icount;

	if(tx == 0 && ty == 0) {
		child = curand(&(inst.rnd_states[bbx])) % inst.icount;

		if(srat[0] < rat[child]) {
			if(srat[0] < inst.best[bbx]) {
				inst.best[bbx] = srat[0];
				inst.best_idx[bbx] = child;
			}

			rat[child] = srat[0];
			child = (bbx * inst.icount + child) *
				inst.width_per_inst;
		} else {
			child = -1;
		}
	}
	__syncthreads();

	if(child == -1)
		return;

	const int sidx = bbx * inst.scount + sratidx[1];
	double* src  = inst.sinstances + sidx * inst.width_per_inst;
	double* dest = inst.instances  + child;

	for(int i = tx; i < inst.width_per_inst; i += blockDim.x) {
		dest[i] = src[i];
	}
}

#define mlock(mutex) while(atomicCAS((mutex), 0, 1) != 0);
#define munlock(mutex) atomicExch((mutex), 0);

/* Here be dragons: this is not pretty but works. */
__global__ void path_mutate_kernel_p1(struct c_instance inst,
		                      int3* stack, unsigned int* top)
{
	const int* end = inst.rules + inst.rules_len - 1;
	const int* rules = inst.rules;
	const double* ind = inst.tmp + bx * inst.width_per_inst;

	int cur_rule = 0;
	int3 entry;

	stack += bx * inst.rules_count * inst.width_per_matrix;
	top += bx;

	if(tx == 0 && ty == 0) {
		MWIDTH = inst.mdim;
		atomicExch(top, 0);

		entry.x = 0;
		entry.y = 0;
		entry.z = 0;
		stack[0] = entry;

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

		entry.x = tx;
		entry.y = ty;
		entry.z = cur_rule;

		if(min(TRES(ty, tx) - (RES(ty, tx)), 0.) != 0.) {
			int pos = atomicAdd(top, 1);
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
	double* ind = inst.tmp + tid * inst.width_per_inst;

	int cur_rule = 0;

	stack += tid * inst.rules_count * inst.width_per_matrix;
	top += tid;

	curandState rnd = inst.rnd_states[tid];

	int3 entry = stack[curand(&rnd) % *top];
	int l = entry.y;
	int r = entry.x;
	int goal;

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
		goal = *(rules+1) < 0 ? r : 1+ curand(&rnd) % (inst.mdim - 2);
		double* pos = ind + (*rules) * inst.width_per_matrix +
				 l * inst.mdim + goal;
		*pos = max(*pos + inst.delta, 0.);
		l = goal;
	}

	inst.rnd_states[tid] = rnd;
}
