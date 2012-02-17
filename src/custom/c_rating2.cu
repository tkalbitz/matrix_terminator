#include <limits.h>
#include <float.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "c_rating2.h"
#include "c_instance.h"

#include "c_calc_function.cu"

__shared__ float old_rat;
__shared__ curandState rnd;

template<int mnum, int mdim>
__device__ void copy_to_child(struct c_instance& inst, unsigned int crnd)
{
	__shared__ int child;
	const int bbx = bx;
	float* const rat = inst.rating + bbx * inst.icount;
	const int iwidth = mnum*mdim*mdim;

	if(tx == 0 && ty == 0) {
		child = crnd % inst.icount;

		if(old_rat < rat[child]) {
			if(old_rat < inst.best[bbx]) {
				inst.best[bbx] = old_rat;
				inst.best_idx[bbx] = child;
			}

			rat[child] = old_rat;
			child = (bbx * inst.icount + child) * iwidth;
		} else {
			child = -1;
		}
	}
	__syncthreads();

	if(child == -1)
		return;

	float* dest = inst.instances + child;
	for(int i = RIDX(ty, tx); i < iwidth; i += mdim*mdim) {
		dest[i] = sind[i];
	}
}

template<int mnum, int mdim>
__device__ void copy_parent(struct c_instance& inst, unsigned int p)
{
	const int iwidth = mnum*mdim*mdim;

	__shared__ int parent;
	if(tx == 0 && ty == 0) {
		parent = p % inst.icount;
		parent = (blockIdx.x * inst.icount + parent) * iwidth;
	}
	__syncthreads();
	float* src = inst.instances + parent;

	for(int i = RIDX(ty, tx); i < iwidth; i += mdim*mdim) {
		sind[i] = src[i];
	}
}

template<int mnum, int mdim>
__device__  void path_mutate_p1(struct c_instance& inst,
		                int3*          __restrict__ stack,
		                unsigned int*  __restrict__ top)
{
	const int* rules = srules;
	const int iwidth = mnum*mdim*mdim;

	int pos;
	int cur_rule = 0;
	int3 entry;

	stack += bx * inst.rules_count * iwidth;
	top += bx;

	if(tx == 0 && ty == 0) {
		atomicExch(top, 0);

		entry.x = 0;
		entry.y = 0;
		entry.z = 0;
		stack[0] = entry;
	}

	__syncthreads();

	const int rows = mdim - 1;
	int special = 0;

	if(inst.cond_right == COND_UPPER_LEFT && ty == 0 && tx == 0)
		special = 1;
	if(inst.cond_right == COND_UPPER_RIGHT && ty == 0 && tx == rows)
		special = 1;
	if(inst.cond_right == COND_UPPER_LEFT_LOWER_RIGHT &&
		((ty == 0 && tx == 0) || (ty == rows && tx == rows)))
		special = 1;

	do {
		rules++;
		rules = eval_interpret_rule<mdim>(rules);

		__syncthreads();
		TRES(ty, tx) = RES(ty, tx);
		__syncthreads();

		rules++;
		rules = eval_interpret_rule<mdim>(rules);
		__syncthreads();

		entry.x = tx;
		entry.y = ty;
		entry.z = cur_rule;
		const float lhs = TRES(ty, tx);
		const float rhs = RES(ty, tx);

		const int ok = special ? ((lhs - rhs) >= 1.f) : lhs >= rhs;
		if(!ok) {
			pos = atomicAdd(top, 1);
			stack[pos] = entry;
		}

		cur_rule++;
		__syncthreads();
	} while(rules != rend);
}

template<int mnum, int mdim>
__device__ void path_mutate_p2(struct c_instance& inst,
		               int3*         __restrict__ stack,
		               unsigned int* __restrict__ top,
		               int rchosen)
{
	const int iwidth = mnum*mdim*mdim;

	const int tid = bx;
	const int* rules = srules;

	int cur_rule = 0;

	stack += tid * inst.rules_count * iwidth;
	top += tid;

	const int chosen = (*top < 2 ? 0 : rchosen % *top);
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
		goal = *(rules+1) < 0 ? r : 1 + curand(&rnd) % (mdim - 2);
		float* pos = sind + (*rules) * iwidth + l * mdim + goal;
		*pos = max(*pos + inst.delta, 1.);
		l = goal;
	}
}

#define MAX_RND 5

template<int mnum, int mdim, int mcond>
__global__ void all_in_one_kernel(struct c_instance inst,
				  int3*          __restrict__ stack,
                		  unsigned int*  __restrict__ top,
                		  const int lucky)
{
	const int bbx = blockIdx.x;

	/* mutation */
	float old_val;
	int    mut_pos;

	__shared__ curandState srnd[MAX_RND];
	__shared__ unsigned int r[MAX_RND];

	if(tx == 0 && ty == 0) {
		rnd = inst.rnd_states[bbx * mdim + MAX_RND];
		rend = srules + inst.rules_len - 1;
		res = sind + mnum * mdim * mdim;
		slhs = res + mdim * mdim;
	}

	if(ty == 0 && tx < MAX_RND) {
		srnd[tx] = inst.rnd_states[bbx * mdim + tx];
		r[tx] = curand(&srnd[tx]);
	}

	/* caching of rules to speed up access */
	for(int i = RIDX(ty, tx); i < inst.rules_len; i += mdim*mdim)
		srules[i] = inst.rules[i];

	copy_parent<mnum, mdim>(inst, r[0]);
	__syncthreads();

	path_mutate_p1<mnum, mdim>(inst, stack, top);
	__syncthreads();

	if(tx == 0 && ty == 0)
		path_mutate_p2<mnum, mdim>(inst, stack, top, r[1]);
	__syncthreads();

	c_calc_res<mdim, mcond>(inst.match);
	if(tx == 0 && ty == 0)
		old_rat = shrd_rating;
	__syncthreads();

	for(int steps = 0; steps < lucky; steps++) {
		/* rnd numbers for this iteration */
		if(ty == 0 && tx < MAX_RND)
			r[tx] = curand(&srnd[tx]);

		__syncthreads();

		if(tx == 0 && ty == 0) {
			const int mat  =      r[0] % mnum;
			const int row  =      r[1] % (mdim -1);
			const int col  = 1 +  r[2] % (mdim -1);
			const int diff = 2 * (r[3] % 2) - 1 ;
			mut_pos = mat * mdim*mdim + row * mdim + col;
			old_val = sind[mut_pos];
			sind[mut_pos] = max(old_val + diff * inst.delta, 0.);
		}
		__syncthreads();

		/* rating of mutated kernel */
		c_calc_res<mdim, mcond>(inst.match);
		__syncthreads();

		/* restore old version when it's worse */
		if(tx == 0 && ty == 0) {
			const int luck = r[4] % lucky;

			if(shrd_rating > old_rat && luck) {
				sind[mut_pos] = old_val;
			} else {
				old_rat = shrd_rating;
			}
		}
	}

	copy_to_child<mnum, mdim>(inst, r[0]);
	inst.rnd_states[bbx * mdim + MAX_RND] = rnd;

	if(ty == 0 && tx < MAX_RND)
		inst.rnd_states[bbx * mdim + tx] = srnd[tx];

}

void start_astep(struct c_instance& inst,
		int3*          __restrict__ stack,
		unsigned int*  __restrict__ top,
		unsigned int asteps)
{
	size_t space =(inst.num_matrices * inst.mdim * inst.mdim +
			inst.mdim * inst.mdim +
			inst.mdim * inst.mdim) * sizeof(float);

	dim3 blocks(BLOCKS);
	dim3 threads(inst.mdim, inst.mdim);

	if(inst.cond_right == COND_UPPER_RIGHT && inst.num_matrices == 2) {
		switch(inst.mdim) {
		case 5:
			all_in_one_kernel<2, 5, COND_UPPER_RIGHT><<<blocks, threads, space>>>(inst, stack, top, asteps);
			CUDA_CALL(cudaGetLastError());
			break;
		case 6:
			all_in_one_kernel<2, 6, COND_UPPER_RIGHT><<<blocks, threads, space>>>(inst, stack, top, asteps);
			CUDA_CALL(cudaGetLastError());
			break;
		case 7:
			all_in_one_kernel<2, 7, COND_UPPER_RIGHT><<<blocks, threads, space>>>(inst, stack, top, asteps);
			CUDA_CALL(cudaGetLastError());
			break;
		case 8:
			all_in_one_kernel<2, 8, COND_UPPER_RIGHT><<<blocks, threads, space>>>(inst, stack, top, asteps);
			CUDA_CALL(cudaGetLastError());
			break;
		case 9:
			all_in_one_kernel<2, 9, COND_UPPER_RIGHT><<<blocks, threads, space>>>(inst, stack, top, asteps);
			CUDA_CALL(cudaGetLastError());
			break;
		case 10:
			all_in_one_kernel<2, 10, COND_UPPER_RIGHT><<<blocks, threads, space>>>(inst, stack, top, asteps);
			CUDA_CALL(cudaGetLastError());
			break;
		case 11:
			all_in_one_kernel<2, 11, COND_UPPER_RIGHT><<<blocks, threads, space>>>(inst, stack, top, asteps);
			CUDA_CALL(cudaGetLastError());
			break;
		case 12:
			all_in_one_kernel<2, 12, COND_UPPER_RIGHT><<<blocks, threads, space>>>(inst, stack, top, asteps);
			CUDA_CALL(cudaGetLastError());
			break;
		case 13:
			all_in_one_kernel<2, 13, COND_UPPER_RIGHT><<<blocks, threads, space>>>(inst, stack, top, asteps);
			CUDA_CALL(cudaGetLastError());
			break;
		case 14:
			all_in_one_kernel<2, 14, COND_UPPER_RIGHT><<<blocks, threads, space>>>(inst, stack, top, asteps);
			CUDA_CALL(cudaGetLastError());
			break;
		case 15:
			all_in_one_kernel<2, 15, COND_UPPER_RIGHT><<<blocks, threads, space>>>(inst, stack, top, asteps);
			CUDA_CALL(cudaGetLastError());
			break;
		case 16:
			all_in_one_kernel<2, 16, COND_UPPER_RIGHT><<<blocks, threads, space>>>(inst, stack, top, asteps);
			CUDA_CALL(cudaGetLastError());
			break;
		}
	}
}
