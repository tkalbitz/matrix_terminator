#include <float.h>
#include <math.h>

#include <curand_kernel.h>

#include "pso.h"
#include "pso_config.h"
#include "pso_memory.h"

#if __CUDA_ARCH__ >= 200
	#define DIV(x,y) __ddiv_rn((x),(y))
#else
	#define DIV(x,y) ((x)/(y))
#endif

#define PRED_STEP(s) if(tx < (s)) { \
			if(shm_rat[tx] > shm_rat[tx + (s)]) { \
				shm_rat[tx] = shm_rat[tx + (s)]; \
				shm_pos[tx] = shm_pos[tx + (s)]; \
			}}

/**
 * can only be launched with PARTICLE_COUNT threads
 */
__global__ void pso_evaluation_lbest(struct pso_instance inst,
		                     const int s)
{
	__shared__ double shm_rat[PARTICLE_COUNT];
	__shared__ int    shm_pos[PARTICLE_COUNT];

	const int block_pos = BLOCK_POS2;
	const int s_count = inst.width_per_line / s;
	const int * const col_permut = inst.col_permut + inst.width_per_line * bx;

	const double * const prat = inst.prat  + s_count * bx * PARTICLE_COUNT;
	double * const lbrat      = inst.lbrat + s_count * bx * PARTICLE_COUNT;
	double * const gbrat      = inst.gbrat + s_count * bx;

	double* const particle = inst.particle;
	double* const particle_lbest = inst.particle_lbest;
	double* const particle_gbest = inst.particle_gbest + bx * inst.width_per_line * PARTICLE_COUNT;

	for(int i = 0; i < s_count; i+= PARTICLE_COUNT){

		//copy rating to shm
		shm_rat[tx] = lbrat[i + tx];
		shm_pos[tx] = tx;

		//copy better parts
		if(shm_rat[tx] > prat[i + tx]) {
			for(int j = 0; j < s; j++) {
				const int idx = ELEM_BIDX(block_pos, tx, col_permut[i * s + j]);
				particle_lbest[idx + j] = particle[idx + j];
			}

			shm_rat[tx] = lbrat[i + tx] = prat[i + tx];
		}

		__syncthreads();

		//reduction step
		if (PARTICLE_COUNT >= 256) { PRED_STEP(128); __syncthreads(); }
		if (PARTICLE_COUNT >= 128) { PRED_STEP(64);  __syncthreads(); }
		if (PARTICLE_COUNT >=  64) { PRED_STEP(32);  __syncthreads(); }
		if (PARTICLE_COUNT >=  32) { PRED_STEP(16);  __syncthreads(); }
		if (PARTICLE_COUNT >=  16) { PRED_STEP( 8);  __syncthreads(); }
		if (PARTICLE_COUNT >=   8) { PRED_STEP( 4);  __syncthreads(); }
		if (PARTICLE_COUNT >=   4) { PRED_STEP( 2);  __syncthreads(); }
		if (PARTICLE_COUNT >=   2) { PRED_STEP( 1);  __syncthreads(); }

		//copy step
		if(gbrat[i] > shm_rat[0]) {
			for(int j = tx; j < s; j += blockDim.x) {
				const int col = col_permut[i * s + j];
				const int idx = ELEM_BIDX(block_pos, shm_pos[0], col);
				particle_gbest[col] = particle_lbest[idx + j];
			}
			gbrat[i] = shm_rat[0];
		}
	}

	__syncthreads();

}

__global__ void pso_neighbor_best(struct pso_instance inst, const int s)
{
	const int s_count = inst.width_per_line / s;

	double * const lbrat  = inst.lbrat     + s_count * bx * PARTICLE_COUNT;
	int    * const lb_idx = inst.lbest_idx + s_count * bx * PARTICLE_COUNT;

	int n_particle = tx + 1;
	int p_particle = tx - 1;

	if(n_particle == PARTICLE_COUNT) {
		n_particle = 0;
	}

	if(p_particle == -1) {
		p_particle = PARTICLE_COUNT - 1;
	}

	for(int i = 0; i < s_count; i++) {
		const double lb_rat_p = lbrat[i * PARTICLE_COUNT + p_particle];
		const double lb_rat_c = lbrat[i * PARTICLE_COUNT + tx];
		const double lb_rat_n = lbrat[i * PARTICLE_COUNT + n_particle];
		const double res = min(min(lb_rat_p, lb_rat_c), lb_rat_n);

		int particle;

		if(res == lb_rat_p) {
			particle = p_particle;
		} else if(res == lb_rat_c) {
			particle = tx;
		} else if(res == lb_rat_n) {
			particle = n_particle;
		}

		lb_idx[i * PARTICLE_COUNT + tx] = particle;
	}
}

__device__ float curand_cauchy(curandState* rnd)
{
	float v = 0.0f;

	do {
		v = curand_normal(rnd);
	} while(v == 0);

	return curand_normal(rnd) / v;
//	return tan(M_PI * curand_uniform(rnd));
}

__device__ static double pso_mut_new_value(struct pso_instance & inst,
					   curandState         * const rnd_state)
{
	/* we want to begin with small numbers */
	const int tmp = (inst.parent_max > 10) ? 10 : (int)inst.parent_max;

	const int rnd_val = (curand(rnd_state) % (tmp - 1)) + 1;
	int factor = (int)(rnd_val / inst.delta);
	if((factor * inst.delta) < 1.0)
		factor++;

	if(factor * inst.delta < 1.0)
		return 1;

	return factor * inst.delta;
}

__device__ void pso_ensure_constraints(struct pso_instance & inst,
				       curandState         * const rnd_state,
				       double              * const elems)
{
	const int end = inst.width_per_line;
	const int width_per_matrix = inst.width_per_matrix;
	const int width_per_line = inst.dim.matrix_width;

	for(int start = 0; start < end; start += width_per_matrix) {
		if(inst.cond_left == COND_UPPER_LEFT) {
			if(elems[start] < 1.0)
				elems[start] = pso_mut_new_value(inst, rnd_state);
		} else if(inst.cond_left == COND_UPPER_RIGHT) {
			if(elems[start + width_per_line] < 1.0)
				elems[start + width_per_line] = pso_mut_new_value(inst, rnd_state);
		} else if(inst.cond_left == COND_UPPER_LEFT_LOWER_RIGHT) {
			if(elems[start] < 1.0)
				elems[start] = pso_mut_new_value(inst, rnd_state);

			if(elems[start + width_per_matrix - 1] < 1.0)
				elems[start + width_per_matrix - 1] = pso_mut_new_value(inst, rnd_state);
		} else {
			/*
			 * This should be recognized ;) It's only a 1.3 card
			 *  so there is no printf :/
			 */
			for(int i = 0; i < width_per_matrix; i++) {
				elems[start + i] = 1337;
			}
		}
	}
}

__global__ void pso_swarm_step_ccpso2(struct pso_instance inst, const int s)
{
	const double delta = inst.delta;
	const int id = get_thread_id();
	curandState rnd_state = inst.rnd_states[id];

	double* const elems = inst.particle;
	double* const particle_lbest = inst.particle_lbest;


	const int* const col_perm = inst.col_permut + bx * inst.width_per_line;
	const int col_start = tx / PARTICLE_COUNT;
	const int col_add   = blockDim.x / PARTICLE_COUNT;
	const int end	    = inst.width_per_line;
	const int particle  = tx - col_start * PARTICLE_COUNT;
	const int block_pos = BLOCK_POS2;

	const int* const lb_idx = inst.lbest_idx + (end / s) * bx * PARTICLE_COUNT;

	for(int i = col_start; i < end; i += col_add) {
		const int idx   = ELEM_BIDX(block_pos, particle, col_perm[i]);
		const int cur_s = i / s;
		const double lb  = particle_lbest[idx];
		const double lbn = particle_lbest[ELEM_BIDX(block_pos, lb_idx[cur_s], col_perm[i])];

		double xi = elems[idx];

		if(curand_uniform(&rnd_state) <= 0.5)
			xi = lb  + curand_cauchy(&rnd_state) * abs((lb - lbn));
		else
			xi = lbn + curand_normal(&rnd_state) * abs((lb - lbn));

		xi = __dmul_rn(__double2uint_rn(DIV(xi, delta)), delta);
		xi = min(inst.parent_max, max(0., xi));

		elems[idx] = xi;
	}

	__syncthreads();

	if(tx < PARTICLE_COUNT) {
		const int idx   = ELEM_BIDX(block_pos, tx, 0);
		pso_ensure_constraints(inst, &rnd_state, elems + idx);
	}

	inst.rnd_states[id] = rnd_state;
}
