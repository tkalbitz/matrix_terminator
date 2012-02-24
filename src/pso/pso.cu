/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#include <float.h>

#include <curand_kernel.h>

#include "pso.h"
#include "pso_config.h"
#include "pso_memory.h"

#if __CUDA_ARCH__ >= 200
	#define DIV(x,y) __ddiv_rn((x),(y))
#else
	#define DIV(x,y) ((x)/(y))
#endif

__global__ void pso_evaluation_lbest(struct pso_instance* inst)
{
	struct memory m;
	struct memory* mem = &m;
	pso_init_mem(inst, mem);

	if(mem->lb_rat[blockIdx.y] > mem->p_rat[blockIdx.y]) {
		for(int i = 0; i < inst->num_matrices; i++) {
			int delta = mem->p_zero + i * inst->dim.matrix_width;
			LB_ROW(ty)[delta + tx] = P_ROW(ty)[delta + tx];
		}

		if(tx == 0 && ty == 0)
			mem->lb_rat[blockIdx.y] = mem->p_rat[blockIdx.y];
	}
}

__global__ void pso_evaluation_gbest(struct pso_instance* inst)
{
	struct memory m;
	struct memory* mem = &m;
	pso_init_mem(inst, mem);

	__shared__ int pidx;

	if(tx == 0 && ty == 0) {
		pidx = -1;

		double rat = inst->gb_rat[blockIdx.x];
		for(int i = 0; i < PARTICLE_COUNT; i++) {
			if(rat > mem->lb_rat[i]) {
				rat = mem->lb_rat[i];
				pidx = i;
			}
		}
	}

	__syncthreads();

	if(pidx == -1)
		return;

	int src = pidx * inst->width_per_inst;
	for(int i = 0; i < inst->num_matrices; i++) {
		int delta = i * inst->dim.matrix_width;
		GB_ROW(ty)[delta + tx] = LB_ROW(ty)[src + delta + tx];
	}

	if(tx == 0 && ty == 0)
		inst->gb_rat[blockIdx.x] = mem->lb_rat[pidx];
}


__device__ static double pso_mut_new_value(struct pso_instance * const inst,
					   curandState         * const rnd_state)
{
	/* we want to begin with small numbers */
	const int tmp = (inst->parent_max > 10) ? 10 : (int)inst->parent_max;

	const int rnd_val = (curand(rnd_state) % (tmp - 1)) + 1;
	int factor = (int)(rnd_val / inst->delta);
	if((factor * inst->delta) < 1.0)
		factor++;

	if(factor * inst->delta < 1.0)
		return 1;

	return factor * inst->delta;
}

__device__ void pso_ensure_constraints(struct pso_instance * const inst,
				       struct memory       * const mem,
				       curandState         * const rnd_state)
{
	double* const row   = P_ROW(0);
	double* const lrow  = P_ROW(inst->dim.matrix_height-1);

	const int end = mem->p_end;

	for(int start = mem->p_zero; start < end; start += inst->dim.matrix_width) {
		const int lidx = start + inst->dim.matrix_width - 1;

		if(inst->cond_left == COND_UPPER_LEFT) {
			if(row[start] < 1.0)
			row[start] = pso_mut_new_value(inst, rnd_state);
		} else if(inst->cond_left == COND_UPPER_RIGHT) {
			if(row[lidx] < 1.0)
				row[lidx] = pso_mut_new_value(inst, rnd_state);
		} else if(inst->cond_left == COND_UPPER_LEFT_LOWER_RIGHT) {
			if(row[start] < 1.0)
				row[start] = pso_mut_new_value(inst, rnd_state);

			if(lrow[lidx] < 1.0)
				lrow[lidx] = pso_mut_new_value(inst, rnd_state);
		} else {
			/*
			 * This should be recognized ;) It's only a 1.3 card
			 *  so there is no printf :/
			 */
			for(int i = 0; i < inst->dim.matrix_width; i++) {
				row[start + i] = 1337;
				lrow[start + i] = 1337;
			}
		}
	}
}


__device__ void pso_neighbor_best(struct pso_instance* const inst,
		                  struct memory*       const mem)
{
	int n_block    = blockIdx.x;
	int n_particle = blockIdx.y + 1;
	int p_block    = blockIdx.x;
	int p_particle = blockIdx.y - 1;

	if(n_particle == PARTICLE_COUNT) {
		n_particle = 0;
		n_block++;

		if(n_block == BLOCKS)
			n_block = 0;
	}

	if(p_particle == -1) {
		p_particle = PARTICLE_COUNT - 1;
		p_block--;

		if(p_block == -1)
			p_block = BLOCKS - 1;
	}

	const char* const lbrat_ptr = (char*)inst->dev_lbrat.ptr;
	const double lb_rat_p = ((double*)(lbrat_ptr + p_block    * inst->dev_lbrat.pitch))[p_particle];
	const double lb_rat_c = ((double*)(lbrat_ptr + blockIdx.x * inst->dev_lbrat.pitch))[blockIdx.y];
	const double lb_rat_n = ((double*)(lbrat_ptr + n_block    * inst->dev_lbrat.pitch))[n_particle];

	const double res = min(min(lb_rat_p, lb_rat_c), lb_rat_n);

	int block;
	int particle;

	if(res == lb_rat_p) {
		block = p_block;
		particle = p_particle;
	} else if(res == lb_rat_c) {
		block = blockIdx.x;
		particle = blockIdx.y;
	} else if(res == lb_rat_n) {
		block = n_block;
		particle = n_particle;
	}

	char* const  lbest_dev_ptr = (char*)inst->dev_particle_lbest.ptr;
	const size_t lbest_pitch = inst->dev_particle_lbest.pitch;
	const size_t lbest_slice_pitch = lbest_pitch * inst->dim.matrix_height;
	char* const  lbest_slice = lbest_dev_ptr + block /* z */ * lbest_slice_pitch;
	mem->lbn_pitch = lbest_pitch;
	mem->lbn_slice = lbest_slice;
	mem->lbn_zero = inst->width_per_inst * particle;
}

__global__ void pso_swarm_step(struct pso_instance* inst)
{
	__shared__ struct memory m;
	__shared__ double w;
	__shared__ double c1;
	__shared__ double c2;

	struct memory* mem = &m;

	int id = get_thread_id();
	curandState rnd_state = inst->rnd_states[id];

	if(tx == 0 && ty == 0) {
		pso_init_mem(inst, mem);
		pso_neighbor_best(inst, mem);
		w = W(blockIdx.y);
		c1 = C1(blockIdx.y);
		c2 = C2(blockIdx.y);
	}
	__syncthreads();

	const double delta = inst->delta;

	for(int i = 0; i < inst->num_matrices; i++) {
		const int e_idx = i * inst->dim.matrix_width + tx;
		const int p_idx = mem->p_zero + e_idx;
		const int n_idx = mem->lbn_zero + e_idx;

		double xi = P_ROW(ty)[p_idx];

		const double cog_part = curand_normal(&rnd_state) * c1 * (LB_ROW(ty) [p_idx] - xi);
		const double soc_part = curand_normal(&rnd_state) * c2 * (LBN_ROW(ty)[n_idx] - xi);

//		if(blockIdx.x < (BLOCKS / 2))
//			soc_part = curand_normal(&rnd_state) * c2 * (GB_ROW(ty)[e_idx] - xi);

		double tmp = max(inst->parent_max / 4, inst->delta);
		V_ROW(ty)[p_idx] = w * (V_ROW(ty)[p_idx] + cog_part + soc_part);
		V_ROW(ty)[p_idx] = min(max(V_ROW(ty)[p_idx], -tmp), tmp);

		xi = __dadd_rn(xi, V_ROW(ty)[p_idx]);
		/* we want x * delta, where x is an int */
		xi = __dmul_rn(((unsigned long)DIV(xi, delta)), delta);
		xi = min(inst->parent_max, max(0., xi));
		P_ROW(ty)[p_idx] = xi;
	}

	__syncthreads();

	if(tx == 0 && ty == 0) {
		pso_ensure_constraints(inst, mem, &rnd_state);
	}

	inst->rnd_states[id] = rnd_state;
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

__global__ void pso_swarm_step_ccpso(struct pso_instance* inst)
{
	__shared__ struct memory m;
	struct memory* const mem = &m;
	const double delta = inst->delta;
	const int id = get_thread_id();
	curandState rnd_state = inst->rnd_states[id];

	if(tx == 0 && ty == 0) {
		pso_init_mem(inst, mem);
		pso_neighbor_best(inst, mem);
	}
	__syncthreads();

	for(int i = 0; i < inst->num_matrices; i++) {
		const int e_idx = i * inst->dim.matrix_width + tx;
		const int p_idx = mem->p_zero + e_idx;
		const int n_idx = mem->lbn_zero + e_idx;

		double xi = P_ROW(ty)[p_idx];

		if(curand_uniform(&rnd_state) <= 0.5)
			xi = LB_ROW(ty)[p_idx] + curand_cauchy(&rnd_state) *
			     abs((LB_ROW(ty)[p_idx] - LBN_ROW(ty)[n_idx]));
		else
			xi = LBN_ROW(ty)[p_idx] + curand_normal(&rnd_state) *
			     abs((LB_ROW(ty)[p_idx] - LBN_ROW(ty)[n_idx]));

		xi = __dmul_rn(__double2uint_rn(DIV(xi, delta)), delta);
		xi = min(inst->parent_max, max(0., xi));
		P_ROW(ty)[p_idx] = xi;
	}
	__syncthreads();

	if(tx == 0 && ty == 0) {
		pso_ensure_constraints(inst, mem, &rnd_state);
	}

	inst->rnd_states[id] = rnd_state;
}
