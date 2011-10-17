#include <float.h>

#include "pso.h"
#include "pso_config.h"
#include "pso_memory.h"

__global__ void pso_evaluation_lbest(struct pso_instance* inst)
{
	struct memory m;
	struct memory* mem = &m;
	pso_init_mem(inst, mem);

	if(mem->lb_rat[blockIdx.y] < mem->p_rat[blockIdx.y]) {
		return;
	}

	for(int i = 0; i < inst->num_matrices; i++) {
		int delta = mem->p_zero + i * inst->dim.matrix_width;
		LB_ROW(ty)[delta + tx] = P_ROW(ty)[delta + tx];
	}

	mem->lb_rat[blockIdx.y] = mem->p_rat[blockIdx.y];
}

__global__ void pso_evaluation_gbest(struct pso_instance* inst)
{
	struct memory m;
	struct memory* mem = &m;
	pso_init_mem(inst, mem);

	__shared__ int pidx;

	if(tx == 0 && ty == 0) {
		double rat = FLT_MAX;
		for(int i = 0; i < PARTICLE_COUNT; i++) {
			if(rat > mem->lb_rat[i]) {
				rat = mem->lb_rat[i];
				pidx = i;
			}
		}
	}

	__syncthreads();

	int src = pidx * inst->width_per_inst;
	for(int i = 0; i < inst->num_matrices; i++) {
		int delta = i * inst->dim.matrix_width;
		GB_ROW(ty)[delta + tx] = LB_ROW(ty)[src + delta + tx];
	}
}

__global__ void pso_swarm_step(struct pso_instance* inst)
{

}
