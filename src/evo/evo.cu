/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#include <limits.h>
#include <float.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "instance.h"
#include "config.h"

#include "evo_memory.cu"

__device__ inline void copy_child_to_parent(struct instance * const inst,
					    struct memory   * const mem,
					    const int child,
					    const int parent)
{
	const uint32_t cstart = child  * inst->width_per_inst;
	const uint32_t pstart = parent * inst->width_per_inst;

	for(uint32_t i = 0; i < inst->width_per_inst; i += inst->dim.matrix_width)
		P_ROW(ty)[pstart + i + tx] = C_ROW(ty)[cstart + i + tx];
}

#include "evo_recombination.cu"
#include "evo_adaptive_gauss_mutation.cu"
#include "evo_selection.cu"

__global__ void evo_kernel_part_one(struct instance *inst)
{
	const int id = get_thread_id();

	/* copy global state to local mem for efficiency */
	curandState rnd_state = inst->rnd_states[id];

	struct memory mem;
	evo_init_mem(inst, &mem);
	evo_mutation(inst, &mem, &rnd_state);

	/* backup rnd state to global mem */
	inst->rnd_states[id] = rnd_state;
}

__global__ void evo_kernel_part_two(struct instance *inst)
{
	struct memory m;
	struct memory* const mem = &m;

	evo_init_mem(inst, &m);

	const int id = get_thread_id();
	curandState rnd_state = inst->rnd_states[id];
//	evo_parent_selection_convergence_prevention(inst, mem, &rnd_state, 0.6);
//	evo_parent_selection_turnier2(inst, mem, &rnd_state, 6, 0.5);
	evo_parent_selection_turnier(inst, mem, &rnd_state, 10);
	inst->rnd_states[id] = rnd_state;
	__syncthreads();

	/* Parallel copy of memory */
	for(int i = 0; i < PARENTS; i++) {
		const int child = (int)mem->c_rat[2 * i + 1];
		copy_child_to_parent(inst, mem, child, i);
		mem->p_rat[i] = mem->c_rat[2 * i];
		PSP(i) = SP(child);
		PMR(i) = MR(child);
		PRR(i) = RR(child);
	}
}

