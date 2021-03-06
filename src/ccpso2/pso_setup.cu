/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#include <float.h>

#include "pso_config.h"
#include "pso_setup.h"
#include "pso_memory.h"

__device__ static double evo_mut_new_value(struct pso_instance * const inst,
					   curandState         * const rnd_state)
{
	/* we want to begin with small numbers */
	const int tmp = (inst->parent_max > 10) ? 10 : (int)inst->parent_max;
	const int rnd_val = (curand(rnd_state) % (tmp - 1)) + 1;
	int factor = (int)(rnd_val / inst->delta);
	if((factor * inst->delta) < 1.0)
		factor++;

	const double val = factor * inst->delta;
	if(val < 1.0)
		return 1.0;

	return val;
}

__global__ void setup_pso_rnd_kernel(curandState* const rnd_states,
				 const int seed)
{
	const int id = get_thread_id();

	/*
         * Each thread get the same seed,
         * a different sequence number and no offset.
         */
	curand_init(seed + id, id, 0, &rnd_states[id]);
}

/*
 * Initialize the child memory with random values.
 */
__global__ void
setup_global_particle_kernel(struct pso_instance * const inst)
{
	const int id = get_thread_id();
	curandState rnd = inst->rnd_states[id];

	const int max1 = (int)inst->parent_max;
	int x;

	const int gbest_len = inst->width_per_line * BLOCKS;
	for(x = tx; x < gbest_len; x += blockDim.x) {
		inst->particle_gbest[x] = curand(&rnd) % max1 ;
	}

	__syncthreads();

	const int matrices = inst->num_matrices *
			     inst->dim.blocks;

	if(inst->cond_left == COND_UPPER_LEFT) {
		for(x = tx; x < matrices; x += blockDim.x) {
			const int matrix = x * inst->width_per_matrix;
                        inst->particle_gbest[matrix] = evo_mut_new_value(inst, &rnd);
		}
	} else if(inst->cond_left == COND_UPPER_RIGHT) {
		for(x = tx; x < matrices; x += blockDim.x) {
			const int matrix = x * inst->width_per_matrix +
					   inst->dim.matrix_width - 1;
                        inst->particle_gbest[matrix] = evo_mut_new_value(inst, &rnd);
		}
	} else if(inst->cond_left == COND_UPPER_LEFT_LOWER_RIGHT) {
		for(x = tx; x < matrices; x += blockDim.x) {
			const int matrix1 = x * inst->width_per_matrix;
			const int matrix2 = (x + 1) * inst->width_per_matrix - 1;
                        inst->particle_gbest[matrix1] = evo_mut_new_value(inst, &rnd);
                        inst->particle_gbest[matrix2] = evo_mut_new_value(inst, &rnd);
		}
	}
	inst->rnd_states[id] = rnd;
}

__global__ void
setup_particle_kernel(struct pso_instance * const inst)
{
	const int id = get_thread_id();
	curandState rnd = inst->rnd_states[id];

	const int max1 = (int)inst->parent_max;
	const int end = inst->total;
	int x;

	for(x = tx; x < end; x += blockDim.x) {
		inst->particle[x] = curand(&rnd) % max1 ;
	}

	__syncthreads();

	const int matrices = inst->num_matrices *
			     inst->dim.blocks;

	if(tx < PARTICLE_COUNT) {
		if(inst->cond_left == COND_UPPER_LEFT) {
			for(x = 0; x < matrices; x++) {
				const int matrix = x * inst->width_per_matrix * PARTICLE_COUNT + tx;
				inst->particle[matrix] = evo_mut_new_value(inst, &rnd);
			}
		} else if(inst->cond_left == COND_UPPER_RIGHT) {
			for(x = 0; x < matrices; x++) {
				const int matrix = x * inst->width_per_matrix * PARTICLE_COUNT +
						   inst->dim.matrix_width * PARTICLE_COUNT - 1 - tx;
				inst->particle[matrix] = evo_mut_new_value(inst, &rnd);
			}
		} else if(inst->cond_left == COND_UPPER_LEFT_LOWER_RIGHT) {
			for(x = 0; x < matrices; x++) {
				const int matrix1 = x * inst->width_per_matrix * PARTICLE_COUNT + tx;
				const int matrix2 = (x + 1) * inst->width_per_matrix * PARTICLE_COUNT - 1 - tx;
				inst->particle[matrix1] = evo_mut_new_value(inst, &rnd);
				inst->particle[matrix2] = evo_mut_new_value(inst, &rnd);
			}
		}
	}
	inst->rnd_states[id] = rnd;
}

__global__ void setup_rating(struct pso_instance * const inst)
{
	int i = 0;
	const int len = inst->width_per_line * inst->dim.particles *
			inst->dim.blocks;

	for(i = tx; i < len; i += blockDim.x) {
		inst->prat[i] = FLT_MAX;
		inst->lbrat[i] = FLT_MAX;
	}

	const int end = inst->dim.blocks;
	if(tx < end) {
		inst->gbrat[tx] = FLT_MAX;
	}

	//TODO
	if(tx < BLOCKS) {
		inst->s[tx] = 2;
	}
}

__global__ void setup_col_permut(int* const col_permut,
		                 const int total,
		                 const int width_per_line)
{
	int i;

	for(i = tx; i < total; i += blockDim.x) {
		col_permut[i] = (i % width_per_line);
	}
}
