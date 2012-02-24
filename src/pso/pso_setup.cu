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

	double val = factor * inst->delta;
	if(val < 1.0)
		return 1.0;

	return factor * inst->delta;
}

__global__ void setup_rnd_kernel(curandState* const rnd_states,
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
setup_particle_kernel(struct pso_instance * const inst, bool half)
{
	if(half && blockIdx.x >= 16)
		return;

	const int id = get_thread_id();
	curandState rnd = inst->rnd_states[id];

	const uint32_t tp = threadIdx.x % inst->dim.matrix_width;

	const size_t p_pitch = inst->dev_particle.pitch;
	char* const p_slice = ((char*)inst->dev_particle.ptr) + blockIdx.x *
			              p_pitch * inst->dim.matrix_height;
	double* p_row = (double*) (p_slice + (threadIdx.x / inst->dim.matrix_height) * p_pitch);

	const size_t v_pitch = inst->dev_velocity.pitch;
	char* const v_slice = ((char*)inst->dev_velocity.ptr) + blockIdx.x *
			              v_pitch * inst->dim.matrix_height;
	double* v_row = (double*) (v_slice + (threadIdx.x / inst->dim.matrix_height) * v_pitch);

	const int max1 = (int)inst->parent_max;
	const int max2 = (int)inst->parent_max / 2;
	const int width = inst->width_per_inst;
	const int end = inst->dim.particles * width;
	uint8_t flag = 1;

	for(uint32_t x = tp; x < end; x += inst->dim.matrix_width) {

		//init velocity of particle
		v_row[x] = curand_uniform(&rnd) * inst->parent_max;
		if(curand_normal(&rnd) < 0)
			v_row[x] = -v_row[x];

		if(x % width == 0) {
			flag = (flag + 1) & 1;
		}

		if(curand_uniform(&rnd) < MATRIX_TAKEN_POS) {
	                if(flag) {
	                	p_row[x] = curand(&rnd) % max1 ;
	                } else {
	                        p_row[x] = min(max(0., curand_normal(&rnd)*(curand(&rnd) % max2) + max2), inst->parent_max);
	                }
		} else {
			p_row[x] = 0;
		}
	}

	inst->rnd_states[id] = rnd;

	if(threadIdx.x == 0) {
		const int matrices = inst->num_matrices * inst->dim.particles;
		int y;
		int i;

		if(inst->cond_left == COND_UPPER_LEFT) {
			y = 0;
			p_row = (double*) (p_slice + y * p_pitch);

			for(i = 0; i < matrices; i++) {
				p_row[i * inst->dim.matrix_width] = evo_mut_new_value(inst, &rnd);
			}
		} else if(inst->cond_left == COND_UPPER_RIGHT) {
			y = 0;
			p_row = (double*) (p_slice + y * p_pitch);

			for(i = 0; i < matrices; i++) {
				int idx = i * inst->dim.matrix_width + (inst->dim.matrix_width - 1);
				p_row[idx] = evo_mut_new_value(inst, &rnd);
			}
		} else if(inst->cond_left == COND_UPPER_LEFT_LOWER_RIGHT) {
			y = 0;
			p_row = (double*) (p_slice + y * p_pitch);
			for(i = 0; i < matrices; i++) {
				p_row[i * inst->dim.matrix_width] = evo_mut_new_value(inst, &rnd);
			}

			y = (inst->dim.matrix_height - 1);
			p_row = (double*) (p_slice + y * p_pitch);
			for(i = 0; i < matrices; i++) {
				int idx = i * inst->dim.matrix_width + (inst->dim.matrix_width - 1);
				p_row[idx] = evo_mut_new_value(inst, &rnd);
			}
		}
	}
	inst->rnd_states[id] = rnd;
}

__global__ void setup_param(struct pso_instance * const inst,
			    const double weigth,
			    const double c1,
			    const double c2, bool half)
{
	if(half && blockIdx.x >= 16)
		return;

	struct memory mem;
	pso_init_mem(inst, &mem);
	mem.param[3 * tx]     = weigth;
	mem.param[3 * tx + 1] = c1;
	mem.param[3 * tx + 2] = c2;
}

__global__ void setup_rating(struct pso_instance * const inst)
{
	struct memory mem;
	pso_init_mem(inst, &mem);
	mem.lb_rat[tx] = FLT_MAX;
	mem.p_rat[tx]  = FLT_MAX;

	if(tx == 0)
		inst->gb_rat[blockIdx.x] = FLT_MAX;
}
