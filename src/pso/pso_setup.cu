/*
 * pso_setup.cu
 *
 *  Created on: Sep 28, 2011
 *      Author: tkalbitz
 */

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

	const size_t pitch = inst->dev_particle.pitch;
	char* const slice = ((char*)inst->dev_particle.ptr) + blockIdx.x *
			                        pitch * inst->dim.matrix_height;
	double* row = (double*) (slice + (threadIdx.x / inst->dim.matrix_height) * pitch);

	const int max1 = (int)inst->parent_max;
	const int max2 = (int)inst->parent_max / 2;
	const int width = inst->width_per_inst;
	const int end = inst->dim.particles * width;
	uint8_t flag = 1;

	for(uint32_t x = tp; x < end; x += inst->dim.matrix_width) {

		if(x % width == 0) {
			flag = (flag + 1) & 1;
		}

		if(curand_uniform(&rnd) < MATRIX_TAKEN_POS) {
	                if(flag) {
	                	row[x] = curand(&rnd) % max1 ;
	                } else {
	                        row[x] = min(max(0., curand_normal(&rnd)*(curand(&rnd) % max2) + max2), inst->parent_max);
	                }
		} else {
			row[x] = 0;
		}
	}

	inst->rnd_states[id] = rnd;

	if(threadIdx.x != 0)
		return;

	const int matrices = inst->num_matrices * inst->dim.particles;
	int y;
	int i;

	if(inst->cond_left == COND_UPPER_LEFT) {
		y = 0;
		row = (double*) (slice + y * pitch);

		for(i = 0; i < matrices; i++) {
			row[i * inst->dim.matrix_width] = evo_mut_new_value(inst, &rnd);
		}
	} else if(inst->cond_left == COND_UPPER_RIGHT) {
		y = 0;
		row = (double*) (slice + y * pitch);

		for(i = 0; i < matrices; i++) {
			int idx = i * inst->dim.matrix_width + (inst->dim.matrix_width - 1);
			row[idx] = evo_mut_new_value(inst, &rnd);
		}
	} else if(inst->cond_left == COND_UPPER_LEFT_LOWER_RIGHT) {
		y = 0;
		row = (double*) (slice + y * pitch);
		for(i = 0; i < matrices; i++) {
			row[i * inst->dim.matrix_width] = evo_mut_new_value(inst, &rnd);
		}

		y = (inst->dim.matrix_height - 1);
		row = (double*) (slice + y * pitch);
		for(i = 0; i < matrices; i++) {
			int idx = i * inst->dim.matrix_width + (inst->dim.matrix_width - 1);
			row[idx] = evo_mut_new_value(inst, &rnd);
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

