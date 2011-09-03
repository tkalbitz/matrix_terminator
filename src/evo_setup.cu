/*
 * evo_setup.cu
 *
 *  Created on: Jun 24, 2011
 *      Author: tkalbitz
 */

#include "evo_setup.h"
#include "evo_memory.cu"

__global__ void setup_rnd_kernel(curandState* const rnd_states,
				 const int seed)
{
	const int id = get_thread_id();

	/*
         * Each thread get the same seed,
         * a different sequence number and no offset.
         */
	curand_init(seed, id, 0, &rnd_states[id]);
}

__device__ static double evo_mut_new_value(struct instance * const inst,
					   curandState     * const rnd_state)
{
	/* we want to begin with small numbers */
	const int tmp = (inst->parent_max > 10) ? 10 : (int)inst->parent_max;
//	const int tmp = (int)inst->parent_max;

	const int rnd_val = (curand(rnd_state) % (tmp - 1)) + 1;
	int factor = (int)(rnd_val / inst->delta);
	if((factor * inst->delta) < 1.0)
		factor++;

	double val = factor * inst->delta;
	if(val < 1.0)
		return 1.0;

	return factor * inst->delta;
}

/*
 * Initialize the parent memory with random values.
 */
__global__ void setup_parent_kernel(struct instance * const inst)
{
	if(threadIdx.x >= inst->dim.matrix_height)
		return;

	const int id = get_thread_id();
	curandState rnd = inst->rnd_states[id];

	char* const devPtr = (char*)inst->dev_parent.ptr;
	const size_t pitch = inst->dev_parent.pitch;
	const size_t slicePitch = pitch * inst->dim.matrix_height;
	char* const slice = devPtr + blockIdx.x * slicePitch;
	double* row = (double*) (slice + threadIdx.x * pitch);

	const int tmp = (int)inst->parent_max;

	for(int x = 0; x < inst->dim.parents * inst->width_per_inst; x++) {
		if(curand_uniform(&rnd) < MATRIX_TAKEN_POS) {
			row[x] = curand(&rnd) % tmp;
		} else {
			row[x] = 0;
		}
	}

	inst->rnd_states[id] = rnd;

	if(threadIdx.x != 0)
		return;

	const int matrices = inst->num_matrices * inst->dim.parents;
	int y;

	if(inst->cond_left == COND_UPPER_LEFT) {
		y = 0;
		row = (double*) (slice + y * pitch);

		for(int i = 0; i < matrices; i++) {
			row[i * MATRIX_WIDTH] = evo_mut_new_value(inst, &rnd);
		}
	} else if(inst->cond_left == COND_UPPER_RIGHT) {
		y = 0;
		row = (double*) (slice + y * pitch);

		for(int i = 0; i < matrices; i++) {
			int idx = i * MATRIX_WIDTH + (MATRIX_WIDTH - 1);
			row[idx] = evo_mut_new_value(inst, &rnd);
		}
	} else if(inst->cond_left == COND_UPPER_LEFT_LOWER_RIGHT) {
		y = 0;
		row = (double*) (slice + y * pitch);
		for(int i = 0; i < matrices; i++) {
			row[i * MATRIX_WIDTH] = evo_mut_new_value(inst, &rnd);
		}

		y = (inst->dim.matrix_height - 1);
		row = (double*) (slice + y * pitch);
		for(int i = 0; i < matrices; i++) {
			int idx = i * MATRIX_WIDTH + (MATRIX_WIDTH - 1);
			row[idx] = evo_mut_new_value(inst, &rnd);
		}
	}

	inst->rnd_states[id] = rnd;
}

/*
 * Initialize the child memory with random values.
 */
__global__ void setup_childs_kernel(struct instance * const inst)
{
	if(threadIdx.x >= MATRIX_HEIGHT)
		return;

	const int id = get_thread_id();
	curandState rnd = inst->rnd_states[id];

	char* const devPtr = (char*)inst->dev_child.ptr;
	const size_t pitch = inst->dev_child.pitch;
	const size_t slicePitch = pitch * inst->dim.matrix_height;
	char* const slice = devPtr + blockIdx.x * slicePitch;
	double* row = (double*) (slice + threadIdx.x * pitch);

//	const int tmp = (inst->parent_max > 10) ? 10 : (int)inst->parent_max;
	const int max1 = (int)inst->parent_max;
	const int max2 = (int)inst->parent_max / 2;
	const int width = inst->width_per_inst;
	int flag = 1;

	for(int x = 0; x < CHILDS * PARENTS * width; x++) {

		if(x % width == 0) {
			flag = (flag + 1) & 1;
		}

		if(curand_uniform(&rnd) < MATRIX_TAKEN_POS) {
			if(flag)
				row[x] = curand(&rnd) % max1;
			else
				row[x] = min(max(0., curand_normal(&rnd)*(curand(&rnd) % max2) + max2), inst->parent_max);

		} else {
			row[x] = 0;
		}
	}

	inst->rnd_states[id] = rnd;

	if(threadIdx.x != 0)
		return;

	const int matrices = inst->num_matrices * CHILDS * PARENTS;
	int y;

	if(inst->cond_left == COND_UPPER_LEFT) {
		y = 0;
		row = (double*) (slice + y * pitch);

		for(int i = 0; i < matrices; i++) {
			row[i * MATRIX_WIDTH] = evo_mut_new_value(inst, &rnd);
		}
	} else if(inst->cond_left == COND_UPPER_RIGHT) {
		y = 0;
		row = (double*) (slice + y * pitch);

		for(int i = 0; i < matrices; i++) {
			int idx = i * MATRIX_WIDTH + (MATRIX_WIDTH - 1);
			row[idx] = evo_mut_new_value(inst, &rnd);
		}
	} else if(inst->cond_left == COND_UPPER_LEFT_LOWER_RIGHT) {
		y = 0;
		row = (double*) (slice + y * pitch);
		for(int i = 0; i < matrices; i++) {
			row[i * MATRIX_WIDTH] = evo_mut_new_value(inst, &rnd);
		}

		y = (inst->dim.matrix_height - 1);
		row = (double*) (slice + y * pitch);
		for(int i = 0; i < matrices; i++) {
			int idx = i * MATRIX_WIDTH + (MATRIX_WIDTH - 1);
			row[idx] = evo_mut_new_value(inst, &rnd);
		}
	}

	inst->rnd_states[id] = rnd;
}

__global__ void setup_sparam(struct instance * const inst,
			     const double sparam,
			     const double mut_rate,
			     const double recomb_rate)
{
	struct memory mem;
	evo_init_mem(inst, &mem);
	mem.sparam[3 * tx]     = sparam;
	mem.sparam[3 * tx + 1] = mut_rate;
	mem.sparam[3 * tx + 2] = recomb_rate;

	if(tx < PARENTS) {
		mem.psparam[3 * tx]     = sparam;
		mem.psparam[3 * tx + 1] = mut_rate;
		mem.psparam[3 * tx + 2] = recomb_rate;
	}
}
