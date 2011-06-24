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

	for(int x = 0; x < inst->dim.parents * inst->width_per_inst; x++) {
		if(curand_uniform(&rnd) < MATRIX_TAKEN_POS) {
			row[x] = curand(&rnd) % (int)PARENT_MAX;
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
			row[i * MATRIX_WIDTH] =
				(curand(&rnd) % ((int)PARENT_MAX - 1)) + 1;
		}
	} else if(inst->cond_left == COND_UPPER_RIGHT) {
		y = 0;
		row = (double*) (slice + y * pitch);

		for(int i = 0; i < matrices; i++) {
			int idx = i * MATRIX_WIDTH + (MATRIX_WIDTH - 1);
			row[idx] = (curand(&rnd) % ((int)PARENT_MAX - 1)) + 1;
		}
	} else if(inst->cond_left == COND_UPPER_LEFT_LOWER_RIGHT) {
		y = 0;
		row = (double*) (slice + y * pitch);
		for(int i = 0; i < matrices; i++) {
			row[i * MATRIX_WIDTH] =
				(curand(&rnd) % ((int)PARENT_MAX - 1)) + 1;
		}

		y = (inst->dim.matrix_height - 1);
		row = (double*) (slice + y * pitch);
		for(int i = 0; i < matrices; i++) {
			int idx = i * MATRIX_WIDTH + (MATRIX_WIDTH - 1);
			row[idx] = (curand(&rnd) % ((int)PARENT_MAX - 1)) + 1;
		}
	}

	inst->rnd_states[id] = rnd;
}

__global__ void setup_sparam(struct instance * const inst)
{
	get_sparam_arr(inst)[threadIdx.x] = PARENT_MAX * 0.01;
}
