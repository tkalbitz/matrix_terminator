#include <cuda.h>
#include <curand_kernel.h>

#include "instance.h"
#include "config.h"

/* calculate the thread id for the current block topology */
inline __device__ int get_thread_id() {
	return threadIdx.x + blockIdx.x * blockDim.x;
}

__global__ void setup_rnd_kernel(curandState* rnd_states,
				 int seed)
{
	int id = get_thread_id();

	/* 
         * Each thread get the same seed, 
         * a different sequence number and no offset. 
         */
	curand_init(seed, id, 0, &rnd_states[id]);
}

/*
 * Initialize the parent memory with random values.
 */
__global__ void
setup_parent_kernel(struct instance *inst)
{
	int id = get_thread_id();
	curandState rnd_state = inst->rnd_states[id];

	char* devPtr = (char*)inst->dev_parent.ptr;
	size_t pitch = inst->dev_parent.pitch;
	size_t slicePitch = pitch * inst->dim.matrix_height;

	int z = blockIdx.x;
//	int x = threadIdx.x;
	int y = threadIdx.x;

	char* slice = devPtr + z * slicePitch;
	float* row = (float*) (slice + y * pitch);

	for(int x = 0; x < inst->dim.threads * inst->dim.matrix_width; x++) {
		if(curand_uniform(&rnd_state) < MATRIX_TAKEN_POS) {
			row[x] = curand(&rnd_state);
		}
	}

	inst->rnd_states[id] = rnd_state;

	if(threadIdx.x != 0)
		return;

	if(inst->cond_left == COND_UPPER_LEFT) {
		y = 0;
		row = (float*) (slice + y * pitch);
		row[0] = 1;
	} else if(inst->cond_left == COND_UPPER_RIGHT) {
		y = 0;
		row = (float*) (slice + y * pitch);
		int x = (inst->dim.matrix_width - 1);
		row[x] = 1;
	} else if(inst->cond_left == COND_UPPER_LEFT_LOWER_RIGHT) {
		y = 0;
		row = (float*) (slice + y * pitch);
		row[0] = 1;

		y = (inst->dim.matrix_height - 1);
		int x = (inst->dim.matrix_width - 1);
		row = (float*) (slice + y * pitch);
		row[x] = 1;
	}
}


__global__ void evo_kernel(struct instance *inst)
{
	int id = get_thread_id();

	/* copy global state to local mem for efficiency */
	curandState rnd_state = inst->rnd_states[id];

		

	int x = curand(&rnd_state);

	/* backup rnd state to global mem */
	inst->rnd_states[id] = rnd_state;
}
