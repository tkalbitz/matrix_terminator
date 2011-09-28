/*
 * pso_setup.cu
 *
 *  Created on: Sep 28, 2011
 *      Author: tkalbitz
 */

#include "pso_setup.h"

/* calculate the thread id for the current block topology */
__device__ inline static int get_thread_id() {
	return threadIdx.x + blockIdx.x * blockDim.x;
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
