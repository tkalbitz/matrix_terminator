/*
 * pso_setup.h
 *
 *  Created on: Sep 28, 2011
 *      Author: tkalbitz
 */

#ifndef PSO_SETUP_H_
#define PSO_SETUP_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "pso_instance.h"

__global__ void setup_rating(struct pso_instance * const inst);

__global__ void setup_rnd_kernel(curandState* const rnd_states, const int seed);
__global__ void setup_particle_kernel(struct pso_instance *inst);
__global__ void setup_global_particle_kernel(struct pso_instance * const inst);
__global__ void setup_col_permut(int* const col_permut,
		                 const int total,
		                 const int width_per_line);


#endif /* PSO_SETUP_H_ */
