/*
 * c_setup.h
 *
 *  Created on: Feb 8, 2012
 *      Author: tkalbitz
 */

#ifndef C_SETUP_H_
#define C_SETUP_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "c_instance.h"

__global__ void setup_c_rnd_kernel(struct c_instance inst, const int seed);
__global__ void setup_instances_kernel(struct c_instance inst);
__global__ void setup_global_particle_kernel(struct c_instance inst);
__global__ void setup_best_kernel(struct c_instance inst);

#endif /* C_SETUP_H_ */
