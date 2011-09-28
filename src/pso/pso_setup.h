/*
 * pso_setup.h
 *
 *  Created on: Sep 28, 2011
 *      Author: tkalbitz
 */

#ifndef PSO_SETUP_H_
#define PSO_SETUP_H_

#include <cuda.h>
#include <curand_kernel.h>

__global__ void setup_rnd_kernel(curandState* const rnd_states, const int seed);

#endif /* PSO_SETUP_H_ */
