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

#include "pso_instance.h"

__global__ void setup_rating(struct pso_instance * const inst);

__global__ void setup_rnd_kernel(curandState* const rnd_states, const int seed);
__global__ void setup_particle_kernel(struct pso_instance *inst, bool half);
__global__ void setup_param(struct pso_instance * const inst,
			    const double weigth,
			    const double c1,
			    const double c2, bool half);

#endif /* PSO_SETUP_H_ */
