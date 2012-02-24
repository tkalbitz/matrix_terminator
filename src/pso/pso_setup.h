/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
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
