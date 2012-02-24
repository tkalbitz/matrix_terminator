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
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "pso_instance.h"

__global__ void setup_rating(struct pso_instance * const inst);

__global__ void setup_pso_rnd_kernel(curandState* const rnd_states, const int seed);
__global__ void setup_particle_kernel(struct pso_instance *inst);
__global__ void setup_global_particle_kernel(struct pso_instance * const inst);
__global__ void setup_col_permut(int* const col_permut,
		                 const int total,
		                 const int width_per_line);


#endif /* PSO_SETUP_H_ */
