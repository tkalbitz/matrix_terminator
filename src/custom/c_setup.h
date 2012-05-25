/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef C_SETUP_H_
#define C_SETUP_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "c_instance.h"

__global__ void setup_c_rnd_kernel(struct c_instance inst, int blocks, const int seed);
__global__ void setup_instances_kernel(struct c_instance inst);
__global__ void setup_global_particle_kernel(struct c_instance inst);
__global__ void setup_best_kernel(struct c_instance inst);
__global__ void patch_matrix_kernel(struct c_instance inst);
void setup_rating(struct c_instance& inst, int blocks);

#endif /* C_SETUP_H_ */
