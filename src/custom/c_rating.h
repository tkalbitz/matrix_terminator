/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef C_RATING_H_
#define C_RATING_H_

#include "c_instance.h"

__global__ void setup_rating(struct c_instance inst, const int yoff);
__global__ void copy_parent_kernel(struct c_instance inst);
__global__ void mutate_kernel(struct c_instance inst);
__global__ void rate_mutated_kernel(struct c_instance inst);
__global__ void copy_to_child_kernel(struct c_instance inst);
__global__ void path_mutate_kernel_p1(struct c_instance inst,
		                      int3* stack, unsigned int* top);
__global__ void path_mutate_kernel_p2(struct c_instance inst,
		                      int3* stack, unsigned int* top);
__global__ void calc_res(struct c_instance inst, float* ind, float* dest);
__global__ void copy_to_tmp_kernel(struct c_instance inst, int lucky);
__global__ void calc_tmp_res(struct c_instance inst);

//__global__ void all_in_one_kernel(struct c_instance inst, const int lucky);

#endif /* C_RATING_H_ */
