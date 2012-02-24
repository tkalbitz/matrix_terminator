/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef EVO_SETUP_H_
#define EVO_SETUP_H_

#include "instance.h"

__global__ void setup_rnd_kernel(curandState* rnd_states, int seed);
__global__ void setup_parent_kernel(struct instance *inst);
__global__ void setup_childs_kernel(struct instance * const inst, bool half);
__global__ void setup_sparam(struct instance * const inst,
			     const double sparam,
			     const double mut_rate,
			     const double recomb_rate, bool half);




#endif /* EVO_SETUP_H_ */
