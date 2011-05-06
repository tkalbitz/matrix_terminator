#ifndef __EVO_H__
#define __EVO_H__

#include "instance.h"

__global__ void setup_rnd_kernel(curandState* state, int seed);
__global__ void evo_kernel(struct instance *inst);
__global__ void evo_kernel_test(struct instance *inst);
__global__ void setup_parent_kernel(struct instance *inst);

#endif
