#ifndef __EVO_H__
#define __EVO_H__

#include "instance.h"

__global__ void evo_kernel_part_one(struct instance *inst);
__global__ void evo_kernel_part_two(struct instance *inst);

#endif
