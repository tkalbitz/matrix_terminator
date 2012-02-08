#ifndef C_RATING_H_
#define C_RATING_H_

#include "c_instance.h"

__global__ void setup_rating(struct c_instance inst, const int yoff);
__global__ void copy_parent_kernel(struct c_instance inst);
__global__ void mutate_kernel(struct c_instance inst);
__global__ void rate_mutated_kernel(struct c_instance inst);
#endif /* C_RATING_H_ */
