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
__global__ void calc_res(struct c_instance inst, double* ind, double* dest);
__global__ void copy_to_tmp_kernel(struct c_instance inst, int lucky);
__global__ void calc_tmp_res(struct c_instance inst);

#endif /* C_RATING_H_ */
