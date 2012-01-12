#ifndef EVO_RATING_H_
#define PSO_RATING_H_

#include "pso_instance.h"

__global__ void pso_calc_res(struct pso_instance * const inst,
	                     const int s, const int cur);

#endif /* PSO_RATING_H_ */
