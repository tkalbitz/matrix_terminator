#ifndef PSO_RATING_H_
#define PSO_RATING_H_

#include "pso_instance.h"

__global__ void pso_calc_res(const struct pso_instance inst,
		             const int s, const int cur);

#endif /* PSO_RATING_H_ */