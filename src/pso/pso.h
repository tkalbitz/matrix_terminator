/*
 * pso.h
 *
 *  Created on: Oct 17, 2011
 *      Author: tkalbitz
 */

#ifndef PSO_H_
#define PSO_H_

#include "pso_instance.h"
#include "pso_memory.h"
#include "pso_config.h"

__global__ void pso_swarm_step_ccpso2(const struct pso_instance inst);
__global__ void pso_evaluation_lbest(const struct pso_instance inst, const int cur);
__global__ void pso_neighbor_best(const struct pso_instance inst);

#endif /* PSO_H_ */
