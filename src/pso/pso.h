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

__global__ void pso_swarm_step_ccpso2(struct pso_instance inst, const int s);
__global__ void pso_evaluation_lbest(struct pso_instance inst, const int s);
__global__ void pso_neighbor_best(struct pso_instance inst, const int s);

#endif /* PSO_H_ */
