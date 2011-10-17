/*
 * pso.h
 *
 *  Created on: Oct 17, 2011
 *      Author: tkalbitz
 */

#ifndef PSO_H_
#define PSO_H_

#include "pso_instance.h"

__global__ void pso_evaluation_gbest(struct pso_instance* inst);
__global__ void pso_evaluation_lbest(struct pso_instance* inst);
__global__ void pso_swarm_step(struct pso_instance* inst);

#endif /* PSO_H_ */
