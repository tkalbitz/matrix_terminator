/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef PSO_H_
#define PSO_H_

#include "pso_instance.h"

__global__ void pso_evaluation_gbest(struct pso_instance* inst);
__global__ void pso_evaluation_lbest(struct pso_instance* inst);
__global__ void pso_swarm_step(struct pso_instance* inst);
__global__ void pso_swarm_step_ccpso(struct pso_instance* inst);

#endif /* PSO_H_ */
