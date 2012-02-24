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
#include "pso_memory.h"
#include "pso_config.h"

__global__ void pso_swarm_step_ccpso2(const struct pso_instance inst, const int s);
__global__ void pso_evaluation_lbest(const struct pso_instance inst,
		                     const int s, const int cur);
__global__ void pso_neighbor_best(const struct pso_instance inst, const int s);

#endif /* PSO_H_ */
