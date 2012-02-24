/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef PSO_PRINT_H_
#define PSO_PRINT_H_

#include "pso_instance.h"

void print_global_matrix_pretty(FILE* f, struct pso_instance* inst, int block);
void print_particle_ratings(struct pso_instance *inst);
void print_particle_matrix_pretty(FILE* f, struct pso_instance* inst,
				 int block, int particle);
void print_lbest_particle_matrix_pretty(FILE* f, struct pso_instance* inst,
				        int block, int particle);
void print_gbest_particle_ratings(struct pso_instance *inst);
void print_rules(FILE* f, struct pso_instance *inst);

#endif /* PSO_PRINT_H_ */
