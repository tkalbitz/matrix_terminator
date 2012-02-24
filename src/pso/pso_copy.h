/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef PSO_COPY_H_
#define PSO_COPY_H_

#include "pso_instance.h"


void copy_gb_rating_dev_to_host(struct pso_instance* inst, void* parent_rat_cpy);
void copy_globals_dev_to_host(struct pso_instance* inst, void* global_cpy);
void copy_particle_rating_dev_to_host(struct pso_instance* inst,
				      void* particle_rat_cpy);
void copy_particles_dev_to_host(struct pso_instance* inst, void* particle_cpy);
void copy_lbest_particles_dev_to_host(struct pso_instance* inst, void* particle_cpy);

#endif /* PSO_COPY_H_ */
