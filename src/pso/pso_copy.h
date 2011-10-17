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
