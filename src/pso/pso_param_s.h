/*
 * pso_param_s.h
 *
 *  Created on: Jan 27, 2012
 *      Author: tkalbitz
 */

#ifndef PSO_PARAM_S_H_
#define PSO_PARAM_S_H_

#include <pso_instance.h>

struct param_s {
	int s;
	int s_count;
	int s_set_len;
	int* s_set;
	double* old_rat;
};

void param_s_init(struct pso_instance& inst, struct param_s& ps);
void param_s_destroy(struct param_s& ps);
void param_s_update(struct pso_instance& inst, struct param_s& ps);


#endif /* PSO_PARAM_S_H_ */
