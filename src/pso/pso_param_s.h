/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
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
