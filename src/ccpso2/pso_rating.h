/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef PSO_RATING_H_
#define PSO_RATING_H_

#include "pso_instance.h"

__global__ void pso_calc_res(const struct pso_instance inst,
		             const int s, const int cur);

#endif /* PSO_RATING_H_ */
