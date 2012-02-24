/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef MATRIX_COPY_H_
#define MATRIX_COPY_H_

#include "instance.h"

void copy_parents_dev_to_host(struct instance* inst, void* parent_cpy);
void copy_childs_dev_to_host(struct instance* inst, void* parent_cpy);
void copy_parent_rating_dev_to_host(struct instance* inst, void* parent_rat_cpy);
void copy_results_dev_to_host(struct instance* inst, void* result_cpy);
void copy_sparam_dev_to_host(struct instance* inst, void* sparam_cpy);
void copy_debug_dev_to_host(struct instance* inst, void* debug_cpy);
void copy_child_rating_dev_to_host(struct instance* inst, void* child_rat_cpy);

#endif /* MATRIX_COPY_H_ */
