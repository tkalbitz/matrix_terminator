/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef MATRIX_PRINT_H_
#define MATRIX_PRINT_H_

#include "instance.h"

void print_parent_matrix_pretty(FILE* f, struct instance* inst,
				int block, int parent);
void print_result_matrix_pretty(struct instance* inst, int block, int child);

void print_parent_ratings(struct instance *inst);
void print_rules(FILE* f, struct instance *inst);
void print_sparam(struct instance *inst);
void print_sparam_best(struct instance *inst);
void print_debug_pretty(FILE* f, struct instance* inst, int block, int child);
void print_child_matrix_pretty(FILE* f, struct instance* inst,
				int block, int child);


#endif /* MATRIX_PRINT_H_ */
