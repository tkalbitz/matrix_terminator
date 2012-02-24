/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef C_PRINT_H_
#define C_PRINT_H_

#include "c_instance.h"

void print_matrix_pretty(FILE* f, struct c_instance& inst, int block, int pos);
void print_rules(FILE* f, struct c_instance& inst);

#endif /* C_PRINT_H_ */
