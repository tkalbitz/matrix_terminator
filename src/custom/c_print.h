/*
 * c_print.h
 *
 *  Created on: Feb 9, 2012
 *      Author: tkalbitz
 */

#ifndef C_PRINT_H_
#define C_PRINT_H_

#include "c_instance.h"

void print_matrix_pretty(FILE* f, struct c_instance& inst, int block, int pos);
void print_rules(FILE* f, struct c_instance& inst);

#endif /* C_PRINT_H_ */
