/*
 * matrix_print.h
 *
 *  Created on: Apr 19, 2011
 *      Author: tkalbitz
 */

#ifndef MATRIX_PRINT_H_
#define MATRIX_PRINT_H_

#include "instance.h"

void print_parent_matrix(struct instance* inst);
void print_parent_matrix(struct instance* inst, int block, int parent);
void print_parent_ratings(struct instance *inst);

#endif /* MATRIX_PRINT_H_ */
