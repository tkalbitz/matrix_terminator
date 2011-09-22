/*
 * matrix_generator.h
 *
 *  Created on: Sep 22, 2011
 *      Author: tkalbitz
 */

#ifndef MATRIX_GENERATOR_H_
#define MATRIX_GENERATOR_H_

#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>

#define E_NO_FREE_INST       -1;
#define E_INVALID_INST       -2;
#define E_RULES_FORMAT_WRONG -3;

#define MUL_SEP -1

void evo_init();
int evo_init_instance(const int         matrix_width,
		      const int * const rules,
		      const size_t      rules_len);

int evo_run(const int     instance,
	    const int     cycles,
	    double* const result);

int evo_destroy_instance(int instance);





#endif /* MATRIX_GENERATOR_H_ */
