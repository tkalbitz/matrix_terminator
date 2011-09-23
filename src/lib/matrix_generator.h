/*
 * matrix_generator.h
 *
 *  Created on: Sep 22, 2011
 *      Author: tkalbitz
 */

#ifndef MATRIX_GENERATOR_H_
#define MATRIX_GENERATOR_H_

#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <ctype.h>

#include "evo_error.h"

#define MUL_SEP -1

/**
 * Create all maintenance structures for the library and must be called before
 * any other library function was called.
 */
void evo_lib_init();

/**
 * Destroy created structures and gracefully shutdown CUDA. Must be called when
 * no further library function will be used.
 */
void evo_lib_destroy();

/**
 * Create a instance of the evo. Reasonable defaults are set which are
 * Termination Competition rules, max value of 10 and a delta of 0.1 .
 *
 * @param matrix_width (in) Size of the desired matrices
 * @param rules        (in) Rules that must be matched (will be copied)
 * @param rules        (in) Length of the rules array
 *
 * @return A number >= which is a valid instance. A number < 0 is an error.
 */
int evo_create_instance(const int         matrix_width,
		        const int * const rules,
		        const size_t      rules_len);

/**
 * Destroys a instance if it isn't needed anymore.
 *
 * @param instance (in) instance which should be destroyed
 *
 * @return != a error occurs
 */
int evo_destroy_instance(uint32_t instance);

/**
 * Runs the evo and try to match the given rules.
 *
 * @param instance (in)  Instance which will be processed must be created by
 *                       evo_create_instance
 * @param cyles    (in)  Max evo cycles
 * @param result   (out) Solution will be stored here. The array must have at
 * 			 least (Count of unique matrices in the rule) *
 *                       (matrix_width)^2 items
 *
 * @return < 0       a error occurs
 *         < INT_MAX solution found in this cycle
 *         = INT_MAX no solution found
 */
int evo_run(const uint32_t instance,
	    const int      cycles,
	    double* const  result);

/**
 * Set the maximum value a element in the matrix can have.
 *
 * @param instance (in) The instance for which the value should be set.
 * @param max      (in) The maximum value of the matrix element.
 *
 * @return != 0 a error occured
 */
int evo_set_matrix_max_value(const uint32_t instance, const double max);

/**
 * Set the delta a matrix element must a multiple of.
 *
 * @param instance (in) The instance for which the delta should be set.
 * @param delta    (in) The delta value.
 *
 * @return != 0 a error occured
 */
int evo_set_delta_value(const uint32_t instance, const double delta);

#endif /* MATRIX_GENERATOR_H_ */
