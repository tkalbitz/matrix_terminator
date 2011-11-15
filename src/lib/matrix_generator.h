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

#define MUL_SEP  -1
#define MUL_MARK -2


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
int evo_destroy_instance(const int instance);

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
int evo_run(const int instance,
	    const int      cycles,
	    double* const  result);

#define MATCH_ALL 0
#define MATCH_ANY 1

#define COND_UPPER_LEFT  0
#define COND_UPPER_RIGHT 1
#define COND_UPPER_LEFT_LOWER_RIGHT 2

/**
 * Set all evo parameter which will be used when evo_run is called for the
 * given instance. Please note that the defualt parameters will be set at every
 * evo_create_instance. If a parameter is < 0 it will be ignored. If 0 is
 * forbidden or the parameter is out of range E_INVALID_VALUE is returned. It is
 * garanteed that no parameter was changed when a error is returned.
 *
 * @param instance   (in) The instance for which the value should be set.
 * @param max        (in) The maximum value of the matrix element (must be > 0).
 * @param match      (in) The match value (must MATCH_ALL OR MATCH_ANY).
 * @param cond_left  (in) The constraint which all factor matrices have to
 *                        match. This has to be one of the values
 *                        COND_UPPER_LEFT, COND_UPPER_RIGHT or
 *                        COND_UPPER_LEFT_LOWER_RIGHT
 * @param cond_right (in) The constraint which all result matrices have to
 *                        match. This has to be one of the values
 *                        COND_UPPER_LEFT, COND_UPPER_RIGHT or
 *                        COND_UPPER_LEFT_LOWER_RIGHT
 * @param mut_rate   (in) The mutation rate which should be used at the start
 * 			  of the algorithm (must be > 0 and <= 1).
 * @param strgy_parm (in) Strategy parameter which is used in the algorithm. A
 *                        good start value is 2*delta (must be != 0).
 *
 * @return = 0 at success, at error < 0
 */
int evo_set_params(int instance,
		   double max, double delta, int match,
		   int cond_left, int cond_right,
		   double mut_rate, double strgy_parm);

/**
 * Set all default evo parameter which will be used when ever a new instance is
 * created. If a parameter is < 0 it will be ignored. If 0 is forbidden or the
 * parameter is out of range E_INVALID_VALUE is returned. It is garanteed that
 * no parameter was changed when a error is returned.
 *
 * @param instance   (in) The instance for which the value should be set.
 * @param max        (in) The maximum value of the matrix element (must be > 0).
 * @param match      (in) The match value (must MATCH_ALL OR MATCH_ANY).
 * @param cond_left  (in) The constraint which all factor matrices have to
 *                        match. This has to be one of the values
 *                        COND_UPPER_LEFT, COND_UPPER_RIGHT or
 *                        COND_UPPER_LEFT_LOWER_RIGHT
 * @param cond_right (in) The constraint which all result matrices have to
 *                        match. This has to be one of the values
 *                        COND_UPPER_LEFT, COND_UPPER_RIGHT or
 *                        COND_UPPER_LEFT_LOWER_RIGHT
 * @param mut_rate   (in) The mutation rate which should be used at the start
 * 			  of the algorithm (must be > 0 and <= 1).
 * @param strgy_parm (in) Strategy parameter which is used in the algorithm. A
 *                        good start value is 2*delta.
 *
 * @return = 0 at success, at error < 0
 */
int evo_set_def_params(double max, double delta, int match,
		       int cond_left, int cond_right,
		       double mut_rate, double strgy_parm);

#endif /* MATRIX_GENERATOR_H_ */
