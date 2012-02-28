/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#include <string.h>
#include <stdio.h>
#include <limits.h>

#include "../ya_malloc.h"
#include "matrix_generator.h"

#define MAT_WIDTH 5

static int* parse_rules(const char *crules);
static void print_solution(const float* const result);

int main(int argc, char **argv)
{
	/* rule where a solution should be found */
	char* crules = "YbbaYabbbXababXbaaX";
	int*  rules  = parse_rules(crules);

	/* the solution will be stored here */
	float * result = ya_malloc(2 /* matrizes in the rules */ *
				   MAT_WIDTH /* elements per matrix */ *
				   MAT_WIDTH /* rows */ * sizeof(*result));

	/* initialize the library and allocate instance slots */
	mat_lib_init();

	/*
	 * Create a instance with a desired matrix width and rule set. All
	 * configuration values are set to reasonable defaults.
	 */
	const int inst = c_create_instance(MAT_WIDTH, 100, rules, strlen(crules));
	if(inst < 0) {
		printf("Something bad happened at init: %d\n", inst);
		goto end;
	}

	printf("My instance is: %d\n", inst);

	/* set max value of matrix elements to 8 */
	c_set_params(inst, 8, -1 , -1, -1, -1, -1);

	/*
	 * try to find a solution for the given instance in 2000 rounds and
	 * store a solution in the given pointer.
	 */
	const int rounds = 20000;
	const int asteps = 1000;
	int ret  = c_run(inst, rounds, asteps, result);
	if(ret < 0) {
		printf("Something bad happened at run: %d\n", ret);
		goto end;
	}

	if(ret == INT_MAX) {
		printf("No solution found.\n");
		goto end;
	}

	print_solution(result);
end:
	free(result);
	free(rules);

	/* destroy the created instance and free the memory */
	c_destroy_instance(inst);

	/* destroy all create slots and management structures */
	mat_lib_destroy();
	return 0;
}

static int* parse_rules(const char *crules)
{
	int  rules_len  = strlen(crules);
	int* rules = (int*)ya_malloc(sizeof(int) * rules_len);

	for(int i = 0; i < rules_len; i++) {
		switch(crules[i]) {
		case 'X': {
			rules[i] = MUL_SEP;
			break;
		}
		case 'Y': {
			rules[i] = MUL_MARK;
			break;
		}
		default:
			if(crules[i] >= 'a')
				rules[i] = crules[i] - 'a';
			else
				rules[i] = crules[i] - '0';
			break;
		}
	}

	return rules;
}


static void print_solution(const float* const res) {
	for(int m = 0; m < 2; m++) {
		char matrix = 'A' + m;
		printf("%c: matrix(\n", matrix);

		for (int h = 0; h < MAT_WIDTH; h++) {
			int pos = m * MAT_WIDTH*MAT_WIDTH + h * MAT_WIDTH;
			printf("[ ");

			for (int w = 0; w < MAT_WIDTH - 1; w++) {
				printf("%10.9e, ", res[pos + w]);
			}

			printf("%10.9e ]", res[pos + MAT_WIDTH - 1]);

			if(h < (MAT_WIDTH - 1))
				printf(",");
			printf("\n");
		}
		printf(");\n%c: factor(%c);\n\n", matrix, matrix);
	}

	printf("\n");

}
