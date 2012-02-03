/*
 * lib_test.c
 *
 *  Created on: Sep 22, 2011
 *      Author: tkalbitz
 */
#include <string.h>
#include <stdio.h>
#include <limits.h>

#include "../ya_malloc.h"
#include "matrix_generator.h"

#define MAT_WIDTH 5

static int* parse_rules(const char *crules);
static void print_solution(const double* const result);

int main(int argc, char **argv)
{
	/* rule where a solution should be found */
	char* crules = "YaaaYbXbbXbabXbbbXabbX";
	int*  rules  = parse_rules(crules);

	/* the solution will be stored here */
	double * result = ya_malloc(2 /* matrizes in the rules */ *
				    MAT_WIDTH /* elements per matrix */ *
				    MAT_WIDTH /* rows */ * sizeof(double));

	/* initialize the library and allocate instance slots */
	mat_lib_init();

	/*
	 * Create a instance with a desired matrix width and rule set. All
	 * configuration values are set to reasonable defaults.
	 */
	const int inst = evo_create_instance(MAT_WIDTH, rules, strlen(crules));
	if(inst < 0) {
		printf("Something bad happened at init: %d\n", inst);
		goto end;
	}

	printf("My instance is: %d\n", inst);

	/* set max value of matrix elements to 8 */
	evo_set_params(inst, 8, -1 ,-1, -1, -1, -1, -1);

	/*
	 * try to find a solution for the given instance in 2000 rounds and
	 * store a solution in the given pointer.
	 */
	const int rounds = 2000;
	int ret  = evo_run(inst, rounds, result);
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
	evo_destroy_instance(inst);

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

static void print_solution(const double* const result)
{
	printf("Solution:\n");
	for(int i = 0; i < 2 * MAT_WIDTH * MAT_WIDTH; i++) {
		if(i != 0 && i % MAT_WIDTH == 0)
			printf("| ");

		if(i != 0 && i % (2 * MAT_WIDTH) == 0)
			printf("\n");

		printf("%5.4e, ", result[i]);
	}
	printf("|\n");
}


