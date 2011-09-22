/*
 * lib_test.c
 *
 *  Created on: Sep 22, 2011
 *      Author: tkalbitz
 */
#include <string.h>
#include <stdio.h>

#include "ya_malloc.h"
#include "matrix_generator.h"

static int* parse_rules(const char *crules)
{
	int  rules_len  = strlen(crules);
	int* rules = (int*)ya_malloc(sizeof(int) * rules_len);

	for(int i = 0; i < rules_len; i++) {
		if(crules[i] >= 'a')
			rules[i] = (crules[i] == 'X') ? MUL_SEP : crules[i] - 'a';
		else
			rules[i] = (crules[i] == 'X') ? MUL_SEP : crules[i] - '0';
	}

	return rules;
}

#define MAT_WIDTH 5

void print_solution(const double* const result)
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

int main(int argc, char **argv)
{
	char* crules = "XaaaXbXbbXbabXbbbXabbX";
	int*  rules  = parse_rules(crules);
	double * result = ya_malloc(2 /* matrizes */ *
				    MAT_WIDTH /* elements per matrix */ *
				    MAT_WIDTH /* rows */ * sizeof(double));

	evo_init();

	const int inst = evo_init_instance(MAT_WIDTH, rules, strlen(crules));
	if(inst < 0) {
		printf("Something bad happened: %d\n", inst);
		goto end;
	}

	printf("My instance is: %d\n", inst);
	const int rounds = 2000;
	int ret  = evo_run(inst, rounds, result);
	if(ret < 0) {
		printf("Something bad happened: %d\n", ret);
		goto end;
	}

	if(ret > rounds) {
		printf("No solution found.\n");
		goto end;
	}

	print_solution(result);
end:
	free(result);
	free(rules);
	evo_destroy_instance(inst);
	return 0;
}



