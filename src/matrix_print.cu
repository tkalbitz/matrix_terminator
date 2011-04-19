#include "matrix_print.h"
#include "matrix_copy.h"

#include "config.h"

static void print_parent_matrix_line(struct instance *inst, float* parent_cpy)
{
	for (int w = 0; w < inst->dim.threads * inst->width_per_inst; w++) {
		if((w % inst->dim.matrix_width) == 0) {
			printf(" | ");
		}

		printf("%3.2e ", parent_cpy[w]);
	}
	printf("\n");
}

void print_parent_matrix(struct instance* inst)
{
	float parent_cpy[BLOCKS][MATRIX_HEIGHT][MATRIX_WIDTH * 2 * THREADS];

	copy_parents_dev_to_host(inst, &parent_cpy);

	for (int b = 0; b < inst->dim.blocks; b++) {
		for (int h = 0; h < inst->dim.matrix_height; h++) {
			print_parent_matrix_line(inst, parent_cpy[b][h]);
		}
		printf("--------------------------------------------------\n");
	}
}
