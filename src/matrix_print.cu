#include "matrix_print.h"
#include "matrix_copy.h"

#include "config.h"

static void print_parent_matrix_line(struct instance *inst, float* parent_cpy)
{
	int count = inst->dev_parent_ext.width / sizeof(float);
	for (int w = 0; w < count; w++) {
		if((w % inst->dim.matrix_width) == 0) {
			printf(" | ");
		}

		printf("%3.2e ", parent_cpy[w]);
	}
	printf("\n");
}

static void print_parent_matrix_line(struct instance *inst, float* parent_cpy, int parent)
{
	int count = parent * inst->width_per_inst + inst->width_per_inst;
	for (int w = parent * inst->width_per_inst; w < count; w++) {
		if((w % inst->dim.matrix_width) == 0) {
			printf(" | ");
		}

		printf("%3.2e ", parent_cpy[w]);
	}
	printf("\n");
}

void print_parent_matrix(struct instance* inst, int block, int parent)
{
	/* TODO: HACK */
	float parent_cpy[BLOCKS][MATRIX_HEIGHT][MATRIX_WIDTH * 2 * PARENTS];
	memset(parent_cpy, 1, BLOCKS * MATRIX_HEIGHT *
			      MATRIX_WIDTH * 2 * PARENTS);

	copy_parents_dev_to_host(inst, parent_cpy);

	for (int h = 0; h < inst->dim.matrix_height; h++) {
		print_parent_matrix_line(inst, parent_cpy[block][h], parent);
	}
	printf("--------------------------------------------------\n");

	print_parent_ratings(inst);
}

void print_parent_ratings(struct instance *inst)
{
	float rating[BLOCKS][1][inst->dim.parents];
	memset(rating, 1, BLOCKS * inst->dim.parents * sizeof(float));
	copy_parent_rating_dev_to_host(inst, rating);
	printf("-------------------RATINGS-------------------------------\n");
	for (int b = 0; b < inst->dim.blocks; b++) {
		for (int p = 0; p < inst->dim.parents; p++) {
			printf("%3.2e ", rating[b][0][p]);
		}
		printf("\n");
	}
}

void print_parent_matrix(struct instance* inst)
{
	/* TODO: HACK */
	float parent_cpy[BLOCKS][MATRIX_HEIGHT][MATRIX_WIDTH * 2 * PARENTS];
	memset(parent_cpy, 1, BLOCKS * MATRIX_HEIGHT *
			      MATRIX_WIDTH * 2 * PARENTS);
	copy_parents_dev_to_host(inst, parent_cpy);

	for (int b = 0; b < inst->dim.blocks; b++) {
		for (int h = 0; h < inst->dim.matrix_height; h++) {
			print_parent_matrix_line(inst, parent_cpy[b][h]);
		}
		printf("--------------------------------------------------\n");
	}
	print_parent_ratings(inst);
}
