#include "matrix_print.h"
#include "matrix_copy.h"

#include "config.h"

void print_rules(FILE* f, struct instance *inst)
{
	bool mul_sep_count = false;
	for(int i = 1; i < inst->rules_len; i++)
		if(inst->rules[i] == MUL_SEP) {
			if(mul_sep_count == 0)
				fprintf(f, "-");
			else
				fprintf(f, ";\n\n");

			mul_sep_count = !mul_sep_count;
		} else {
			if(inst->rules[i + 1] == MUL_SEP)
				fprintf(f, "%c", 'A' + inst->rules[i]);
			else
				fprintf(f, "%c.", 'A' + inst->rules[i]);
		}

	fprintf(f, "\n");
}

static void print_parent_matrix_line(struct instance *inst, double* parent_cpy)
{
	int count = inst->dev_parent_ext.width / sizeof(double);
	for (int w = 0; w < count; w++) {
		if((w % inst->dim.matrix_width) == 0) {
			printf(" | ");
		}

		printf("%3.2e ", parent_cpy[w]);
	}
	printf("\n");
}

static void print_parent_matrix_line_pretty(FILE* f, struct instance *inst,
					    double* parent_cpy,
					    int parent, int m)
{
	int count = parent * inst->width_per_inst + (m + 1) * MATRIX_WIDTH - 1;
	int w = parent * inst->width_per_inst + m * inst->dim.matrix_width;

	fprintf(f, "[ ");

	for (; w < count; w++) {
		fprintf(f, "%10.9e, ", parent_cpy[w]);
	}

	fprintf(f, "%10.9e ]", parent_cpy[w]);
}

void print_parent_matrix_pretty(FILE* f, struct instance* inst,
				int block, int parent)
{
	int width = inst->dim.parents    * /* there are n parents per block */
		    inst->width_per_inst *
		    sizeof(double) *
		    inst->dim.matrix_height * inst->dim.blocks;

	double* parent_cpy = (double*)malloc(width);
	memset(parent_cpy, 1, width);

	copy_parents_dev_to_host(inst, parent_cpy);

	int line = inst->dim.parents *  inst->width_per_inst;
	int block_offset = line * inst->dim.matrix_height;
	double* block_ptr = parent_cpy + block_offset * block;

	for(int m = 0; m < inst->num_matrices; m++) {
		fprintf(f, "%c: matrix(\n", 'A'+m);
		for (int h = 0; h < inst->dim.matrix_height; h++) {
			print_parent_matrix_line_pretty(f, inst,
							block_ptr + h*line,
							parent, m);

			if(h < (inst->dim.matrix_height - 1))
				fprintf(f, ",");
			fprintf(f, "\n");
		}
		fprintf(f, ");\n\n");
	}

	fprintf(f, "\n");
	free(parent_cpy);
}

static void print_result_matrix_line_pretty(struct instance *inst,
					    double* result_cpy,
					    int child, int m)
{
	int w =     child * 2 * MATRIX_WIDTH + m * MATRIX_WIDTH;
	int count = child * 2 * MATRIX_WIDTH + (m + 1) * MATRIX_WIDTH - 1;

	printf("[ ");

	for (; w < count; w++) {
		printf("%10.9e, ", result_cpy[w]);
	}

	printf("%10.9e ]", result_cpy[w]);
}

void print_result_matrix_pretty(struct instance* inst, int block, int child)
{
	int width = inst->dim.childs * inst->dim.parents *
		    2 * inst->dim.matrix_width * sizeof(double) *
		    inst->dim.matrix_height * inst->dim.blocks;

	double* result_cpy = (double*)malloc(width);
	memset(result_cpy, 1, width);

	copy_results_dev_to_host(inst, result_cpy);

	int line = inst->dim.childs * inst->dim.parents *
		   2 * inst->dim.matrix_width;
	int block_offset = line * inst->dim.matrix_height;
	double* block_ptr = result_cpy + block_offset * block;

	for(int m = 0; m < 2; m++) {
		printf("%c: matrix(\n", 'A'+m);
		for (int h = 0; h < inst->dim.matrix_height; h++) {
			print_result_matrix_line_pretty(inst,
							block_ptr + h*line,
							child, m);

			if(h < (inst->dim.matrix_height - 1))
				printf(",");
			printf("\n");
		}
		printf(");\n\n");
	}

	printf("--------------------------------------------------\n");
	free(result_cpy);
}

void print_parent_ratings(struct instance *inst)
{
	int width = inst->dim.parents * inst->dim.blocks;
	double *rating = (double*)malloc(width * sizeof(double));
	memset(rating, 1, width * sizeof(double));
	copy_parent_rating_dev_to_host(inst, rating);
	printf("-------------------RATINGS-------------------------------\n");
	int i = 0;
	for (int b = 0; b < inst->dim.blocks; b++) {
		printf("block %3d: ", b);

		for (; i < (b + 1) * (inst->dim.parents); i++) {
			printf("%3.2e ", rating[i]);
		}
		printf("\n");
	}
	free(rating);
}

void print_sparam(struct instance *inst)
{
	int width = inst->dim.parents * inst->dim.childs * inst->dim.blocks;
	double *sparam = (double*)malloc(width * sizeof(double));
	memset(sparam, 1, width * sizeof(double));
	copy_sparam_dev_to_host(inst, sparam);
	printf("-------------------s param-------------------------------\n");
	int i = 0;
	for (int b = 0; b < inst->dim.blocks; b++) {
		printf("block %3d: ", b);

		for (; i < (b + 1) * (inst->dim.parents * inst->dim.childs); i++) {
			printf("%3.2e ", sparam[i]);
		}
		printf("\n");
	}
	free(sparam);
}

void print_parent_matrix(struct instance* inst)
{
	int width = inst->dim.parents    * /* there are n parents per block */
		    inst->width_per_inst *
		    sizeof(double);

	const size_t count = width * inst->dim.matrix_height * inst->dim.blocks;
	double* parent_cpy = (double*)malloc(count);
	memset(parent_cpy, 1, count);

	copy_parents_dev_to_host(inst, parent_cpy);

	int line = inst->dim.parents *  inst->width_per_inst;
	int block_offset = inst->width_per_inst;

	for (int h = 0; h < inst->dim.matrix_height; h++) {
		for (int b = 0; b < inst->dim.blocks; b++) {
			double* line_ptr = parent_cpy + (h * line + b * block_offset);
			print_parent_matrix_line(inst, line_ptr);
		}
		printf("--------------------------------------------------\n");
	}
	free(parent_cpy);
	//print_parent_ratings(inst);
}
