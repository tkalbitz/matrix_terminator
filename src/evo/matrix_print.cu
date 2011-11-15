#include "matrix_print.h"
#include "matrix_copy.h"

#include "config.h"
#include "ya_malloc.h"


void print_rules(FILE* f, struct instance *inst)
{
	bool mul_sep_count = false;
	bool old_mul_sep_count = true;

	printf("ratprint: false;\n");
	for(int i = 1; i < inst->rules_len; i++) {

		if(old_mul_sep_count != mul_sep_count) {
			if(mul_sep_count == false)
				fprintf(f, "ratsimp(factor(");
			old_mul_sep_count = mul_sep_count;
		}

		if(inst->rules[i] == MUL_SEP || inst->rules[i] == MUL_MARK) {
			if(mul_sep_count == false)
				fprintf(f, "ident(%d)-", inst->dim.matrix_height);
			else
				fprintf(f, "ident(%d)));\n", inst->dim.matrix_height);

			mul_sep_count = !mul_sep_count;
		} else {
				fprintf(f, "%c.", 'A' + inst->rules[i]);
		}
	}

	fprintf(f, "\n");

	for(int i = 1; i < inst->rules_len; i++) {

		if(old_mul_sep_count != mul_sep_count) {
			if(mul_sep_count == false)
				fprintf(f, "ratsimp(factor(");
			old_mul_sep_count = mul_sep_count;
		}

		if(inst->rules[i] == MUL_SEP || inst->rules[i] == MUL_MARK) {
			if(mul_sep_count == false)
				fprintf(f, "ident(%d)));\nratsimp(factor(", inst->dim.matrix_height);
			else
				fprintf(f, "ident(%d)));\n\n", inst->dim.matrix_height);

			mul_sep_count = !mul_sep_count;
		} else {
				fprintf(f, "%c.", 'A' + inst->rules[i]);
		}
	}
	fprintf(f, "\n");
}

static void print_parent_matrix_line_pretty(FILE* f, struct instance *inst,
					    double* parent_cpy,
					    int parent, int m)
{
	int w = parent * inst->width_per_inst + m * inst->dim.matrix_width;
	int count = parent * inst->width_per_inst + (m + 1) * inst->dim.matrix_width - 1;

	fprintf(f, "[ ");

	for (; w < count; w++) {
		fprintf(f, "%10.9e, ", parent_cpy[w]);
	}

	fprintf(f, "%10.9e ]", parent_cpy[w]);
}

static void print_debug_line_pretty(FILE* f, struct instance *inst,
				    double* debug_cpy,
				    int child, int rule, int m)
{
#ifdef DEBUG
	int w = child * 2 * inst->dim.matrix_width * inst->rules_count +
			2 * inst->dim.matrix_width * rule +
			m * inst->dim.matrix_width;
	int count = child * 2 * inst->dim.matrix_width * inst->rules_count +
			    2 * inst->dim.matrix_width * rule +
			    (m + 1) * inst->dim.matrix_width - 1;

	fprintf(f, "[ ");

	for (; w < count; w++) {
		fprintf(f, "%10.9e, ", debug_cpy[w]);
	}

	fprintf(f, "%10.9e ]", debug_cpy[w]);
#endif
}

void print_debug_pretty(FILE* f, struct instance* inst,
				int block, int child)
{
#ifdef DEBUG
	const int width = inst->dev_debug_ext.width * inst->dim.matrix_height *
			  inst->dim.blocks;

	double* debug_cpy = (double*)ya_malloc(width);
	memset(debug_cpy, 1, width);

	copy_debug_dev_to_host(inst, debug_cpy);

	int line = inst->dev_debug_ext.width / sizeof(*debug_cpy);
	int block_offset = line * inst->dim.matrix_height;
	double* block_ptr = debug_cpy + block_offset * block;

	for(int rule = 0; rule < inst->rules_count; rule++) {
		for(int m = 0; m < 2; m++) {
			char matrix = 'L' + m;
			fprintf(f, "Rule Side %c: matrix(\n", matrix);
			for (int h = 0; h < inst->dim.matrix_height; h++) {
				print_debug_line_pretty(f, inst,
							block_ptr + h*line,
							child, rule, m);

				if(h < (inst->dim.matrix_height - 1))
					fprintf(f, ",");
				fprintf(f, "\n");
			}
			fprintf(f, ");\n%c: factor(%c);\n\n", matrix, matrix);
		}
	}

	fprintf(f, "\n");
	free(debug_cpy);
#endif
}

void print_parent_matrix_pretty(FILE* f, struct instance* inst,
				int block, int parent)
{
	int width = inst->dim.parents    * /* there are n parents per block */
		    inst->width_per_inst *
		    sizeof(double) *
		    inst->dim.matrix_height * inst->dim.blocks;

	double* parent_cpy = (double*)ya_malloc(width);
	memset(parent_cpy, 1, width);

	copy_parents_dev_to_host(inst, parent_cpy);

	int line = inst->dim.parents *  inst->width_per_inst;
	int block_offset = line * inst->dim.matrix_height;
	double* block_ptr = parent_cpy + block_offset * block;

	for(int m = 0; m < inst->num_matrices; m++) {
		char matrix = 'A' + m;
		fprintf(f, "%c: matrix(\n", matrix);
		for (int h = 0; h < inst->dim.matrix_height; h++) {
			print_parent_matrix_line_pretty(f, inst,
							block_ptr + h*line,
							parent, m);

			if(h < (inst->dim.matrix_height - 1))
				fprintf(f, ",");
			fprintf(f, "\n");
		}
		fprintf(f, ");\n%c: factor(%c);\n\n", matrix, matrix);
	}

	fprintf(f, "\n");
	free(parent_cpy);
}

void print_child_matrix_pretty(FILE* f, struct instance* inst,
				int block, int child)
{
	int width = inst->dim.parents * inst->dim.childs *
		    inst->width_per_inst *
		    sizeof(double) *
		    inst->dim.matrix_height * inst->dim.blocks;

	double* child_cpy = (double*)ya_malloc(width);
	memset(child_cpy, 1, width);

	copy_childs_dev_to_host(inst, child_cpy);

	int line = inst->dim.parents * inst->dim.childs * inst->width_per_inst;
	int block_offset = line * inst->dim.matrix_height;
	double* block_ptr = child_cpy + block_offset * block;

	for(int m = 0; m < inst->num_matrices; m++) {
		char matrix = 'A' + m;
		fprintf(f, "%c: matrix(\n", matrix);
		for (int h = 0; h < inst->dim.matrix_height; h++) {
			print_parent_matrix_line_pretty(f, inst,
							block_ptr + h*line,
							child, m);

			if(h < (inst->dim.matrix_height - 1))
				fprintf(f, ",");
			fprintf(f, "\n");
		}
		fprintf(f, ");\n%c: factor(%c);\n\n", matrix, matrix);
	}

	fprintf(f, "\n");
	free(child_cpy);
}

static void print_result_matrix_line_pretty(struct instance *inst,
					    double* result_cpy,
					    int child, int m)
{
	int w =     child * inst->width_per_inst + m * inst->dim.matrix_width;
	int count = child * inst->width_per_inst + (m + 1) * inst->dim.matrix_width - 1;

	printf("[ ");

	for (; w < count; w++) {
		printf("%10.9e, ", result_cpy[w]);
	}

	printf("%10.9e ]", result_cpy[w]);
}

void print_result_matrix_pretty(struct instance* inst, int block, int child)
{
	int width = inst->dim.childs * inst->dim.parents *
		    inst->width_per_inst * sizeof(double) *
		    inst->dim.matrix_height * inst->dim.blocks;

	double* const result_cpy = (double*)ya_malloc(width);
	memset(result_cpy, 1, width);

	copy_results_dev_to_host(inst, result_cpy);

	int line = inst->dim.childs * inst->dim.parents * inst->width_per_inst;
	int block_offset = line * inst->dim.matrix_height;
	double* block_ptr = result_cpy + block_offset * block;

	for(int m = 0; m < inst->num_matrices; m++) {
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

	free(result_cpy);
}

void print_parent_ratings(struct instance *inst)
{
	int width = inst->dim.parents * inst->dim.blocks;
	double *rating = (double*)ya_malloc(width * sizeof(double));
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
	int width = inst->dim.parents * inst->dim.childs * 3 * inst->dim.blocks;
	double *sparam = (double*)ya_malloc(width * sizeof(double));
	memset(sparam, 1, width * sizeof(double));
	copy_sparam_dev_to_host(inst, sparam);
	printf("-------------------s param-------------------------------\n");
	int i = 0;
	for (int b = 0; b < inst->dim.blocks; b++) {
		printf("block %3d: ", b);

		for (; i < 3 * (b + 1) * (inst->dim.parents * inst->dim.childs); i++) {
			printf("%3.2e ", sparam[i]);
		}
		printf("\n");
	}
	free(sparam);
}

void print_sparam_best(struct instance *inst)
{
	int width = inst->dim.parents * inst->dim.childs * 3 * inst->dim.blocks;
	double *sparam = (double*)ya_malloc(width * sizeof(double));
	memset(sparam, 1, width * sizeof(double));
	copy_sparam_dev_to_host(inst, sparam);
	int i = 0;
	for (int b = 0; b < inst->dim.blocks; b++) {
		printf("%3.2e %3.2e | ", sparam[i], sparam[i + 1]);
		i += 3 * (inst->dim.parents * inst->dim.childs);
	}
	printf("\n");
	free(sparam);
}
