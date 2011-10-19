#include "pso_print.h"
#include "pso_copy.h"
#include "ya_malloc.h"

static void print_global_matrix_line_pretty(FILE* f, struct pso_instance *inst,
					    double* global_cpy,
					    int idx, int m)
{
	int count = idx * inst->width_per_inst + (m + 1) * inst->dim.matrix_width - 1;
	int w = idx * inst->width_per_inst + m * inst->dim.matrix_width;

	fprintf(f, "[ ");

	for (; w < count; w++) {
		fprintf(f, "%10.9e, ", global_cpy[w]);
	}

	fprintf(f, "%10.9e ]", global_cpy[w]);
}

void print_global_matrix_pretty(FILE* f, struct pso_instance* inst, int block)
{
	int width = inst->dim.blocks *
		    inst->width_per_inst *
		    inst->dim.matrix_height *
		    sizeof(double);

	double* global_cpy = (double*)ya_malloc(width);
	memset(global_cpy, 1, width);

	copy_globals_dev_to_host(inst, global_cpy);

	int line = inst->width_per_inst;
	int block_offset = line * inst->dim.matrix_height;
	double* block_ptr = global_cpy + block_offset * block;

	for(int m = 0; m < inst->num_matrices; m++) {
		char matrix = 'A' + m;
		fprintf(f, "%c: matrix(\n", matrix);
		for (int h = 0; h < inst->dim.matrix_height; h++) {
			print_global_matrix_line_pretty(f, inst,
							block_ptr + h*line,
							0, m);

			if(h < (inst->dim.matrix_height - 1))
				fprintf(f, ",");
			fprintf(f, "\n");
		}
		fprintf(f, ");\n%c: factor(%c);\n\n", matrix, matrix);
	}

	fprintf(f, "\n");
	free(global_cpy);
}

void print_particle_matrix_pretty(FILE* f, struct pso_instance* inst,
				 int block, int particle)
{
	int width = inst->dim.blocks *
		    inst->width_per_inst * inst->dim.particles *
		    inst->dim.matrix_height *
		    sizeof(double);

	double* particle_cpy = (double*)ya_malloc(width);
	memset(particle_cpy, 1, width);

	copy_particles_dev_to_host(inst, particle_cpy);

	int line = inst->width_per_inst * inst->dim.particles;
	int block_offset = line * inst->dim.matrix_height;
	double* block_ptr = particle_cpy + block_offset * block;

	for(int m = 0; m < inst->num_matrices; m++) {
		char matrix = 'A' + m;
		fprintf(f, "%c: matrix(\n", matrix);
		for (int h = 0; h < inst->dim.matrix_height; h++) {
			print_global_matrix_line_pretty(f, inst,
							block_ptr + h*line,
							particle,
							m);

			if(h < (inst->dim.matrix_height - 1))
				fprintf(f, ",");
			fprintf(f, "\n");
		}
		fprintf(f, ");\n%c: factor(%c);\n\n", matrix, matrix);
	}

	fprintf(f, "\n");
	free(particle_cpy);
}

void print_lbest_particle_matrix_pretty(FILE* f, struct pso_instance* inst,
				        int block, int particle)
{
	int width = inst->dim.blocks *
		    inst->width_per_inst * inst->dim.particles *
		    inst->dim.matrix_height *
		    sizeof(double);

	double* particle_cpy = (double*)ya_malloc(width);
	memset(particle_cpy, 1, width);

	copy_lbest_particles_dev_to_host(inst, particle_cpy);

	int line = inst->width_per_inst * inst->dim.particles;
	int block_offset = line * inst->dim.matrix_height;
	double* block_ptr = particle_cpy + block_offset * block;

	for(int m = 0; m < inst->num_matrices; m++) {
		char matrix = 'A' + m;
		fprintf(f, "%c: matrix(\n", matrix);
		for (int h = 0; h < inst->dim.matrix_height; h++) {
			print_global_matrix_line_pretty(f, inst,
							block_ptr + h*line,
							particle,
							m);

			if(h < (inst->dim.matrix_height - 1))
				fprintf(f, ",");
			fprintf(f, "\n");
		}
		fprintf(f, ");\n%c: factor(%c);\n\n", matrix, matrix);
	}

	fprintf(f, "\n");
	free(particle_cpy);
}

void print_particle_ratings(struct pso_instance *inst)
{
	int width = inst->dim.particles * inst->dim.blocks;
	double *rating = (double*)ya_malloc(width * sizeof(double));
	memset(rating, 1, width * sizeof(double));
	copy_particle_rating_dev_to_host(inst, rating);
	printf("-------------------RATINGS-------------------------------\n");
	int i = 0;
	for (int b = 0; b < inst->dim.blocks; b++) {
		printf("block %3d: ", b);

		for (; i < inst->dim.particles + b * inst->dim.blocks; i++) {
			printf("%3.2e ", rating[i]);
		}
		printf("\n");
	}
	free(rating);
}

void print_gbest_particle_ratings(struct pso_instance *inst)
{
	int width = inst->dim.blocks;
	double *rating = (double*)ya_malloc(width * sizeof(double));
	memset(rating, 1, width * sizeof(double));

	CUDA_CALL(cudaMemcpy(rating, inst->gb_rat,
			width * sizeof(double), cudaMemcpyDeviceToHost));

	printf("------------------- GLOBAL RATINGS-------------------------\n");
	for (int i = 0; i < width; i++) {
		printf("%3.2e ", rating[i]);
	}
	printf("\n");
	free(rating);
}

void print_rules(FILE* f, struct pso_instance *inst)
{
	bool mul_sep_count = false;
	bool old_mul_sep_count = true;

	for(int i = 1; i < inst->rules_len; i++) {

		if(old_mul_sep_count != mul_sep_count) {
			if(mul_sep_count == false)
				fprintf(f, "ratsimp(factor(");
			old_mul_sep_count = mul_sep_count;
		}

		if(inst->rules[i] == MUL_SEP) {
			if(mul_sep_count == false)
				fprintf(f, "ident(%d)-", inst->dim.matrix_height);
			else
				fprintf(f, "ident(%d)));\n\n", inst->dim.matrix_height);

			mul_sep_count = !mul_sep_count;
		} else {
				fprintf(f, "%c.", 'A' + inst->rules[i]);
		}
	}

	fprintf(f, "\n");
}
