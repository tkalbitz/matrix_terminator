#include "c_print.h"
#include "ya_malloc.h"

void print_matrix_pretty(FILE* f, struct c_instance& inst, int block, int bpos)
{
	int width = inst.itotal * sizeof(double);
	double* global_cpy = (double*)ya_malloc(width);
	memset(global_cpy, 1, width);

	CUDA_CALL(cudaMemcpy(global_cpy, inst.instances, width,
			cudaMemcpyDeviceToHost));

	int block_offset = inst.width_per_inst * inst.icount * block;
	double* ptr = global_cpy + block_offset + bpos * inst.width_per_inst;

//	printf("{");
//	for(int i = 0; i < inst.width_per_inst; i++)
//		printf("%.2e, ", ptr[i]);
//
//	printf("}\n");

	for(int m = 0; m < inst.num_matrices; m++) {
		char matrix = 'A' + m;
		fprintf(f, "%c: matrix(\n", matrix);

		for (int h = 0; h < inst.mdim; h++) {
			int pos = m * inst.width_per_matrix +
				  h * inst.mdim;
			fprintf(f, "[ ");

			for (int w = 0; w < inst.mdim - 1; w++) {
				fprintf(f, "%10.9e, ", ptr[pos + w]);
			}

			fprintf(f, "%10.9e ]", ptr[pos + inst.mdim - 1]);

			if(h < (inst.mdim - 1))
				fprintf(f, ",");
			fprintf(f, "\n");
		}
		fprintf(f, ");\n%c: factor(%c);\n\n", matrix, matrix);
	}

	fprintf(f, "\n");
	free(global_cpy);
}

void print_rules(FILE* f, struct c_instance& inst)
{
	bool mul_sep_count = false;
	bool old_mul_sep_count = true;

	for(uint32_t i = 1; i < inst.rules_len; i++) {

		if(old_mul_sep_count != mul_sep_count) {
			if(mul_sep_count == false)
				fprintf(f, "ratsimp(factor(");
			old_mul_sep_count = mul_sep_count;
		}

		if(inst.rules[i] == MUL_SEP) {
			if(mul_sep_count == false)
				fprintf(f, "ident(%d)-", inst.mdim);
			else
				fprintf(f, "ident(%d)));\n\n", inst.mdim);

			mul_sep_count = !mul_sep_count;
		} else {
				fprintf(f, "%c.", 'A' + inst.rules[i]);
		}
	}

	fprintf(f, "\n");
}
