#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include <cuda.h> 
#include <curand_kernel.h>

#include "instance.h"

#include "evo.h"
#include "evo_rating.h"
#include "evo_setup.h"

#include "matrix_print.h"
#include "matrix_copy.h"

/*
 * Allocate memory for the parent matrices. the memory is layouted for faster
 * access. The block count is the depth of the allocated memory. All threads of
 * one block had to operate on a part of the width.
 */
void alloc_parent_matrix(struct instance *inst)
{
	assert(inst->num_matrices != 0);

	int width = inst->dim.parents    * /* there are n parents per block */
		    inst->width_per_inst *
		    sizeof(double);

	inst->dev_parent_ext = make_cudaExtent(width,
					       inst->dim.matrix_height,
					       inst->dim.blocks);

	cudaPitchedPtr pitched_ptr;
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_parent_ext));

	inst->dev_parent = pitched_ptr;
}

/*
 * Allocate memory for the child matrices. the memory is layouted for faster
 * access. The bloc count is the depth of the allocated memory. All threads of
 * one block had to operate on a part of the width.
 */
void alloc_child_matrix(struct instance *inst)
{
	assert(inst->num_matrices != 0);

	int width = inst->dim.parents * inst->dim.childs * /* each parent should have n childs */
		    inst->width_per_inst * sizeof(double);

	inst->dev_child_ext = make_cudaExtent(width,
					      inst->dim.matrix_height,
					      inst->dim.blocks);

	cudaPitchedPtr pitched_ptr;
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_child_ext));
	inst->dev_child = pitched_ptr;
}

/**
 * Allocate the matrix for each thread which is 
 * needed for the multiplication and evaluation.
 */
void alloc_result_matrix(struct instance *inst)
{
	inst->dev_res_ext = make_cudaExtent(inst->dim.childs * inst->dim.parents *
					    2 * inst->dim.matrix_width * sizeof(double),
					    inst->dim.matrix_height,
					    inst->dim.blocks);

	cudaPitchedPtr pitched_ptr;
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_res_ext));
	inst->dev_res = pitched_ptr;
}

void alloc_sparam(struct instance *inst)
{
	inst->dev_sparam_ext = make_cudaExtent(inst->dim.childs * inst->dim.parents * sizeof(double),
					       1,
					       inst->dim.blocks);

	cudaPitchedPtr pitched_ptr;
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_sparam_ext));
	inst->dev_sparam = pitched_ptr;
}
inline int get_evo_threads(struct instance *inst) {
	return inst->dim.parents * inst->dim.childs;
}

void alloc_rating(struct instance *inst)
{
	inst->dev_crat_ext = make_cudaExtent(2 * get_evo_threads(inst) * sizeof(double),
	 			    	     1,
	 			    	     inst->dim.blocks);

	inst->dev_prat_ext = make_cudaExtent(inst->dim.parents * sizeof(double),
					     1,
					     inst->dim.blocks);

	cudaPitchedPtr pitched_ptr;
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_crat_ext));
	CUDA_CALL(cudaMemset3D(pitched_ptr, 0, inst->dev_crat_ext));
	inst->dev_crat = pitched_ptr;

	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_prat_ext));
	CUDA_CALL(cudaMemset3D(pitched_ptr, 0, inst->dev_prat_ext));
	inst->dev_prat = pitched_ptr;
}

void init_rnd_generator(struct instance *inst, int seed)
{	
	curandState *rnd_states;
	const int count = max(get_evo_threads(inst), MATRIX_HEIGHT);

	CUDA_CALL(cudaMalloc((void **)&rnd_states, 
			     count * BLOCKS * sizeof(curandState)));
	setup_rnd_kernel<<<BLOCKS, count>>>(rnd_states, seed);
	CUDA_CALL(cudaGetLastError());
	cudaThreadSynchronize();
	inst->rnd_states = rnd_states;
}

void set_num_matrices(struct instance* inst)
{
	int m = INT_MIN;
	for(int i = 0; i < inst->rules_len; i++)
		m = max(m, inst->rules[i]);

	inst->num_matrices = m + 1; /* matrices are zero based */
	printf("num_matrices set to %d\n", inst->num_matrices);
}

void init_instance(struct instance* inst, char* rules)
{
	inst->rules_len  = strlen(rules);
	inst->rules = (int*)malloc(sizeof(int) * inst->rules_len);

	for(int i = 0; i < inst->rules_len; i++) {
		if(rules[i] > 96)
			inst->rules[i] = (rules[i] == 'X') ? MUL_SEP : rules[i] - 'a';
		else
			inst->rules[i] = (rules[i] == 'X') ? MUL_SEP : rules[i] - '0';
	}

	inst->delta = 1;
	inst->match = MATCH_ALL;
	inst->cond_left  = COND_UPPER_LEFT;
	inst->cond_right = COND_UPPER_LEFT;

	inst->dim.blocks  = BLOCKS;
	inst->dim.childs  = CHILDS;
	inst->dim.parents = PARENTS;
	inst->dim.matrix_width  = MATRIX_WIDTH;
	inst->dim.matrix_height = MATRIX_HEIGHT;
	
	inst->rounds = 0;
	inst->isnan = 0;

	inst->res_block = 0;
	inst->res_parent = 0;
	inst->res_child_block = 0;
	inst->res_child_idx = 0;

	set_num_matrices(inst);

	inst->width_per_inst = inst->num_matrices *    /* there are n matrices needed for the rules */
			       inst->dim.matrix_width; /* each one has a fixed width */

	alloc_parent_matrix(inst);
	alloc_child_matrix(inst);
	alloc_result_matrix(inst);
	alloc_rating(inst);
	alloc_sparam(inst);
	init_rnd_generator(inst, time(0));
}

void cleanup(struct instance *inst, struct instance * dev_inst) {
	free(inst->rules);

	cudaFree(dev_inst);
	/* dev_inst-> rules? */

	cudaFree(inst->rnd_states);
	cudaFree(inst->dev_child.ptr);
	cudaFree(inst->dev_parent.ptr);
	cudaFree(inst->dev_res.ptr);
	cudaFree(inst->dev_crat.ptr);
	cudaFree(inst->dev_prat.ptr);
	cudaFree(inst->dev_sparam.ptr);
}

struct instance* create_dev_inst(struct instance *inst)
{
	struct instance *dev_inst;
	int *rules = inst->rules;
	int *dev_rules;
	CUDA_CALL(cudaMalloc(&dev_rules, inst->rules_len * sizeof(int)));
	CUDA_CALL(cudaMemcpy(dev_rules, rules, inst->rules_len * sizeof(int),
					cudaMemcpyHostToDevice));
	inst->rules = dev_rules;
	CUDA_CALL(cudaMalloc(&dev_inst, sizeof(*dev_inst)));
	CUDA_CALL(cudaMemcpy(dev_inst, inst, sizeof(*dev_inst),
					cudaMemcpyHostToDevice));

	inst->rules = rules;
	return dev_inst;
}

void copy_inst_dev_to_host(struct instance *dev, struct instance *host)
{
	int *rules = host->rules;
	CUDA_CALL(cudaMemcpy(host, dev, sizeof(*dev), cudaMemcpyDeviceToHost));
	host->rules = rules;
}

int main(int argc, char** argv)
{
	if(argc < 2) {
		printf("Please supply a rule\n");
		exit(1);
	}

	/* there is no runtime limit for kernels */
	CUDA_CALL(cudaSetDevice(0));

	struct instance inst;
	struct instance *dev_inst;

	init_instance(&inst, argv[1]);
	dev_inst = create_dev_inst(&inst);

	printf("Rules: ");
	print_rules(&inst);

	setup_parent_kernel<<<BLOCKS, inst.dim.matrix_height>>>(dev_inst);
	cudaThreadSynchronize();
	CUDA_CALL(cudaGetLastError());

	int evo_threads = get_evo_threads(&inst);
	dim3 blocks(BLOCKS, PARENTS*CHILDS);
	dim3 threads(MATRIX_WIDTH, MATRIX_HEIGHT);

	setup_sparam<<<BLOCKS, evo_threads>>>(dev_inst);
	cudaThreadSynchronize();
	CUDA_CALL(cudaGetLastError());

	// Prepare
	cudaEvent_t start, stop;
	float elapsedTime;
	float elapsedTimeTotal = 0.f;

	int width = inst.dim.parents * inst.dim.blocks;
	double *rating = (double*)malloc(width * sizeof(double));
	int rounds = -1;

	int max_rounds = 500;
	int block = 0; int thread = 0;

	for(int i = 0; i < max_rounds; i++) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		// Start record
		cudaEventRecord(start, 0);

		evo_kernel<<<BLOCKS, evo_threads>>>(dev_inst, 0);
		CUDA_CALL(cudaGetLastError());
		cudaThreadSynchronize();
		CUDA_CALL(cudaGetLastError());

		evo_calc_res<<<blocks, threads>>>(dev_inst);
		CUDA_CALL(cudaGetLastError());
		cudaThreadSynchronize();
		CUDA_CALL(cudaGetLastError());

		evo_kernel<<<BLOCKS, evo_threads>>>(dev_inst, 1);
		CUDA_CALL(cudaGetLastError());
		cudaThreadSynchronize();
		CUDA_CALL(cudaGetLastError());

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
		elapsedTimeTotal += elapsedTime;
		// Clean up:
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		if(i & 1111 != 0 && i != (max_rounds - 1))
			continue;

//		print_parent_matrix_pretty(&inst, inst.res_block, inst.res_parent);
//		print_parent_ratings(&inst);
//		print_sparam(&inst);
		copy_parent_rating_dev_to_host(&inst, rating);
		for(int j = 0; j < width; j += PARENTS) {
			if(rating[j] == 0.) {
				block = j / PARENTS;
				thread = j % PARENTS;
				rounds = i;
				i = max_rounds;
				break;
			}
		}
	}

	free(rating);
	copy_inst_dev_to_host(dev_inst, &inst);

	print_sparam(&inst);
	print_parent_ratings(&inst);
	printf("Result:\n");
	print_result_matrix_pretty(&inst, block, thread);
	printf("Parents:\n");
	print_parent_matrix_pretty(&inst, block, thread);
	print_rules(&inst);
	printf("Time needed: %f\n", elapsedTimeTotal);
	printf("Needed rounds: %d\n", rounds);
	printf("Is NaN: %d\n", inst.isnan);
	printf("Result is block: %d, parent: %d\n", block, thread);
	printf("Result was in block: %d, child: %d\n", inst.res_child_block, inst.res_child_idx);

	printf("Clean up and exit.\n");
	cleanup(&inst, dev_inst);

	if(rounds == -1)
		return 0;

	return 1;
}
