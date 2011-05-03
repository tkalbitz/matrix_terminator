#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include <cuda.h> 
#include <curand_kernel.h>

#include "config.h"
#include "instance.h"
#include "evo.h"

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
		    sizeof(float);

	inst->dev_parent_ext = make_cudaExtent(width,
					       inst->dim.matrix_height,
					       inst->dim.blocks);

	cudaPitchedPtr pitched_ptr;
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_parent_ext));
	CUDA_CALL(cudaMemset3D(pitched_ptr, 0, inst->dev_parent_ext));

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
		    inst->width_per_inst * sizeof(float);

	inst->dev_child_ext = make_cudaExtent(width,
					      inst->dim.matrix_height,
					      inst->dim.blocks);

	cudaPitchedPtr pitched_ptr;
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_child_ext));
	CUDA_CALL(cudaMemset3D(pitched_ptr, 0, inst->dev_child_ext));
	inst->dev_child = pitched_ptr;
}

/**
 * Allocate the matrix for each thread which is 
 * needed for the multiplication and evaluation.
 */
void alloc_result_matrix(struct instance *inst)
{
	inst->dev_res_ext = make_cudaExtent(inst->dim.childs * inst->dim.parents *
					    2 * inst->dim.matrix_width * sizeof(float),
					    inst->dim.matrix_height,
					    inst->dim.blocks);

	cudaPitchedPtr pitched_ptr;
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_res_ext));
	CUDA_CALL(cudaMemset3D(pitched_ptr, 0, inst->dev_res_ext));
	inst->dev_res = pitched_ptr;
}

inline int get_evo_threads(struct instance *inst) {
	return inst->dim.parents * inst->dim.childs;
}

void alloc_rating(struct instance *inst)
{
	inst->dev_crat_ext = make_cudaExtent(2 * get_evo_threads(inst) * sizeof(float),
	 			    	     1,
	 			    	     inst->dim.blocks);

	inst->dev_prat_ext = make_cudaExtent(inst->dim.parents * sizeof(float),
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
	const int count = get_evo_threads(inst);

	CUDA_CALL(cudaMalloc((void **)&rnd_states, 
			     count * BLOCKS * sizeof(curandState)));
	setup_rnd_kernel<<<BLOCKS, count>>>(rnd_states, seed);
	CUDA_CALL(cudaGetLastError());
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

void init_instance(struct instance* inst)
{
	inst->rule_count = 3;
	inst->rules_len  = 22;
	inst->rules = (int*)malloc(sizeof(int) * inst->rules_len);
	inst->rules[0] = MUL_SEP;
	inst->rules[1] = 1;
	inst->rules[2] = 1;
	inst->rules[3] = 1;
	inst->rules[4] = MUL_SEP;
	inst->rules[5] = 0;
	inst->rules[6] = MUL_SEP;

	inst->rules[7] = 0;
	inst->rules[8] = 0;
	inst->rules[9] = MUL_SEP;
	inst->rules[10] = 0;
	inst->rules[11] = 1;
	inst->rules[12] = 0;
	inst->rules[13] = MUL_SEP;

	inst->rules[14] = 0;
	inst->rules[15] = 0;
	inst->rules[16] = 0;
	inst->rules[17] = MUL_SEP;
	inst->rules[18] = 1;
	inst->rules[19] = 0;
	inst->rules[20] = 0;
	inst->rules[21] = MUL_SEP;

	inst->delta = 0.1;
	inst->match = MATCH_ALL;
	inst->cond_left  = COND_UPPER_RIGHT;
	inst->cond_right = COND_UPPER_RIGHT;

	inst->dim.blocks  = BLOCKS;
	inst->dim.childs  = CHILDS;
	inst->dim.parents = PARENTS;
	inst->dim.matrix_width  = MATRIX_WIDTH;
	inst->dim.matrix_height = MATRIX_HEIGHT;
	
	inst->rounds = 0;

	set_num_matrices(inst);

	inst->width_per_inst = inst->num_matrices *    /* there are n matrices needed for the rules */
			       inst->dim.matrix_width; /* each one has a fixed width */

	alloc_parent_matrix(inst);
	alloc_child_matrix(inst);
	alloc_result_matrix(inst);
	alloc_rating(inst);
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
}

struct instance* create_dev_inst(struct instance *inst)
{
	struct instance *dev_inst;
	int *rules = inst->rules;
	CUDA_CALL(cudaMalloc(&(inst->rules), inst->rules_len * sizeof(int)));
	CUDA_CALL(cudaMemcpy(inst->rules, rules, inst->rules_len * sizeof(int),
					cudaMemcpyHostToDevice));
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
	struct instance inst;
	struct instance *dev_inst;

	init_instance(&inst);
	dev_inst = create_dev_inst(&inst);

	setup_parent_kernel<<<BLOCKS, inst.dim.matrix_height>>>(dev_inst);
	CUDA_CALL(cudaGetLastError());
	//print_parent_matrix(&inst);

	int evo_threads = get_evo_threads(&inst);
	evo_kernel<<<BLOCKS, evo_threads>>>(dev_inst);
	CUDA_CALL(cudaGetLastError());

	copy_inst_dev_to_host(dev_inst, &inst);

	printf("Needed rounds: %d\n", inst.rounds);
	printf("Result is block: %d, parent: %d\n", inst.res_block, inst.res_parent);
	print_parent_matrix(&inst, inst.res_block, inst.res_parent);

	printf("Clean up and exit.\n");
	cleanup(&inst, dev_inst);
}
