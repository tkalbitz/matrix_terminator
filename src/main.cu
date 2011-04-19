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
 * access. The bloc count is the depth of the allocated memory. All threads of
 * one block had to operate on a part of the width.
 */
void alloc_parent_matrix(struct instance *inst)
{
	assert(inst->num_matrices != 0);

	int width = inst->dim.threads    * /* each thread need his own parents */
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

	int width = inst->dim.threads * inst->dim.childs * /* each thread should have n childs */
		    inst->width_per_inst * sizeof(float);

	inst->dev_child_ext = make_cudaExtent(width,
					      inst->dim.matrix_height,
					      inst->dim.blocks);

	cudaPitchedPtr pitched_ptr;
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_child_ext));
	CUDA_CALL(cudaMemset3D(pitched_ptr, 0, inst->dev_child_ext));
	inst->dev_child = pitched_ptr;
}

void init_rnd_generator(struct instance *inst, int seed)
{	
	curandState *rnd_states;
	int count = inst->dim.blocks * inst->dim.threads * sizeof(curandState);

	CUDA_CALL(cudaMalloc((void **)&rnd_states, count));
	setup_rnd_kernel<<<BLOCKS, THREADS>>>(rnd_states, seed);
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
	inst->rules_len = 5;
	inst->rules = (int*)malloc(sizeof(int) * inst->rules_len);
	inst->rules[0] = 0;
	inst->rules[1] = 1;
	inst->rules[2] = MUL_SEP;
	inst->rules[3] = 1;
	inst->rules[4] = 0;

	inst->delta = 0.1;
	inst->match = MATCH_ALL;
	inst->cond_left  = COND_UPPER_RIGHT;
	inst->cond_right = COND_UPPER_RIGHT;

	inst->dim.blocks  = BLOCKS;
	inst->dim.threads = THREADS;
	inst->dim.childs  = CHILDS;
	inst->dim.matrix_width  = MATRIX_WIDTH;
	inst->dim.matrix_height = MATRIX_HEIGHT;
	
	set_num_matrices(inst);

	inst->width_per_inst = inst->num_matrices *    /* there are n matrices needed for the rules */
			       inst->dim.matrix_width; /* each one has a fixed width */

	alloc_parent_matrix(inst);
	alloc_child_matrix(inst);

	init_rnd_generator(inst, time(0));
}

void cleanup(struct instance *inst, struct instance * dev_inst) {
	free(inst->rules);

	cudaFree(dev_inst);
	/* dev_inst-> rules? */

	cudaFree(inst->rnd_states);
	cudaFree(inst->dev_child.ptr);
	cudaFree(inst->dev_parent.ptr);
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

int main(int argc, char** argv)
{
	struct instance inst;
	struct instance *dev_inst;

	init_instance(&inst);
	dev_inst = create_dev_inst(&inst);

	setup_parent_kernel<<<BLOCKS, MATRIX_HEIGHT>>>(dev_inst);
	CUDA_CALL(cudaGetLastError());
	print_parent_matrix(&inst);

//	evo_kernel<<<BLOCKS, THREADS>>>(dev_inst);
//	CUDA_CALL(cudaGetLastError());

	printf("Clean up and exit.\n");
	cleanup(&inst, dev_inst);
}
