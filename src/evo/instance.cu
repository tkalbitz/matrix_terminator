#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "instance.h"
#include "evo_setup.h"

int get_evo_threads(const struct instance * const inst)
{
	return inst->dim.parents * inst->dim.childs;
}

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
	const int width = inst->dim.childs * inst->dim.parents *
			    inst->dim.matrix_width * sizeof(double);

	inst->dev_res_ext = make_cudaExtent(width,
					    inst->dim.matrix_height,
					    inst->dim.blocks);

	cudaPitchedPtr pitched_ptr;
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_res_ext));
	CUDA_CALL(cudaMemset3D(pitched_ptr, 1, inst->dev_res_ext));
	inst->dev_res = pitched_ptr;
}

void alloc_sparam(struct instance *inst)
{
	inst->dev_sparam_ext = make_cudaExtent(inst->dim.childs *
					       inst->dim.parents * 3 *
					       sizeof(double),
					       1,
					       inst->dim.blocks);

	cudaPitchedPtr pitched_ptr;
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_sparam_ext));
	inst->dev_sparam = pitched_ptr;

	inst->dev_psparam_ext = make_cudaExtent(inst->dim.parents * 3 *
						sizeof(double),
					        1,
					        inst->dim.blocks);

	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_psparam_ext));
	inst->dev_psparam = pitched_ptr;
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
	CUDA_CALL(cudaMemset3D(pitched_ptr, 33, inst->dev_crat_ext));
	inst->dev_crat = pitched_ptr;

	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_prat_ext));
	CUDA_CALL(cudaMemset3D(pitched_ptr, 33, inst->dev_prat_ext));
	inst->dev_prat = pitched_ptr;
}

void alloc_debug_matrix(struct instance *inst)
{
#ifdef DEBUG
	/* memory for every rule side of every rule */
	const int width = inst->dim.childs * inst->dim.parents *
			    inst->dim.matrix_width * inst->rules_count *
			    2 * sizeof(double);
#else
	const int width = 1;
#endif

	inst->dev_debug_ext = make_cudaExtent(width,
					    inst->dim.matrix_height,
					    inst->dim.blocks);

	cudaPitchedPtr pitched_ptr;
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_debug_ext));
	CUDA_CALL(cudaMemset3D(pitched_ptr, 1, inst->dev_debug_ext));
	inst->dev_debug = pitched_ptr;
}

void init_rnd_generator(struct instance *inst, int seed)
{
	curandState *rnd_states;
	const int count = max(get_evo_threads(inst), inst->dim.matrix_height);

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
//	printf("num_matrices set to %d\n", inst->num_matrices);
}

void inst_init(struct instance* const inst, int matrix_width)
{
	inst->dim.blocks  = BLOCKS;
	inst->dim.childs  = CHILDS;
	inst->dim.parents = PARENTS;
	inst->dim.matrix_width  = matrix_width;
	inst->dim.matrix_height = matrix_width;

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
	alloc_debug_matrix(inst);
	init_rnd_generator(inst, time(0));
}

void inst_cleanup(struct instance * const inst,
		  struct instance * const dev_inst)
{
	if(dev_inst != NULL)
		cudaFree(dev_inst);

	cudaFree(inst->rnd_states);
	cudaFree(inst->dev_child.ptr);
	cudaFree(inst->dev_parent.ptr);
	cudaFree(inst->dev_res.ptr);
	cudaFree(inst->dev_crat.ptr);
	cudaFree(inst->dev_prat.ptr);
	cudaFree(inst->dev_sparam.ptr);
	cudaFree(inst->dev_debug.ptr);
}

struct instance* inst_create_dev_inst(struct instance *inst, int** dev_rules)
{
	struct instance *dev_inst;
	int *rules = inst->rules;
	int *tmp_dev_rules;
	CUDA_CALL(cudaMalloc(&tmp_dev_rules, inst->rules_len * sizeof(int)));
	CUDA_CALL(cudaMemcpy(tmp_dev_rules,  rules, inst->rules_len * sizeof(int),
					cudaMemcpyHostToDevice));
	inst->rules = tmp_dev_rules;
	CUDA_CALL(cudaMalloc(&dev_inst, sizeof(*dev_inst)));
	CUDA_CALL(cudaMemcpy(dev_inst, inst, sizeof(*dev_inst),
					cudaMemcpyHostToDevice));
	inst->rules = rules;

	if(dev_rules != NULL)
		*dev_rules = tmp_dev_rules;

	return dev_inst;
}

void inst_copy_dev_to_host(struct instance * const dev,
			   struct instance * const host)
{
	int *rules = host->rules;
	CUDA_CALL(cudaMemcpy(host, dev, sizeof(*dev), cudaMemcpyDeviceToHost));
	host->rules = rules;
}

