/*
 * pso_instance.cu
 *
 *  Created on: Sep 28, 2011
 *      Author: tkalbitz
 */

#include "assert.h"

#include "pso_config.h"
#include "pso_instance.h"
#include "pso_setup.h"

int get_pso_threads(const struct pso_instance * const inst)
{
	return inst->dim.particles * inst->dim.matrix_width;
}

void init_rnd_generator(struct pso_instance *inst, int seed)
{
	curandState *rnd_states;
	const int count = max(get_pso_threads(inst), inst->dim.matrix_height);

	CUDA_CALL(cudaMalloc((void **)&rnd_states,
			     count * BLOCKS * sizeof(curandState)));
	setup_rnd_kernel<<<BLOCKS, count>>>(rnd_states, seed);
	CUDA_CALL(cudaGetLastError());
	cudaThreadSynchronize();
	inst->rnd_states = rnd_states;
}

void set_num_matrices(struct pso_instance* inst)
{
	int m = INT_MIN;
	for(int i = 0; i < inst->rules_len; i++)
		m = max(m, inst->rules[i]);

	inst->num_matrices = m + 1; /* matrices are zero based */
}

void alloc_particle_matrix(struct pso_instance *inst)
{
	assert(inst->num_matrices != 0);

	int width = inst->dim.particles  * /* there are n particles per block */
		    inst->width_per_inst *
		    sizeof(double);

	cudaPitchedPtr pitched_ptr;

	inst->dev_particle_ext = make_cudaExtent(width,
					         inst->dim.matrix_height,
					         inst->dim.blocks);
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_particle_ext));
	inst->dev_particle = pitched_ptr;

	inst->dev_particle_lbest_ext = make_cudaExtent(width,
					               inst->dim.matrix_height,
					               inst->dim.blocks);
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_particle_lbest_ext));
	inst->dev_particle_lbest = pitched_ptr;

	inst->dev_particle_gbest_ext = make_cudaExtent(inst->width_per_inst,
					               inst->dim.matrix_height,
					               inst->dim.blocks);
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_particle_gbest_ext));
	inst->dev_particle_gbest = pitched_ptr;
}

void alloc_params(struct pso_instance *inst)
{
	assert(inst->num_matrices != 0);

	int width = PARAM_COUNT * inst->dim.particles *
		    sizeof(double);

	cudaPitchedPtr pitched_ptr;

	inst->dev_params_ext = make_cudaExtent(width,
					         inst->dim.matrix_height,
					         inst->dim.blocks);
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_params_ext));
	inst->dev_params = pitched_ptr;
}

void alloc_result_matrix(struct pso_instance *inst)
{
	const int width = inst->dim.particles *
			    inst->dim.matrix_width * sizeof(double);

	inst->dev_res_ext = make_cudaExtent(width,
					    inst->dim.matrix_height,
					    inst->dim.blocks);

	cudaPitchedPtr pitched_ptr;
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_res_ext));
	CUDA_CALL(cudaMemset3D(pitched_ptr, 1, inst->dev_res_ext));
	inst->dev_res = pitched_ptr;
}

void alloc_rating(struct pso_instance *inst)
{
	inst->dev_prat_ext = make_cudaExtent(inst->dim.particles * sizeof(double),
					     1,
					     inst->dim.blocks);

	cudaPitchedPtr pitched_ptr;
	CUDA_CALL(cudaMalloc3D(&pitched_ptr, inst->dev_prat_ext));
	CUDA_CALL(cudaMemset3D(pitched_ptr, 33, inst->dev_prat_ext));
	inst->dev_prat = pitched_ptr;
}

void pso_inst_init(struct pso_instance* const inst, int matrix_width)
{
	inst->dim.blocks    = BLOCKS;
	inst->dim.particles = PARTICLE_COUNT;
	inst->dim.matrix_width  = matrix_width;
	inst->dim.matrix_height = matrix_width;

	inst->res_block = 0;
	inst->res_parent = 0;
	inst->res_child_block = 0;
	inst->res_child_idx = 0;

	set_num_matrices(inst);

	inst->width_per_inst = inst->num_matrices *    /* there are n matrices needed for the rules */
			       inst->dim.matrix_width; /* each one has a fixed width */

	alloc_particle_matrix(inst);
	alloc_params(inst);
	alloc_result_matrix(inst);
	alloc_rating(inst);
	init_rnd_generator(inst, time(0));
}

void pso_inst_cleanup(struct pso_instance * const inst,
		  struct pso_instance * const dev_inst)
{
	if(dev_inst != NULL)
		cudaFree(dev_inst);

	cudaFree(inst->rnd_states);
	cudaFree(inst->dev_params.ptr);
	cudaFree(inst->dev_particle.ptr);
	cudaFree(inst->dev_particle_gbest.ptr);
	cudaFree(inst->dev_particle_lbest.ptr);
	cudaFree(inst->dev_res.ptr);
	cudaFree(inst->dev_prat.ptr);
}

struct pso_instance* pso_inst_create_dev_inst(struct pso_instance *inst,
					      int** dev_rules)
{
	struct pso_instance *dev_inst;
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

void pso_inst_copy_dev_to_host(struct pso_instance * const dev,
			   struct pso_instance * const host)
{
	int *rules = host->rules;
	CUDA_CALL(cudaMemcpy(host, dev, sizeof(*dev), cudaMemcpyDeviceToHost));
	host->rules = rules;
}

