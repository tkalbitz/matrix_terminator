#include "matrix_copy.h"

void copy_parents_dev_to_host(struct instance* inst, void* parent_cpy)
{
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr = inst->dev_parent;
	p.dstPtr = make_cudaPitchedPtr(
			parent_cpy,
			inst->dev_parent_ext.width,
			inst->dev_parent_ext.width / sizeof(float),
			inst->dim.matrix_height);

	p.extent = inst->dev_parent_ext;
	p.kind = cudaMemcpyDeviceToHost;
	CUDA_CALL(cudaMemcpy3D(&p));
}

void copy_parent_rating_dev_to_host(struct instance* inst, void* parent_rat_cpy)
{
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr = inst->dev_prat;
	p.dstPtr = make_cudaPitchedPtr(
			parent_rat_cpy,
			inst->dim.parents * sizeof(float),
			inst->dim.parents,
			1);

	p.extent = inst->dev_prat_ext;
	p.kind = cudaMemcpyDeviceToHost;
	CUDA_CALL(cudaMemcpy3D(&p));
}

void copy_results_dev_to_host(struct instance* inst, void* result_cpy)
{
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr = inst->dev_res;
	p.dstPtr = make_cudaPitchedPtr(
			result_cpy,
			inst->dev_res_ext.width,
			inst->dev_res_ext.width / sizeof(float),
			inst->dim.matrix_height);

	p.extent = inst->dev_res_ext;
	p.kind = cudaMemcpyDeviceToHost;
	CUDA_CALL(cudaMemcpy3D(&p));
}
