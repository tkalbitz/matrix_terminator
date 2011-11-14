#include "matrix_copy.h"

void copy_parents_dev_to_host(struct instance* inst, void* parent_cpy)
{
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr = inst->dev_parent;
	p.dstPtr = make_cudaPitchedPtr(
			parent_cpy,
			inst->dev_parent_ext.width,
			inst->dev_parent_ext.width / sizeof(double),
			inst->dim.matrix_height);

	p.extent = inst->dev_parent_ext;
	p.kind = cudaMemcpyDeviceToHost;
	CUDA_CALL(cudaMemcpy3D(&p));
}

void copy_childs_dev_to_host(struct instance* inst, void* parent_cpy)
{
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr = inst->dev_child;
	p.dstPtr = make_cudaPitchedPtr(
			parent_cpy,
			inst->dev_child_ext.width,
			inst->dev_child_ext.width / sizeof(double),
			inst->dim.matrix_height);

	p.extent = inst->dev_child_ext;
	p.kind = cudaMemcpyDeviceToHost;
	CUDA_CALL(cudaMemcpy3D(&p));
}

void copy_parent_rating_dev_to_host(struct instance* inst, void* parent_rat_cpy)
{
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr = inst->dev_prat;
	p.dstPtr = make_cudaPitchedPtr(
			parent_rat_cpy,
			inst->dim.parents * sizeof(double),
			inst->dim.parents,
			1);

	p.extent = inst->dev_prat_ext;
	p.kind = cudaMemcpyDeviceToHost;
	CUDA_CALL(cudaMemcpy3D(&p));
}

void copy_child_rating_dev_to_host(struct instance* inst, void* child_rat_cpy)
{
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr = inst->dev_crat;
	p.dstPtr = make_cudaPitchedPtr(
			child_rat_cpy,
			inst->dev_crat_ext.width,
			inst->dev_crat_ext.width / sizeof(double),
			1);

	p.extent = inst->dev_crat_ext;
	p.kind = cudaMemcpyDeviceToHost;
	CUDA_CALL(cudaMemcpy3D(&p));
}

void copy_sparam_dev_to_host(struct instance* inst, void* sparam_cpy)
{
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr = inst->dev_sparam;
	p.dstPtr = make_cudaPitchedPtr(
			sparam_cpy,
			inst->dim.parents * inst->dim.childs * 3 * sizeof(double),
			inst->dim.parents * inst->dim.childs * 3,
			1);

	p.extent = inst->dev_sparam_ext;
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
			inst->dev_res_ext.width / sizeof(double),
			inst->dim.matrix_height);

	p.extent = inst->dev_res_ext;
	p.kind = cudaMemcpyDeviceToHost;
	CUDA_CALL(cudaMemcpy3D(&p));
}

void copy_debug_dev_to_host(struct instance* inst, void* debug_cpy)
{
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr = inst->dev_debug;
	p.dstPtr = make_cudaPitchedPtr(
			debug_cpy,
			inst->dev_debug_ext.width,
			inst->dev_debug_ext.width / sizeof(double),
			inst->dim.matrix_height);

	p.extent = inst->dev_debug_ext;
	p.kind = cudaMemcpyDeviceToHost;
	CUDA_CALL(cudaMemcpy3D(&p));
}
