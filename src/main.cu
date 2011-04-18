#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <cuda.h> 
#include <curand_kernel.h>

#include "config.h"
#include "instance.h"
#include "evo.h"

#define CUDA_CALL(x) do { cudaError_t xxs = (x); \
	if((xxs) != cudaSuccess) { \
		fprintf(stderr, "Error '%s' at %s:%d\n",cudaGetErrorString(xxs),__FILE__,__LINE__); \
		exit(EXIT_FAILURE);}} while(0)

/*
 * Allocate memory for the parent matrices. the memory is layouted for faster
 * access. The bloc count is the depth of the allocated memory. All threads of
 * one block had to operate on a part of the width.
 */
void alloc_parent_matrix(struct instance *inst)
{
	inst->dev_parent_ext = make_cudaExtent(
		inst->dim.threads * inst->dim.matrix_width * sizeof(float),
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
	inst->dev_child_ext = make_cudaExtent(
		inst->dim.threads * inst->dim.childs * inst->dim.matrix_width * sizeof(float),
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

void init_instance(struct instance* inst)
{
	inst->rules = "ab,ba";
	inst->delta = 0.1;
	inst->match = MATCH_ALL;
	inst->cond_left  = COND_UPPER_RIGHT;
	inst->cond_right = COND_UPPER_RIGHT;

	inst->dim.blocks  = BLOCKS;
	inst->dim.threads = THREADS;
	inst->dim.childs  = CHILDS;
	inst->dim.matrix_width  = MATRIX_WIDTH;
	inst->dim.matrix_height = MATRIX_HEIGHT;
	
	alloc_parent_matrix(inst);
	alloc_child_matrix(inst);

	init_rnd_generator(inst, time(0));
}

void cleanup(struct instance *inst) {
	cudaFree(inst->rnd_states);
}

void printParentMatrix(struct instance inst)
{
    float parent_cpy[BLOCKS][MATRIX_HEIGHT][MATRIX_WIDTH*THREADS];

    cudaMemcpy3DParms p = {0};
    p.srcPtr = inst.dev_parent;
    p.dstPtr = make_cudaPitchedPtr((void*)parent_cpy, inst.dim.threads * inst.dim.matrix_width * sizeof (float), inst.dim.threads * inst.dim.matrix_width, inst.dim.matrix_height);
    p.extent = inst.dev_parent_ext;
    p.kind = cudaMemcpyDeviceToHost;
    CUDA_CALL(cudaMemcpy3D(&p));
    for(int b = 0;b < inst.dim.blocks;b++){
        for(int h = 0;h < inst.dim.matrix_height;h++){
            for(int w = 0;w < inst.dim.threads * inst.dim.matrix_width;w++){
                printf("%3.2e ", parent_cpy[b][h][w]);
            }
            printf("\n");
        }

    }
}

int main(int argc, char** argv)
{
	struct instance inst;
	struct instance *dev_inst;

	init_instance(&inst);

	CUDA_CALL(cudaMalloc(&dev_inst, sizeof(struct instance)));
	CUDA_CALL(cudaMemcpy(dev_inst, &inst, sizeof(struct instance), cudaMemcpyHostToDevice));

	setup_parent_kernel<<<BLOCKS, THREADS>>>(dev_inst);
	CUDA_CALL(cudaGetLastError());

	printParentMatrix(inst);
	//evo_kernel<<<BLOCKS, THREADS>>>(dev_inst);

	cudaFree(dev_inst);
	printf("Clean up and exit.\n");
}
