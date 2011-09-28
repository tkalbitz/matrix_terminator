/*
 * pso_memory.h
 *
 *  Created on: Sep 28, 2011
 *      Author: tkalbitz
 */

#ifndef PSO_MEMORY_H_
#define PSO_MEMORY_H_

#define P_ROW(y) ((double*) (mem->p_slice + (y) * mem->p_pitch))
#define R_ROW(y) ((double*) (mem->r_slice + (y) * mem->r_pitch))

#define W(x)  (mem->param[3*(x)])
#define C1(x) (mem->param[3*(x)+1])
#define C2(x) (mem->param[3*(x)+2])

struct memory {
	size_t p_pitch;
	char  *p_slice;

	size_t r_pitch;
	char  *r_slice;

	int p_zero;
	int p_end;

	int r_zero;
	int r_end;

	double* p_rat;
	double* param;
};

__device__ static void pso_init_mem(const struct pso_instance* const inst,
		                    struct memory * const mem)
{
	char* const p_dev_ptr = (char*)inst->dev_particle.ptr;
	const size_t p_pitch = inst->dev_particle.pitch;
	const size_t p_slice_pitch = p_pitch * inst->dim.matrix_height;
	char* const p_slice = p_dev_ptr + blockIdx.x /* z */ * p_slice_pitch;
	mem->p_pitch = p_pitch;
	mem->p_slice = p_slice;

	/*
	 * each thread represent one child which has a
	 * defined pos in the matrix
	 */
	mem->p_zero = inst->width_per_inst * blockIdx.y;
	mem->p_end  = inst->width_per_inst * (blockIdx.y + 1);

	char* const r_dev_ptr = (char*)inst->dev_res.ptr;
	const size_t r_pitch = inst->dev_res.pitch;
	const size_t r_slice_pitch = r_pitch * inst->dim.matrix_height;
	char* const r_slice = r_dev_ptr + blockIdx.x /* z */ * r_slice_pitch;
	mem->r_pitch = r_pitch;
	mem->r_slice = r_slice;

	mem->r_zero = blockIdx.y * inst->dim.matrix_width;
	mem->r_end  = mem->r_zero + inst->dim.matrix_width;

	const char* const t_dev_ptr2 = (char*)inst->dev_prat.ptr;
	mem->p_rat = (double*) (t_dev_ptr2 + blockIdx.x * inst->dev_prat.pitch);

	const char* const s_dev_ptr = (char*)inst->dev_params.ptr;
	mem->param  = (double*)(s_dev_ptr + blockIdx.x * inst->dev_params.pitch);
}

/* calculate the thread id for the current block topology */
__device__ inline static int get_thread_id() {
	return threadIdx.x + blockIdx.x * blockDim.x;
}


#endif /* PSO_MEMORY_H_ */
