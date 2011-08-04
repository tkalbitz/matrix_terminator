#ifndef EVO_MEMORY_H_
#define EVO_MEMORY_H_

#define C_ROW(y) ((double*) (mem->c_slice + (y) * mem->c_pitch))
#define P_ROW(y) ((double*) (mem->p_slice + (y) * mem->p_pitch))
#define R_ROW(y) ((double*) (mem->r_slice + (y) * mem->r_pitch))
#define CR_ROW(y) ((double*) (res_mem.r_slice + (y) * res_mem.r_pitch))

struct memory {
	size_t p_pitch;
	char  *p_slice;

	size_t c_pitch;
	char  *c_slice;

	int c_zero;
	int c_end;

#ifdef DEBUG
	size_t r_pitch;
	char  *r_slice;

	int r_zero1;
	int r_zero2;
	int r_end1;
	int r_end2;
#endif

	double* c_rat;
	double* p_rat;
	double* sparam;
	double* psparam;
};

__device__ static void evo_init_mem(const struct instance* const inst,
		                    struct memory * const mem)
{
	char* const p_dev_ptr = (char*)inst->dev_parent.ptr;
	const size_t p_pitch = inst->dev_parent.pitch;
	const size_t p_slice_pitch = p_pitch * inst->dim.matrix_height;
	char* const p_slice = p_dev_ptr + blockIdx.x /* z */ * p_slice_pitch;
	mem->p_pitch = p_pitch;
	mem->p_slice = p_slice;

	char* const c_dev_ptr = (char*)inst->dev_child.ptr;
	const size_t c_pitch = inst->dev_child.pitch;
	const size_t c_slice_pitch = c_pitch * inst->dim.matrix_height;
	char* const c_slice = c_dev_ptr + blockIdx.x /* z */ * c_slice_pitch;
	mem->c_pitch = c_pitch;
	mem->c_slice = c_slice;

	/*
	 * each thread represent one child which has a
	 * defined pos in the matrix
	 */
	mem->c_zero = inst->width_per_inst * threadIdx.x;
	mem->c_end  = inst->width_per_inst * (threadIdx.x + 1);

#ifdef DEBUG
	char* const r_dev_ptr = (char*)inst->dev_res.ptr;
	const size_t r_pitch = inst->dev_res.pitch;
	const size_t r_slice_pitch = r_pitch * inst->dim.matrix_height;
	char* const r_slice = r_dev_ptr + blockIdx.x /* z */ * r_slice_pitch;

	mem->r_pitch = r_pitch;
	mem->r_slice = r_slice;

	mem->r_zero1 = threadIdx.x * 2 * inst->dim.matrix_width;
	mem->r_end1  = mem->r_zero1 + inst->dim.matrix_width;
	mem->r_zero2 = mem->r_zero1 + inst->dim.matrix_width;
	mem->r_end2  = mem->r_zero2 + inst->dim.matrix_width;
#endif

	const char* const t_dev_ptr = (char*)inst->dev_crat.ptr;
	mem->c_rat = (double*) (t_dev_ptr + blockIdx.x * inst->dev_crat.pitch);

	const char* const t_dev_ptr2 = (char*)inst->dev_prat.ptr;
	mem->p_rat = (double*) (t_dev_ptr2 + blockIdx.x * inst->dev_prat.pitch);

	const char* const s_dev_ptr = (char*)inst->dev_sparam.ptr;
	mem->sparam  = (double*)(s_dev_ptr + blockIdx.x * inst->dev_sparam.pitch);
	const char* const ps_dev_ptr = (char*)inst->dev_psparam.ptr;
	mem->psparam = (double*)(ps_dev_ptr + blockIdx.x * inst->dev_psparam.pitch);
}

/* calculate the thread id for the current block topology */
__device__ inline static int get_thread_id() {
	return threadIdx.x + blockIdx.x * blockDim.x;
}

#endif /* EVO_MEMORY_H_ */
