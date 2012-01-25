/*
 * pso_memory.h
 *
 *  Created on: Sep 28, 2011
 *      Author: tkalbitz
 */

#ifndef PSO_MEMORY_H_
#define PSO_MEMORY_H_

#define BLOCK_POS2      (blockIdx.x * inst.width_per_line * inst.dim.particles)
#define BLOCK_POS       (blockIdx.x * inst.width_per_line * inst.dim.particles)
#define MAT_POS(mat)    (BLOCK_POS + inst.dim.matrix_width * inst.dim.matrix_width * inst.dim.particles * (mat))
#define ELEM(mat, cy, cx) (MAT_POS(mat) + (cy) * inst.dim.matrix_width * inst.dim.particles + (cx) * inst.dim.particles + blockIdx.y)
#define ELEM_IDX(idx) (BLOCK_POS + (idx) * inst.dim.particles + blockIdx.y)
#define ELEM_BIDX(block_pos, particle, idx) ((block_pos) + (idx) * PARTICLE_COUNT + particle)

#define P_ROW(y)   ((double*) (mem->p_slice  + (y) * mem->p_pitch))
#define R_ROW(y)   ((double*) (mem->r_slice  + (y) * mem->r_pitch))
#define LB_ROW(y)  ((double*) (mem->lb_slice + (y) * mem->lb_pitch))
#define GB_ROW(y)  ((double*) (mem->gb_slice + (y) * mem->gb_pitch))
#define LBN_ROW(y) ((double*) (mem->lbn_slice + (y) * mem->lbn_pitch))
#define RT_ROW(y)  ((double*) (mem->rat_tmp_slice  + (y) * mem->rat_tmp_pitch))

//struct memory {
//	size_t p_pitch;
//	char  *p_slice;
//
//	size_t rat_tmp_pitch;
//	char  *rat_tmp_slice;
//
//	size_t lb_pitch;
//	char  *lb_slice;
//
//	size_t gb_pitch;
//	char  *gb_slice;
//
//	size_t r_pitch;
//	char  *r_slice;
//
//	size_t lbr_pitch;
//	char  *lbr_slice;
//
//	size_t lbn_pitch;
//	char  *lbn_slice;
//	int lbn_zero;
//
//	int p_zero;
//	int p_end;
//
//	int r_zero;
//	int r_end;
//
//	double* p_rat;
//	double* lb_rat;
//};
//
//__device__ static void pso_init_mem(const struct pso_instance* const inst,
//		                    struct memory * const mem)
//{
//	char* const  p_dev_ptr = (char*)inst->dev_particle.ptr;
//	const size_t p_pitch = inst->dev_particle.pitch;
//	const size_t p_slice_pitch = p_pitch * inst->dim.matrix_height;
//	char* const  p_slice = p_dev_ptr + blockIdx.x /* z */ * p_slice_pitch;
//	mem->p_pitch = p_pitch;
//	mem->p_slice = p_slice;
//
//	char* const  rat_tmp_dev_ptr = (char*)inst->dev_rat_tmp.ptr;
//	const size_t rat_tmp_pitch = inst->dev_rat_tmp.pitch;
//	const size_t rat_tmp_slice_pitch = rat_tmp_pitch * inst->dim.matrix_height;
//	char* const  rat_tmp_slice = rat_tmp_dev_ptr + blockIdx.x /* z */ * rat_tmp_slice_pitch;
//	mem->rat_tmp_pitch = rat_tmp_pitch;
//	mem->rat_tmp_slice = rat_tmp_slice;
//
//	char* const  lbest_dev_ptr = (char*)inst->dev_particle_lbest.ptr;
//	const size_t lbest_pitch = inst->dev_particle_lbest.pitch;
//	const size_t lbest_slice_pitch = lbest_pitch * inst->dim.matrix_height;
//	char* const  lbest_slice = lbest_dev_ptr + blockIdx.x /* z */ * lbest_slice_pitch;
//	mem->lb_pitch = lbest_pitch;
//	mem->lb_slice = lbest_slice;
//
//	char* const  gbest_dev_ptr = (char*)inst->dev_particle_gbest.ptr;
//	const size_t gbest_pitch = inst->dev_particle_gbest.pitch;
//	const size_t gbest_slice_pitch = gbest_pitch * inst->dim.matrix_height;
//	char* const  gbest_slice = gbest_dev_ptr + blockIdx.x /* z */ * gbest_slice_pitch;
//	mem->gb_pitch = gbest_pitch;
//	mem->gb_slice = gbest_slice;
//
//	/*
//	 * each thread represent one child which has a
//	 * defined pos in the matrix
//	 */
//	mem->p_zero = inst->width_per_inst * blockIdx.y;
//	mem->p_end  = inst->width_per_inst * (blockIdx.y + 1);
//
//	char* const  r_dev_ptr = (char*)inst->dev_res.ptr;
//	const size_t r_pitch = inst->dev_res.pitch;
//	const size_t r_slice_pitch = r_pitch * inst->dim.matrix_height;
//	char* const  r_slice = r_dev_ptr + blockIdx.x /* z */ * r_slice_pitch;
//	mem->r_pitch = r_pitch;
//	mem->r_slice = r_slice;
//
//	mem->r_zero = blockIdx.y * inst->dim.matrix_width;
//	mem->r_end  = mem->r_zero + inst->dim.matrix_width;
//
//	const char* const t_dev_ptr2 = (char*)inst->dev_prat.ptr;
//	mem->p_rat = (double*)(t_dev_ptr2 + blockIdx.x * inst->dev_prat.pitch);
//
//	const char* const t_dev_ptr3 = (char*)inst->dev_lbrat.ptr;
//	mem->lb_rat = (double*)(t_dev_ptr3 + blockIdx.x * inst->dev_lbrat.pitch);
//}

/* calculate the thread id for the current block topology */
__device__ inline int get_thread_id() {
	const int uniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
	const int uniqueThreadIndex = uniqueBlockIndex * blockDim.y * blockDim.x +
			              threadIdx.y * blockDim.x + threadIdx.x;
	return uniqueThreadIndex;
}


#endif /* PSO_MEMORY_H_ */
