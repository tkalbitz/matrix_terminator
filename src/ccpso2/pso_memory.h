/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef PSO_MEMORY_H_
#define PSO_MEMORY_H_

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

/* calculate the thread id for the current block topology */
__device__ inline int get_thread_id() {
	const int uniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
	const int uniqueThreadIndex = uniqueBlockIndex * blockDim.y * blockDim.x +
			              threadIdx.y * blockDim.x + threadIdx.x;
	return uniqueThreadIndex;
}


#endif /* PSO_MEMORY_H_ */
