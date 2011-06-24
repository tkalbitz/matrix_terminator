#include <limits.h>
#include <float.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "instance.h"
#include "config.h"

#include "evo_memory.cu"

__device__ void double_memcpy(double* to, double* from, int size)
{
	while(size--) {
		*to = *from;
		to++;
		from++;
	}
}

__device__ void copy_child_to_parent(struct instance * const inst,
				     struct memory   * const mem,
				     const int child,
				     const int parent)
{
	const int cstart = child * inst->width_per_inst;
	const int pstart = parent * inst->width_per_inst;
	const int rows = MATRIX_HEIGHT;

	for(int r = 0; r < rows; r++) {
		double* const prow = P_ROW(r);
		double* const crow = C_ROW(r);

		double_memcpy(&(prow[pstart]),
		              &(crow[cstart]),
		              inst->width_per_inst);
	}
}

#include "evo_recombination.cu"
#include "evo_adaptive_gauss_mutation.cu"
#include "evo_selection.cu"

__global__ void evo_kernel(struct instance *inst, int flag)
{
	const int id = get_thread_id();

	/* copy global state to local mem for efficiency */
	curandState rnd_state = inst->rnd_states[id];

	struct memory mem;
	evo_init_mem(inst, &mem);

	int p_sel[2];
	double* sparam = get_sparam_arr(inst);

	const int tx = threadIdx.x;

	if(flag == 0) {
		evo_recomb_selection(inst, &rnd_state, p_sel);
		evo_recombination(inst, &mem, &rnd_state, p_sel);
		evo_mutation(inst, &mem, &rnd_state, &sparam[tx]);
	} else {
		evo_parent_selection_turnier(inst, &mem, &rnd_state, 3);
		__syncthreads();

		/* Parallel copy of memory */
		if(threadIdx.x < inst->dim.parents) {
			copy_child_to_parent(inst, &mem,
					     (int)mem.c_rat[2 * tx + 1], tx);
			mem.p_rat[threadIdx.x] = mem.c_rat[2 * tx];
		}
	}

	/* backup rnd state to global mem */
	inst->rnd_states[id] = rnd_state;
}
