#include <limits.h>
#include <float.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "instance.h"
#include "config.h"

__device__ double* get_sparam_arr(struct instance* inst)
{
	char* s_dev_ptr = (char*)inst->dev_sparam.ptr;
	size_t s_pitch = inst->dev_sparam.pitch;
	size_t s_slice_pitch = s_pitch * 1;
	char* s_slice = s_dev_ptr + blockIdx.x /* z */ * s_slice_pitch;
	double* sparam = (double*) (s_slice + 0 * s_pitch);
	return sparam;
}

__device__ void double_memcpy(double* to, double* from, int size)
{
//	memcpy(to, from, size * sizeof(double));
	while(size--) {
		*to = *from;
		to++;
		from++;
	}

//	if(((unsigned long long)to   & 0x4) ||
//	   ((unsigned long long)from & 0x4)) {
//		while(size--) {
//			*to = *from;
//			to++;
//			from++;
//		}
//	} else {
//		long long *t = (long long*)to;
//		long long *f = (long long*)from;
//		int s = size >> 1;
//
//		while(s--) {
//			*t = *f;
//			t++;
//			f++;
//		}
//
//		if(size & 1) {
//			to   = (double*)t;
//			from = (double*)f;
//			*to  = *from;
//		}
//	}
}

/* calculate the thread id for the current block topology */
inline __device__ int get_thread_id() {
	return threadIdx.x + blockIdx.x * blockDim.x;
}

__global__ void setup_rnd_kernel(curandState* rnd_states,
				 int seed)
{
	int id = get_thread_id();

	/* 
         * Each thread get the same seed, 
         * a different sequence number and no offset. 
         */
	curand_init(seed, id, 0, &rnd_states[id]);
}

/*
 * Initialize the parent memory with random values.
 */
__global__ void setup_parent_kernel(struct instance *inst)
{
	if(threadIdx.x >= inst->dim.matrix_height)
		return;

	int id = get_thread_id();
	curandState rnd_state = inst->rnd_states[id];

	char* devPtr = (char*)inst->dev_parent.ptr;
	size_t pitch = inst->dev_parent.pitch;
	size_t slicePitch = pitch * inst->dim.matrix_height;
	char* slice = devPtr + blockIdx.x * slicePitch;
	double* row = (double*) (slice + threadIdx.x * pitch);

	for(int x = 0; x < inst->dim.parents * inst->width_per_inst; x++) {
		if(curand_uniform(&rnd_state) < MATRIX_TAKEN_POS) {
			row[x] = curand(&rnd_state) % (int)PARENT_MAX;
		} else {
			row[x] = 0;
		}
	}

	inst->rnd_states[id] = rnd_state;

	if(threadIdx.x != 0)
		return;

	const int matrices = inst->num_matrices * inst->dim.parents;
	int y;

	if(inst->cond_left == COND_UPPER_LEFT) {
		y = 0;
		row = (double*) (slice + y * pitch);

		for(int i = 0; i < matrices; i++) {
			row[i * inst->dim.matrix_width] = 1;
		}
	} else if(inst->cond_left == COND_UPPER_RIGHT) {
		y = 0;
		row = (double*) (slice + y * pitch);

		for(int i = 0; i < matrices; i++) {
			int idx = i * inst->dim.matrix_width + (inst->dim.matrix_width - 1);
			row[idx] = 1;
		}
	} else if(inst->cond_left == COND_UPPER_LEFT_LOWER_RIGHT) {
		y = 0;
		row = (double*) (slice + y * pitch);
		for(int i = 0; i < matrices; i++) {
			row[i * inst->dim.matrix_width] = 1;
		}

		y = (inst->dim.matrix_height - 1);
		row = (double*) (slice + y * pitch);
		for(int i = 0; i < matrices; i++) {
			int idx = i * inst->dim.matrix_width + (inst->dim.matrix_width - 1);
			row[idx] = 1;
		}
	}
}

#define C_ROW(y) ((double*) (mem->c_slice + y * mem->c_pitch))
#define P_ROW(y) ((double*) (mem->p_slice + y * mem->p_pitch))
#define R_ROW(y) ((double*) (mem->r_slice + y * mem->r_pitch))
#define CR_ROW(y) ((double*) (res_mem.r_slice + y * res_mem.r_pitch))

struct memory {
	size_t p_pitch;
	char  *p_slice;

	size_t c_pitch;
	char  *c_slice;

	int c_zero;
	int c_end;

	size_t r_pitch;
	char  *r_slice;

	int r_zero1;
	int r_zero2;
	int r_end1;
	int r_end2;

	double* c_rat;
	double* p_rat;
};

__device__ void evo_init_mem(struct instance* inst, struct memory *mem)
{
	char* p_dev_ptr = (char*)inst->dev_parent.ptr;
	size_t p_pitch = inst->dev_parent.pitch;
	size_t p_slice_pitch = p_pitch * inst->dim.matrix_height;
	char* p_slice = p_dev_ptr + blockIdx.x /* z */ * p_slice_pitch;

	char* c_dev_ptr = (char*)inst->dev_child.ptr;
	size_t c_pitch = inst->dev_child.pitch;
	size_t c_slice_pitch = c_pitch * inst->dim.matrix_height;
	char* c_slice = c_dev_ptr + blockIdx.x /* z */ * c_slice_pitch;

	char* r_dev_ptr = (char*)inst->dev_res.ptr;
	size_t r_pitch = inst->dev_res.pitch;
	size_t r_slice_pitch = r_pitch * inst->dim.matrix_height;
	char* r_slice = r_dev_ptr + blockIdx.x /* z */ * r_slice_pitch;

	mem->p_pitch = p_pitch;
	mem->p_slice = p_slice;
	mem->c_pitch = c_pitch;
	mem->c_slice = c_slice;
	mem->r_pitch = r_pitch;
	mem->r_slice = r_slice;

	/*
	 * each thread represent one child which has a
	 * defined pos in the matrix
	 */
	mem->c_zero = inst->width_per_inst * threadIdx.x;
	mem->c_end  = inst->width_per_inst * (threadIdx.x + 1);

	mem->r_zero1 = threadIdx.x * 2 * inst->dim.matrix_width;
	mem->r_end1  = mem->r_zero1 + inst->dim.matrix_width;
	mem->r_zero2 = mem->r_zero1 + inst->dim.matrix_width;
	mem->r_end2  = mem->r_zero2 + inst->dim.matrix_width;

	char* t_dev_ptr = (char*)inst->dev_crat.ptr;
	size_t t_pitch = inst->dev_crat.pitch;
	size_t t_slice_pitch = t_pitch * 1;
	char* t_slice = t_dev_ptr + blockIdx.x /* z */ * t_slice_pitch;
	mem->c_rat = (double*) (t_slice + 0 * t_pitch);

	t_dev_ptr = (char*)inst->dev_prat.ptr;
	t_pitch = inst->dev_prat.pitch;
	t_slice_pitch = t_pitch * 1;
	t_slice = t_dev_ptr + blockIdx.x /* z */ * t_slice_pitch;
	mem->p_rat = (double*) (t_slice + 0 * t_pitch);
}

/*
 * Select two parents for recombination. Selection is currently complete uniform.
 */
inline __device__ void evo_recomb_selection(struct instance *inst, curandState *rnd_state, int *sel)
{
	sel[0] = curand(rnd_state) % inst->dim.parents;
	sel[1] = curand(rnd_state) % inst->dim.parents;
}

/* A uniform crossover recombination. */
__device__ void evo_recombination(struct instance *inst, 
				  struct memory   *mem,
				  curandState *rnd_state,
				  int *sel)
{
	const int rows = MATRIX_HEIGHT;
	const int cols = inst->width_per_inst;

	const int p1   = sel[0] * inst->width_per_inst;
	const int p2   = sel[1] * inst->width_per_inst;
	const int cIdx = mem->c_zero;

	for(int r = 0; r < rows; r++) {
		double* const c_row = C_ROW(r);
		double* const p_row = P_ROW(r);

		for(int c = 0; c < cols; c++) {
			if(curand_uniform(rnd_state) > RECOMB_RATE) {
				c_row[cIdx + c] = p_row[p1 + c];
			} else {
				c_row[cIdx + c] = p_row[p2 + c];
			}
		}
	}
}

__device__ void evo_ensure_constraints(struct instance *inst,
				       struct memory   *mem)
{
	double* row = C_ROW(0);
	int end   = mem->c_end;

	int factor = (int)(1.f / inst->delta);
	if((factor * inst->delta) < 1.f)
		factor++;

	double val = factor * inst->delta;

	for(int start = mem->c_zero; start < end; start += inst->dim.matrix_width) {
		row = C_ROW(0);
		if(inst->cond_right == COND_UPPER_LEFT && row[start] < 1.f) {
			row[start] = val;
		} else if(inst->cond_right == COND_UPPER_RIGHT &&
			  row[start + inst->dim.matrix_width - 1] < 1.f)
		{
			row[start + inst->dim.matrix_width - 1] = val;
		} else if(inst->cond_right == COND_UPPER_LEFT_LOWER_RIGHT) {
			if(row[start] < 1.f)
				row[start] = val;

			row = C_ROW(inst->dim.matrix_height - 1);
			if(row[start + inst->dim.matrix_width - 1] < 1.f)
				row[start + inst->dim.matrix_width - 1] = val;
		}
	}
}

__device__ void evo_mutation(struct instance * const inst,
			     struct memory   * const mem,
			     curandState     * const rnd_state,
                             double          * const s_param)
{
	*s_param = *s_param * exp(curand_normal(rnd_state) / 1000);
	const int rows = MATRIX_HEIGHT;
	const double delta = inst->delta;
	double tmp;

	#pragma unroll
	for(int r = 0; r < rows; r++) {
		double* const row = C_ROW(r);

		for(int c = mem->c_zero; c < mem->c_end; c++) {

			if(curand_uniform(rnd_state) > MUT_RATE)
				continue;

			tmp = row[c];
			tmp = tmp + (double)(curand_normal(rnd_state) * (*s_param));
			/* we want x * delta, where x is an int */  	
			tmp = ((unsigned long)(tmp / delta)) * delta;
			tmp = max(tmp, 0.0);
			tmp = min(PARENT_MAX, tmp);

			row[c] = tmp;
		}
	}

	evo_ensure_constraints(inst, mem);
}

__device__ void evo_parent_selection(struct instance *inst, struct memory *mem)
{
	const int elems = 2 * inst->dim.childs * inst->dim.parents;
	double* const arr = mem->c_rat;

	double key, child;

	/* insertion sort */
	for(int i = 2; i < elems; i+=2) {
		key   = arr[i];
		child = arr[i+1];

		int j = i - 2;
		while(j >=0 && arr[j] > key) {
			arr[j + 2] = arr[j];
			arr[j + 3] = arr[j+1];
			j = j - 2;
		}
		arr[j + 2] = key;
		arr[j + 3] = child;
	}
}

__device__ void copy_child_to_parent(struct instance *inst,
				     struct memory   *mem,
				     int child,
				     int parent)
{
	const int cstart = child * inst->width_per_inst;
	const int pstart = parent * inst->width_per_inst;
	const int rows = MATRIX_HEIGHT;

	for(int r = 0; r < rows; r++) {
		double* prow = P_ROW(r);
		double* crow = C_ROW(r);

		double_memcpy(&(prow[pstart]),
		             &(crow[cstart]),
		             inst->width_per_inst);
	}
}

/* extern device functions can't be inlined, so include them */
//#include "ensure.cu"
#include "evo_rating2.cu"

__global__ void init_sparam(struct instance *inst)
{
	get_sparam_arr(inst)[threadIdx.x] = 5.;
}

__global__ void evo_kernel(struct instance *inst)
{
	int id = get_thread_id();

	/* copy global state to local mem for efficiency */
	curandState rnd_state = inst->rnd_states[id];

	struct memory mem;
	evo_init_mem(inst, &mem);

	int p_sel[2];
	double s_param = 5.f; /* TODO: For every matrix? */

	while(inst->cont && inst->rounds < 5000) {
		evo_recomb_selection(inst, &rnd_state, p_sel);

		evo_recombination(inst, &mem, &rnd_state, p_sel);
		evo_mutation(inst, &mem, &rnd_state, &s_param);

		//mem.c_rat[2 * threadIdx.x]     = evo_calc_res(inst, &mem);
		mem.c_rat[2 * threadIdx.x + 1] = threadIdx.x;
		if(mem.c_rat[2 * threadIdx.x] == 0.f) {
			atomicExch(&inst->res_child_block, (unsigned int)blockIdx.x);
			atomicExch(&inst->res_child_idx,   (unsigned int)threadIdx.x);
			atomicExch(&(inst->cont), 0);
		}

		__syncthreads();

		/*
		 * All threads should rated their results here.
		 * It's time to get the new parents :D
		 */
		if(threadIdx.x == 0) {
			if(blockIdx.x == 0) {
				inst->rounds++;
			}

			evo_parent_selection(inst, &mem);
			if(mem.c_rat[0] == 0.f) {
				atomicExch(&(inst->res_block), blockIdx.x);
				atomicExch(&(inst->res_parent), threadIdx.x);
			}
		}

		__syncthreads();

		/* Parallel copy of memory */
		if(threadIdx.x < inst->dim.parents) {
			copy_child_to_parent(inst, &mem,
					     (int)mem.c_rat[2 * threadIdx.x + 1],
					     threadIdx.x);
			mem.p_rat[threadIdx.x] = mem.c_rat[2 * threadIdx.x];
		}

		__syncthreads();
	}

	/* backup rnd state to global mem */
	inst->rnd_states[id] = rnd_state;
}

__global__ void evo_kernel_test(struct instance *inst, int flag)
{
	int id = get_thread_id();

	/* copy global state to local mem for efficiency */
	curandState rnd_state = inst->rnd_states[id];

	struct memory mem;
	evo_init_mem(inst, &mem);

	int p_sel[2];
	double* sparam = get_sparam_arr(inst);

	if(flag == 0) {
		evo_recomb_selection(inst, &rnd_state, p_sel);
		evo_recombination(inst, &mem, &rnd_state, p_sel);
		evo_mutation(inst, &mem, &rnd_state, &sparam[threadIdx.x]);
	} else {
		if(threadIdx.x == 0) {
			evo_parent_selection(inst, &mem);
			if(mem.c_rat[0] == 0.f) {
				atomicExch(&(inst->res_block), blockIdx.x);
				atomicExch(&(inst->res_parent), threadIdx.x);
			}
		}

		__syncthreads();

		/* Parallel copy of memory */
		if(threadIdx.x < inst->dim.parents) {
			copy_child_to_parent(inst, &mem,
					     (int)mem.c_rat[2 * threadIdx.x + 1],
					     threadIdx.x);
			mem.p_rat[threadIdx.x] = mem.c_rat[2 * threadIdx.x];
		}
	}

	/* backup rnd state to global mem */
	inst->rnd_states[id] = rnd_state;
}

__global__ void evo_kernel_test2(struct instance *inst) {
	int id = get_thread_id();

	/* copy global state to local mem for efficiency */
	curandState rnd_state = inst->rnd_states[id];

	struct memory mem;
	evo_init_mem(inst, &mem);

	int p_sel[2];
	double s_param = 5.f; /* TODO: For every matrix? */

	while(inst->rounds < 100) {
		evo_recomb_selection(inst, &rnd_state, p_sel);

		evo_recombination(inst, &mem, &rnd_state, p_sel);
		evo_mutation(inst, &mem, &rnd_state, &s_param);

//		mem.c_rat[2 * threadIdx.x]     = evo_calc_res(inst, &mem);
		mem.c_rat[2 * threadIdx.x + 1] = threadIdx.x;
		if(mem.c_rat[2 * threadIdx.x] == 0.f) {
			atomicExch(&inst->res_child_block, (unsigned int)blockIdx.x);
			atomicExch(&inst->res_child_idx,   (unsigned int)threadIdx.x);
			atomicExch(&(inst->cont), 0);
		}

		__syncthreads();

		/*
		 * All threads should rated their results here.
		 * It's time to get the new parents :D
		 */
		if(threadIdx.x == 0) {
			if(blockIdx.x == 0) {
				inst->rounds++;
			}

			evo_parent_selection(inst, &mem);
			if(mem.c_rat[0] == 0.f) {
				atomicExch(&(inst->res_block), blockIdx.x);
				atomicExch(&(inst->res_parent), threadIdx.x);
			}
		}

		__syncthreads();

		/* Parallel copy of memory */
		if(threadIdx.x < inst->dim.parents) {
			copy_child_to_parent(inst, &mem,
					     (int)mem.c_rat[2 * threadIdx.x + 1],
					     threadIdx.x);
			mem.p_rat[threadIdx.x] = mem.c_rat[2 * threadIdx.x];
		}

		__syncthreads();
	}

	/* backup rnd state to global mem */
	inst->rnd_states[id] = rnd_state;
}
