/*
 * evo_selection.cu
 *
 *  Created on: Jun 24, 2011
 *      Author: tkalbitz
 */

__device__ void evo_parent_selection_best(struct instance * const inst,
					  struct memory   * const mem)
{
	if(ty != 0)
		return;

	__shared__ struct double2 res[PARENTS * CHILDS];
	double2* const arr   = (double2*)mem->c_rat;

	for(int i = tx; i < PARENTS * CHILDS; i += MATRIX_WIDTH) {
		res[i]   = arr[i];
	}

	__syncthreads();

	if(tx == 0) {
		double2 key;

		/* insertion sort */
		for(int i = 1; i < PARENTS * CHILDS; i++) {
			key = res[i];

			int j = i - 1;
			while(j >=0 && res[j].x > key.x) {
				res[j + 1] = res[j];
				j = j - 1;
			}
			res[j + 1] = key;
		}
	}

	__syncthreads();

	for(int i = tx; i < PARENTS * CHILDS; i += MATRIX_WIDTH) {
		arr[i] = res[i];
	}
}

__device__ void evo_parent_selection_turnier(struct instance * const inst,
		                             struct memory   * const mem,
					     curandState* rnd_state,
					     const uint8_t q)
{
	if(threadIdx.y != 0 && threadIdx.x >= PARENTS)
		return;

	__shared__ double res[2 * PARENTS];

	double* const arr = mem->c_rat;
	uint32_t idx = curand(rnd_state) % (PARENTS * CHILDS);

	for(uint8_t t = 0; t < q; t++) {
		uint32_t opponent = curand(rnd_state) % (PARENTS * CHILDS);

		if(arr[opponent * 2] < arr[idx * 2])
			idx = opponent;
	}

	res[2 * threadIdx.x]     = arr[idx * 2];
	res[2 * threadIdx.x + 1] = arr[idx * 2 + 1];

	__syncthreads();

	if(threadIdx.x == 0) {
		/* sort entries */
		const uint8_t elems = 2 * PARENTS;
		double key, child;

		/* insertion sort */
		for(int8_t i = 2; i < elems; i+=2) {
			key   = res[i];
			child = res[i+1];

			int8_t j = i - 2;
			while(j >=0 && res[j] > key) {
				res[j + 2] = res[j];
				res[j + 3] = res[j+1];
				j = j - 2;
			}
			res[j + 2] = key;
			res[j + 3] = child;
		}

		#ifdef DEBUG
		if(res[0] == 0.0) {
			inst->res_parent = res[1];

			const int off  = inst->width_per_inst;
			const int off2 = res[1] * inst->width_per_inst;
			for(uint8_t row = 0; row < MATRIX_HEIGHT; row++) {
				for(int i = 0; i < inst->width_per_inst; i++) {
					R_ROW(row)[off + i] =
							C_ROW(row)[mem->c_zero + off2 + i];
				}
			}

		}
		#endif
	}
	__syncthreads();
	arr[2 * threadIdx.x]     = res[2 * threadIdx.x];
	arr[2 * threadIdx.x + 1] = res[2 * threadIdx.x + 1];
}


