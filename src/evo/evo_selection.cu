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

	for(int i = tx; i < PARENTS * CHILDS; i += inst->dim.matrix_width) {
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

	for(int i = tx; i < PARENTS * CHILDS; i += inst->dim.matrix_width) {
		arr[i] = res[i];
	}
}

__device__ void evo_parent_selection_turnier(struct instance * const inst,
		                             struct memory   * const mem,
					     curandState* rnd_state,
					     const uint8_t q)
{
	__shared__ struct double2 src[PARENTS * CHILDS];
	__shared__ struct double2 dest[PARENTS];
	double2* const arr = (double2*)mem->c_rat;

	if(threadIdx.y == 0 && threadIdx.x < PARENTS) {
		for(int i = tx; i < PARENTS * CHILDS; i += inst->dim.matrix_width) {
			src[i]   = arr[i];
		}
	}

	__syncthreads();

	if(threadIdx.y == 0 && threadIdx.x < PARENTS) {
		for(int pos = tx; pos < PARENTS; pos += inst->dim.matrix_width) {
			uint32_t idx = curand(rnd_state) % (PARENTS * CHILDS);
			for(uint8_t t = 0; t < q; t++) {
				uint32_t opponent = curand(rnd_state) % (PARENTS * CHILDS);

				if(src[opponent].x < src[idx].x)
					idx = opponent;
			}

			dest[pos] = src[idx];
		}
	}

	__syncthreads();

	if(threadIdx.x == 0) {
		double2 key;

		/* insertion sort */
		for(int i = 1; i < PARENTS; i++) {
			key = dest[i];

			int j = i - 1;
			while(j >=0 && dest[j].x > key.x) {
				dest[j + 1] = dest[j];
				j = j - 1;
			}
			dest[j + 1] = key;
		}

	}

	__syncthreads();

	if(threadIdx.y == 0 && threadIdx.x < PARENTS) {
		for(int i = tx; i < PARENTS; i += inst->dim.matrix_width) {
			arr[i] = dest[i];
		}
	}
}

__device__ void evo_parent_selection_convergence_prevention(
					     struct instance * const inst,
		                             struct memory   * const mem,
					     curandState* rnd_state,
					     const float cp)
{
	if(ty != 0)
		return;

	__shared__ struct double2 res[PARENTS * CHILDS];

	double2* const arr   = (double2*)mem->c_rat;

	for(int i = tx; i < PARENTS * CHILDS; i += inst->dim.matrix_width) {
		res[i]   = arr[i];
	}

	__syncthreads();

	double2 key;

	for(int k = 64; k < NEXT_2POW; k *= 2) {
		for(int p = k * tx; p < PARENTS * CHILDS; p += k * inst->dim.matrix_width) {
			const int end =  min(k * (tx + 1), PARENTS * CHILDS);

			/* insertion sort */
			for(int i = p + 1; i < end; i++) {
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
	}

	if(tx == 0) {
//		double2 key;
//
//		/* insertion sort */
//		for(int i = 1; i < PARENTS * CHILDS; i++) {
//			key = res[i];
//
//			int j = i - 1;
//			while(j >=0 && res[j].x > key.x) {
//				res[j + 1] = res[j];
//				j = j - 1;
//			}
//			res[j + 1] = key;
//		}
//
		int last = 0;
		for(int i = 1; i < PARENTS * CHILDS; i++) {
			if(res[last].x == res[i].x) {
				if(curand_normal(rnd_state) < cp) {
					res[i].x = FLT_MAX;
				}
			} else {
				last = i;
			}
		}

		/* insertion sort */
		for(int i = 1; i < PARENTS * CHILDS; i++) {
			key = res[i];

			/*
			 * we need only parents count and know that
			 * the array was already sorted
			 */
			if(i > PARENTS && res[PARENTS - 1].x < FLT_MAX)
				break;

			int j = i - 1;
			while(j >=0 && res[j].x > key.x) {
				res[j + 1] = res[j];
				j = j - 1;
			}
			res[j + 1] = key;
		}
	}
	__syncthreads();

	for(int i = tx; i < PARENTS * CHILDS; i += inst->dim.matrix_width) {
		arr[i] = res[i];
	}
}
