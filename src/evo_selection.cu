/*
 * evo_selection.cu
 *
 *  Created on: Jun 24, 2011
 *      Author: tkalbitz
 */

__device__ void evo_parent_selection_best(struct instance * const inst,
					  struct memory   * const mem)
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

__device__ void evo_parent_selection_turnier(struct instance * const inst,
		                             struct memory   * const mem,
					     curandState* rnd_state,
					     const int q)
{
	if(threadIdx.x >= PARENTS)
		return;

	double* const arr = mem->c_rat;
	int idx = curand(rnd_state) % (PARENTS * CHILDS);

	for(int t = 0; t < q; t++) {
		int opponent = curand(rnd_state) % (PARENTS * CHILDS);

		if(arr[opponent * 2] < arr[idx * 2])
			idx = opponent;
	}

	const double rating = arr[idx * 2];
	__syncthreads();
	arr[2 * threadIdx.x] = rating;
	arr[2 * threadIdx.x + 1] = idx;
	__syncthreads();

	if(threadIdx.x != 0)
		return;

	/* sort entries */
	const int elems = 2 * PARENTS;
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


