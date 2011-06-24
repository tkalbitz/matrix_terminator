/*
 * Select two parents for recombination.
 * Selection is currently complete uniform.
 */
inline __device__ void evo_recomb_selection(const struct instance* const inst,
					    curandState* const rnd_state,
					    int* const sel)
{
	sel[0] = curand(rnd_state) % inst->dim.parents;
	sel[1] = curand(rnd_state) % inst->dim.parents;
}

/* A uniform crossover recombination. */
__device__ void evo_recombination(struct instance * const inst,
				  struct memory   * const mem,
				  curandState     * const rnd_state,
				  const int       * const sel)
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
