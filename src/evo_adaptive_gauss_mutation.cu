__device__ void evo_ensure_constraints(struct instance * const inst,
				       struct memory   * const mem,
				       curandState     * const rnd_state)
{
	double* row = C_ROW(0);
	const int end   = mem->c_end;

	const int rnd_val = curand(rnd_state) % ((int)PARENT_MAX - 1) + 1;
	int factor = (int)(rnd_val / inst->delta);
	if((factor * inst->delta) < 1.f)
		factor++;

	const double val = factor * inst->delta;

	for(int start = mem->c_zero; start < end; start += MATRIX_WIDTH) {
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
			     curandState     * const rnd_s,
                             double          * const s_param)
{
	*s_param = *s_param * exp(curand_normal(rnd_s) / MATRIX_HEIGHT);
	const int rows = MATRIX_HEIGHT;
	const double delta = inst->delta;
	double tmp;

	#pragma unroll
	for(int r = 0; r < rows; r++) {
		double* const row = C_ROW(r);

		for(int c = mem->c_zero; c < mem->c_end; c++) {

			if(curand_uniform(rnd_s) > MUT_RATE)
				continue;

			tmp = row[c];
			tmp = tmp + (double)(curand_normal(rnd_s) * (*s_param));
			/* we want x * delta, where x is an int */
			tmp = ((unsigned long)(tmp / delta)) * delta;
			tmp = max(tmp, 0.0);
			tmp = min(PARENT_MAX, tmp);

			row[c] = tmp;
		}
	}

	evo_ensure_constraints(inst, mem, rnd_s);
}
