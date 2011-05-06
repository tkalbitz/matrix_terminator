__device__ int ensure_isnt_nan(struct instance *inst,
		    	       struct memory *mem) {
//	return 0;
	for(int r = 0; r < inst->dim.matrix_height; r++) {
		float* row = C_ROW(r);
		for(int c = mem->c_zero; c < mem->c_end; c++) {
			if(isnan(row[c])) {
				cont = 0;
				atomicAdd(&(inst->isnan), 1);
				return 1;
			}
		}
	}

	for(int r = 0; r < inst->dim.matrix_height; r++) {
		float* row = R_ROW(r);
		for(int c = mem->r_zero1; c < mem->r_end2; c++) {
			if(isnan(row[c])) {
				cont = 0;
				atomicAdd(&(inst->isnan), 2);
				return 1;
			}
		}
	}

	for(int r = 0; r < inst->dim.matrix_height; r++) {
		float* row = P_ROW(r);
		for(int c = 0; c < inst->dim.parents * inst->width_per_inst; c++) {
			if(isnan(row[c])) {
				cont = 0;
				atomicAdd(&(inst->isnan), 3);
				return 1;
			}
		}
	}

	return 0;
}

__device__ int ensure_range_child(struct instance *inst,
		    	    	  struct memory *mem) {
//	return 0;

	for(int r = 0; r < inst->dim.matrix_height; r++) {
		float* row = C_ROW(r);
		for(int c = mem->c_zero; c < mem->c_end; c++) {
			if(row[c] > PARENT_MAX) {
				cont = 0;
				atomicAdd(&(inst->isnan), 333);
				return 1;
			}
		}
	}
	return 0;
}

__device__ int ensure_range_parent(struct instance *inst,
		    	    	  struct memory *mem)
{
//	return 0;
	int count = inst->dim.parents * inst->width_per_inst;

	for(int r = 0; r < inst->dim.matrix_height; r++) {
		float* row = P_ROW(r);
		for(int c = 0; c < count; c++) {
			if(row[c] > 10000) {
				cont = 0;
				atomicAdd(&(inst->isnan), 334);
				return 1;
			}
		}
	}
	return 0;
}

__device__ int ensure_correct_copy(struct instance *inst, struct memory *mem,
		        int child, int parent)
{
//	return 0;
	int cstart = child * inst->width_per_inst;
	int pstart = parent * inst->width_per_inst;
	int row  = inst->dim.matrix_height;
	int cols = inst->num_matrices;

	for(int r = 0; r < row; r++) {
		float* prow = P_ROW(r);
		float* crow = C_ROW(r);

		for(int c = 0; c < cols; c++) {
			if(isnan(crow[cstart + c]) || isnan(prow[pstart + c]))
				continue;

			if(crow[cstart + c] != prow[pstart + c]) {
				return 1;
			}
		}
	}

	return 0;
}
