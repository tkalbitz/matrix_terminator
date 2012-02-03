/*
 * pso_instance.cu
 *
 *  Created on: Sep 23, 2011
 *      Author: tkalbitz
 */

extern "C" {
#include "matrix_generator.h"
}

#include "pso_instance.h"
#include "mat_lib_info.h"
#include "ya_malloc.h"

int pso_create_instance(const int         matrix_width,
		        const int * const rules,
		        const size_t      rules_len)
{
	if(matrix_width < 2 ||
	   matrix_width > MATRIX_WIDTH) {
		return E_INVALID_MAT_WIDTH;
	}

	int free_inst;
	struct pso_info_t* const pso_info = pso_get_empty(&free_inst);
	struct pso_info_t* const def_info = pso_get_default();
	if(pso_info == NULL)
		return E_NO_FREE_INST;

	struct pso_instance* inst = pso_info->inst;
	struct pso_instance* def_inst = def_info->inst;
	inst->match       = def_inst->match;
	inst->cond_left   = def_inst->cond_left;
	inst->cond_right  = def_inst->cond_right;
	inst->delta       = def_inst->delta;
	inst->parent_max  = def_inst->parent_max;
	inst->rules_len   = rules_len;
	inst->rules       = (int*)ya_malloc(sizeof(int) * rules_len);
	memcpy(inst->rules, rules, sizeof(int) * rules_len);

	int rules_count = get_rules_count(rules, rules_len);
	if(rules_count < 0)
		return rules_count;
	inst->rules_count = rules_count;

	pso_inst_init(inst, matrix_width);
	return free_inst;
}


int pso_destroy_instance(const int instance)
{
	struct pso_info_t* pso_info = pso_get(instance);
	if(pso_info == NULL)
		return E_INVALID_INST;

	pso_inst_cleanup(pso_info->inst, NULL);
	free(pso_info->inst->rules);
	pso_info->is_initialized = 0;

	return 0;
}

int pso_set_params(int instance,
		   double max, double delta, int match,
		   int cond_left, int cond_right)
{
	struct pso_info_t* info = pso_get(instance);
	if(info == NULL)
		return E_INVALID_INST;

	if(max == 0)
		return E_INVALID_VALUE;
	if(delta == 0)
		return E_INVALID_VALUE;
	if(match >= 0 && match != MATCH_ALL && match != MATCH_ANY)
		return E_INVALID_VALUE;

	if(cond_left >= 0 && cond_left != COND_UPPER_LEFT &&
	   cond_left != COND_UPPER_RIGHT &&
	   cond_left != COND_UPPER_LEFT_LOWER_RIGHT)
		return E_INVALID_VALUE;

	if(cond_right >= 0 && cond_right != COND_UPPER_LEFT &&
	   cond_right != COND_UPPER_RIGHT &&
	   cond_right != COND_UPPER_LEFT_LOWER_RIGHT)
		return E_INVALID_VALUE;


	if(max > 0)
		info->inst->parent_max = max;
	if(delta > 0)
		info->inst->delta = delta;
	if(match >= 0)
		info->inst->match = match;
	if(cond_left >= 0)
		info->inst->cond_left = cond_left;
	if(cond_right >= 0)
		info->inst->cond_right = cond_right;

	return 0;
}

int pso_set_def_params(double max, double delta, int match,
		       int cond_left, int cond_right)
{
	return pso_set_params(DEF_INST, max, delta, match, cond_left,
				cond_right);
}
