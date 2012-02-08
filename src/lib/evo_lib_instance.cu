/*
 * evo_instance.cu
 *
 *  Created on: Sep 23, 2011
 *      Author: tkalbitz
 */

extern "C" {
#include "matrix_generator.h"
}

#include "evo/instance.h"
#include "mat_lib_info.h"
#include "ya_malloc.h"

int evo_create_instance(const int         matrix_width,
		        const int * const rules,
		        const size_t      rules_len)
{
	if(matrix_width < 2 ||
	   matrix_width > MATRIX_WIDTH) {
		return E_INVALID_MAT_WIDTH;
	}

	int free_inst;
	struct evo_info_t* const evo_info = evo_get_empty(&free_inst);
	struct evo_info_t* const def_info = evo_get_default();
	if(evo_info == NULL)
		return E_NO_FREE_INST;

	evo_info->mut_rate    = def_info->mut_rate;
	evo_info->recomb_rate = def_info->recomb_rate;
	evo_info->sparam      = def_info->sparam;

	struct instance* inst = evo_info->inst;
	struct instance* def_inst = def_info->inst;
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

	inst_init(inst, matrix_width);
	return free_inst;
}


int evo_destroy_instance(const int instance)
{
	struct evo_info_t* evo_info = evo_get(instance);
	if(evo_info == NULL)
		return E_INVALID_INST;

	inst_cleanup(evo_info->inst, NULL);
	free(evo_info->inst->rules);
	evo_info->is_initialized = 0;

	return 0;
}

int evo_set_params(int instance,
		   double max, double delta, int match,
		   int cond_left, int cond_right,
		   double mut_rate, double strgy_parm)
{
	struct evo_info_t* info = evo_get(instance);
	if(info == NULL)
		return E_INVALID_INST;

	if(max == 0)
		return E_INVALID_VALUE;
	if(delta == 0)
		return E_INVALID_VALUE;
	if(match >= 0 && match != MATCH_ALL && match != MATCH_ANY)
		return E_INVALID_VALUE;
	if(mut_rate == 0 || mut_rate > 1)
		return E_INVALID_VALUE;
	if(strgy_parm == 0)
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
	if(mut_rate > 0)
		info->mut_rate = mut_rate;
	if(strgy_parm > 0)
		info->sparam = strgy_parm;

	return 0;
}

int evo_set_def_params(double max, double delta, int match,
		       int cond_left, int cond_right,
		       double mut_rate, double strgy_parm)
{
	return evo_set_params(DEF_INST, max, delta, match, cond_left,
				cond_right, mut_rate, strgy_parm);
}



