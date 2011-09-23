/*
 * evo_instance.cu
 *
 *  Created on: Sep 23, 2011
 *      Author: tkalbitz
 */

extern "C" {
#include "matrix_generator.h"
}

#include "instance.h"
#include "evo_info.h"
#include "ya_malloc.h"

static int evo_get_rules_count(const int * const rules,
			       const size_t      rules_len)
{
	uint8_t tmp = 0;
	int rules_count = 0;
	for(size_t i = 0; i < rules_len; i++) {
		if(rules[i] == MUL_SEP) {
			tmp = (tmp + 1) % 2;
			if(!tmp) {
				rules_count++;
			}
		}
	}

	if(rules[rules_len - 1] != MUL_SEP || tmp != 1)
		return E_RULES_FORMAT_WRONG;

	return rules_count;
}

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
	if(evo_info == NULL)
		return E_NO_FREE_INST;

	evo_info->mut_rate    = MUT_RATE;
	evo_info->recomb_rate = RECOMB_RATE;
	evo_info->sparam      = SPARAM;
	struct instance* inst = evo_info->inst;

	inst->match       = MATCH_ANY;
	inst->cond_left   = COND_UPPER_LEFT_LOWER_RIGHT;
	inst->cond_right  = COND_UPPER_RIGHT;
	inst->delta       = 0.1;
	inst->parent_max  = 10;
	inst->rules_len   = rules_len;
	inst->rules       = (int*)ya_malloc(sizeof(int) * rules_len);
	memcpy(inst->rules, rules, sizeof(int) * rules_len);

	int rules_count = evo_get_rules_count(rules, rules_len);
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

int evo_set_matrix_max_value(const int instance, const double max)
{
	struct evo_info_t* info = evo_get(instance);
	if(info == NULL)
		return E_INVALID_INST;

	if(max <= 0)
		return E_INVALID_VALUE;

	info->inst->parent_max = max;
	return 0;
}

int evo_set_delta_value(const int instance, const double delta)
{
	struct evo_info_t* info = evo_get(instance);
	if(info == NULL)
		return E_INVALID_INST;

	if(delta <= 0)
		return E_INVALID_VALUE;

	info->inst->delta = delta;
	return 0;
}


