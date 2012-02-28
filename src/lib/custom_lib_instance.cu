/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#include <math.h>
#include <float.h>

extern "C" {
#include "matrix_generator.h"
}

#include "custom/c_instance.h"
#include "mat_lib_info.h"
#include "ya_malloc.h"

static void set_eps(struct c_instance* inst)
{
	int max_len = 0;
	int cur_len = 0;

	for(size_t i = 0; i < inst->rules_len; i++) {
		if(inst->rules[i] < MUL_SPECIAL) {
			max_len = max(max_len, cur_len);
			cur_len = 0;
		} else {
			cur_len++;
		}
	}

	inst->eps = max(powf(inst->delta, (float)max_len), FLT_EPSILON);
}

int c_create_instance(const int         matrix_width,
		      const int         icount,
		      const int * const rules,
		      const size_t      rules_len)
{
	if(matrix_width < 2 ||
	   matrix_width > MATRIX_WIDTH) {
		return E_INVALID_MAT_WIDTH;
	}

	int free_inst;
	struct c_info_t* const c_info = c_get_empty(&free_inst);
	struct c_info_t* const def_info = c_get_default();
	if(c_info == NULL)
		return E_NO_FREE_INST;

	struct c_instance* inst = c_info->inst;
	struct c_instance* def_inst = def_info->inst;
	inst->match       = def_inst->match;
	inst->cond_left   = def_inst->cond_left;
	inst->cond_right  = def_inst->cond_right;
	inst->delta       = def_inst->delta;
	inst->parent_max  = def_inst->parent_max;
	inst->rules_len   = rules_len;
	inst->rules       = (int*)ya_malloc(sizeof(int) * rules_len);
	memcpy(inst->rules, rules, sizeof(int) * rules_len);

	inst->icount = icount;
	int rules_count = get_rules_count(rules, rules_len);
	if(rules_count < 0)
		return rules_count;
	inst->rules_count = rules_count;
	set_eps(inst);

	c_inst_init(*inst, matrix_width);
	return free_inst;
}


int c_destroy_instance(const int instance)
{
	struct c_info_t* c_info = c_get(instance);
	if(c_info == NULL)
		return E_INVALID_INST;

	c_inst_cleanup(*c_info->inst);
	free(c_info->inst->rules);
	c_info->is_initialized = 0;

	return 0;
}

int c_set_params(int instance,
		 float max, float delta, int match,
		 int cond_left, int cond_right)
{
	struct c_info_t* info = c_get(instance);
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

int c_set_def_params(float max, float delta, int match,
		     int cond_left, int cond_right)
{
	return c_set_params(DEF_INST, max, delta, match, cond_left, cond_right);
}



