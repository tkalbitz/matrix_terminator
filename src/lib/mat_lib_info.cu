/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#include <stdlib.h>
#include "evo/instance.h"
#include "ccpso2/pso_instance.h"
#include "ya_malloc.h"

#include "mat_lib_info.h"

extern "C" {
#include "evo_error.h"
}

#define INFO_LEN 16

static struct evo_info_t evo_def_info;
static struct evo_info_t evo_info[INFO_LEN];

static struct pso_info_t pso_def_info;
static struct pso_info_t pso_info[INFO_LEN];

void mat_lib_init()
{
	evo_def_info.mut_rate    = MUT_RATE;
	evo_def_info.recomb_rate = RECOMB_RATE;
	evo_def_info.sparam      = SPARAM;

	evo_def_info.inst = (struct instance*)ya_malloc(sizeof(struct instance));
	pso_def_info.inst = (struct pso_instance*)ya_malloc(sizeof(struct pso_instance));

	struct instance* evo_inst = evo_def_info.inst;
	struct pso_instance* pso_inst = pso_def_info.inst;

	pso_inst->match      = evo_inst->match       = MATCH_ANY;
	pso_inst->cond_left  = evo_inst->cond_left   = COND_UPPER_LEFT_LOWER_RIGHT;
	pso_inst->cond_right = evo_inst->cond_right  = COND_UPPER_RIGHT;
	pso_inst->delta      = evo_inst->delta       = 0.1;
	pso_inst->parent_max = evo_inst->parent_max  = 10;

	for(int i = 0; i < INFO_LEN; i++) {
		evo_info[i].is_initialized = 0;
		pso_info[i].is_initialized = 0;
		evo_info[i].inst = (struct instance*)ya_malloc(sizeof(*evo_info));
		pso_info[i].inst = (struct pso_instance*)ya_malloc(sizeof(*pso_info));
	}

	srand((unsigned int)time(0));
}

void mat_lib_destroy()
{
	free(evo_def_info.inst);

	for(int i = 0; i < INFO_LEN; i++) {
		evo_info[i].is_initialized = 0;
		pso_info[i].is_initialized = 0;
		free(evo_info[i].inst);
		free(pso_info[i].inst);
	}

	cudaThreadExit();
}

struct evo_info_t* evo_get_empty(int* const instance)
{
	int free_inst = -1;
	for(int i = 0; i < INFO_LEN; i++) {
		if(evo_info[i].is_initialized == 0) {
			free_inst = i;
			break;
		}
	}

	if(free_inst == -1)
		return NULL;

	evo_info[free_inst].is_initialized = 1;
	*instance = free_inst;
	return &evo_info[free_inst];
}

struct evo_info_t* evo_get(const int instance)
{
	if(instance < -1 || instance >= INFO_LEN ||
			evo_info[instance].is_initialized == 0) {
		if(instance == DEF_INST)
			return &evo_def_info;
		else
			return NULL;
	}

	return &evo_info[instance];
}

struct evo_info_t* evo_get_default()
{
	return &evo_def_info;
}

struct pso_info_t* pso_get_empty(int* const instance)
{
	int free_inst = -1;
	for(int i = 0; i < INFO_LEN; i++) {
		if(pso_info[i].is_initialized == 0) {
			free_inst = i;
			break;
		}
	}

	if(free_inst == -1)
		return NULL;

	pso_info[free_inst].is_initialized = 1;
	*instance = free_inst;
	return &pso_info[free_inst];
}

struct pso_info_t* pso_get(const int instance)
{
	if(instance < -1 || instance >= INFO_LEN ||
			pso_info[instance].is_initialized == 0) {
		if(instance == DEF_INST)
			return &pso_def_info;
		else
			return NULL;
	}

	return &pso_info[instance];
}

struct pso_info_t* pso_get_default()
{
	return &pso_def_info;
}

int get_rules_count(const int * const rules,
		    const size_t      rules_len)
{
	uint8_t tmp = 0;
	int rules_count = 0;
	for(size_t i = 0; i < rules_len; i++) {
		if(rules[i] == MUL_SEP || rules[i] == MUL_MARK) {
			tmp = (tmp + 1) % 2;
			if(!tmp) {
				rules_count++;
			}
		}
	}

	if((rules[rules_len - 1] != MUL_SEP && rules[rules_len - 1] != MUL_MARK) || tmp != 1)
		return E_RULES_FORMAT_WRONG;

	return rules_count;
}

