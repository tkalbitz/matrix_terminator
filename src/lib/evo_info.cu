/*
 * evo_info.c
 *
 *  Created on: Sep 23, 2011
 *      Author: tkalbitz
 */

#include <stdlib.h>
#include "instance.h"

#include "evo_info.h"

extern "C" {
#include "evo_error.h"
}

#define INFO_LEN 16

static struct evo_info_t evo_def_info;
static struct evo_info_t evo_info[INFO_LEN];

void evo_lib_init()
{
	evo_def_info.mut_rate    = MUT_RATE;
	evo_def_info.recomb_rate = RECOMB_RATE;
	evo_def_info.sparam      = SPARAM;
	evo_def_info.inst = (struct instance*)malloc(sizeof(struct instance));
	struct instance* inst = evo_def_info.inst;

	inst->match       = MATCH_ANY;
	inst->cond_left   = COND_UPPER_LEFT_LOWER_RIGHT;
	inst->cond_right  = COND_UPPER_RIGHT;
	inst->delta       = 0.1;
	inst->parent_max  = 10;

	for(int i = 0; i < INFO_LEN; i++) {
		evo_info[i].is_initialized = 0;
		evo_info[i].inst = (struct instance*)malloc(sizeof(struct instance));
	}
}

void evo_lib_destroy()
{
	free(evo_def_info.inst);

	for(int i = 0; i < INFO_LEN; i++) {
		evo_info[i].is_initialized = 0;
		free(evo_info[i].inst);
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



