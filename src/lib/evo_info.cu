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

static struct evo_info_t evo_info[INFO_LEN];

void evo_lib_init()
{
	for(int i = 0; i < INFO_LEN; i++) {
		evo_info[i].is_initialized = 0;
		evo_info[i].inst = (struct instance*)malloc(sizeof(struct instance));
	}
}

void evo_lib_destroy()
{
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
	if(instance < 0 || instance >= INFO_LEN ||
			evo_info[instance].is_initialized == 0)
		return NULL;

	return &evo_info[instance];
}



