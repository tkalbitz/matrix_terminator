/*
 * evo_info.h
 *
 *  Created on: Sep 23, 2011
 *      Author: tkalbitz
 */

#ifndef EVO_INFO_H_
#define EVO_INFO_H_

#include <stdint.h>

struct evo_info_t
{
	char is_initialized;
	struct instance* inst;

	double mut_rate;
	double recomb_rate;
	double sparam;
};

struct pso_info_t
{
	char is_initialized;
	struct pso_instance* inst;
};

extern "C"
{
	void mat_lib_init();
	void mat_lib_destroy();
};

#define DEF_INST -1

struct evo_info_t* evo_get_empty(int* const instance);
struct evo_info_t* evo_get(const int instance);
struct evo_info_t* evo_get_default();

struct pso_info_t* pso_get_empty(int* const instance);
struct pso_info_t* pso_get(const int instance);
struct pso_info_t* pso_get_default();

int get_rules_count(const int * const rules,
		    const size_t      rules_len);

#endif /* EVO_INFO_H_ */
