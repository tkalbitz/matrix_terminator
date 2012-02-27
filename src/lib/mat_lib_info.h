/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
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

struct c_info_t
{
	char is_initialized;
	struct c_instance* inst;
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

struct c_info_t* c_get_empty(int* const instance);
struct c_info_t* c_get(const int inst);
struct c_info_t* c_get_default();

int get_rules_count(const int * const rules,
		    const size_t      rules_len);

#endif /* EVO_INFO_H_ */
