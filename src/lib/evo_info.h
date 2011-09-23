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

extern "C" {
	void evo_lib_init();
	void evo_lib_destroy();
};

struct evo_info_t* evo_get_empty(uint32_t* instance);
struct evo_info_t* evo_get(uint32_t instance);

#endif /* EVO_INFO_H_ */
