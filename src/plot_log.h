/*
 * plot_log.h
 *
 *  Created on: Aug 7, 2011
 *      Author: tkalbitz
 */

#ifndef PLOT_LOG_H_
#define PLOT_LOG_H_

#include "instance.h"

struct plot_log
{
	char* dat_name;
	char* plot_name;
	FILE* plot;
	FILE* dat;
	char best;
};

struct plot_log* init_plot_log(char activate, char best_only);
void clean_plot_log(struct plot_log* const pl);
void plot_log(struct plot_log* const pl,
	      const int round,
	      const double* const rating);


#endif /* PLOT_LOG_H_ */
