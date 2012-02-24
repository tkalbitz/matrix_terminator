/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
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
