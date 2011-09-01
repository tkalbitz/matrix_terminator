/*
 * gnuplot_log.c
 *
 *  Created on: Aug 6, 2011
 *      Author: tkalbitz
 */

#include <time.h>
#include <stdlib.h>
#include <float.h>

#include "config.h"
#include "ya_malloc.h"
#include "plot_log.h"

static char* plot_preamble_txt = "#!/usr/bin/gnuplot\n\
set grid\n\
set title \"Fittness ueber Generationen\"\n\
set xlabel \"Generation\"\n\
set ylabel \"Fittness\"\n\
set xtics nomirror rotate by -45\n\
set ytics border in scale 0,0 mirror norotate offset character 0, 0, 0\n\
set border 3\n\
set grid y linestyle 4\n\
set key below\n\
set log y\n\
\n\
#set terminal png size 1024,768\n\
set terminal pdf monochrome dashed\n\
set output '%s.pdf'\n\
\n\
plot ";

static char* plot_txt = "\"%s\" using 1:($%d) title \"Block %d\" with lines";

static void write_plot_tmpl(struct plot_log* const pl)
{
	fprintf(pl->plot, plot_preamble_txt, pl->dat_name);

	const int end = pl->best ? 1 : BLOCKS;

	for(int i = 0; i < end; i++) {
		fprintf(pl->plot, plot_txt, pl->dat_name, i + 2, i + 1);

		if(i < end - 1) {
			fprintf(pl->plot, ",\\\n");
		}
	}
}

struct plot_log* init_plot_log(char activate, char best_only)
{
	if(!activate)
		return NULL;

	const int dat_len  = 26;
	const int plot_len = 27;

	struct plot_log *pl = (struct plot_log*)ya_malloc(sizeof(*pl));
	pl->dat_name  = (char*)ya_malloc(dat_len);
	pl->plot_name = (char*)ya_malloc(plot_len);
	pl->best = (best_only > 0) ? 1 : 0;

	time_t tmp_time = time(NULL);
	struct tm* tm = localtime(&tmp_time);
	strftime(pl->dat_name,  dat_len, "%F-%H-%M-%S.dat",  tm);
	strftime(pl->plot_name, dat_len, "%F-%H-%M-%S.plot", tm);

	pl->dat  = fopen(pl->dat_name,  "wb");
	pl->plot = fopen(pl->plot_name, "wb");

	if(pl->dat == NULL || pl->plot == NULL) {
		perror("plot-log (fopen) failed");
		abort();
	}

	write_plot_tmpl(pl);

	fflush(pl->plot);
	fclose(pl->plot);

	return pl;
}

void clean_plot_log(struct plot_log* const pl)
{
	if(pl == NULL)
		return;

	fflush(pl->dat);
	fclose(pl->dat);

	free(pl->dat_name);
	free(pl->plot_name);

	free(pl);
}

void plot_log(struct plot_log* const pl,
		  const int round,
		  const double* const rating)
{
	if(pl == NULL)
		return;

	const int width = PARENTS * BLOCKS;

	fprintf(pl->dat, "%10d\t", round);

	if(pl->best) {
		double tmp = FLT_MAX;
		for(int i = 0; i < width; i += PARENTS) {
			tmp = min(tmp, rating[i]);
		}

		fprintf(pl->dat, "%10.9e\t", tmp);
	} else {
		for(int i = 0; i < width; i += PARENTS) {
			fprintf(pl->dat, "%10.9e\t", rating[i]);
		}
	}

	fprintf(pl->dat, "\n");
}


