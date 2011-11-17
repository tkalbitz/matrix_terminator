/*
 * pso_matrix_generator.cu
 *
 *  Created on: Sep 28, 2011
 *      Author: tkalbitz
 */

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <getopt.h>
#include <ctype.h>
#include <errno.h>

#include <sys/wait.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "pso_config.h"
#include "pso_instance.h"

#include "pso.h"
#include "pso_rating.h"
#include "pso_setup.h"

#include "pso_print.h"
#include "pso_copy.h"
#include "ya_malloc.h"
//#include "plot_log.h"

struct matrix_option {
	int      matrix_width;
	double   w;
	double   c1;
	double   c2;
	uint32_t rounds;
	char enable_maxima;
	char plot_log_enable;
	char plot_log_best;
};

static void print_usage()
{
	printf("Usage: matrix_generator [options] rules\n\n");
	printf("  -m|--match any|all -- default: any\n");
	printf("	`- any -- termination contest rules\n");
	printf("	   all -- ICFP rules\n\n");
	printf("  -l|--left-cond  uleft|uright|uleft_lright -- default: uleft_lright\n");
	printf("	`- uleft        -- upper left  has to be >= 1\n");
	printf("	   uright       -- upper right has to be >= 1\n");
	printf("	   uleft_lright -- upper left and lower right has to be >= 1\n\n");
	printf("  -r|--right-cond uleft|uright|uleft_lright -- default: uright\n");
	printf("	`- uleft        -- upper left  has to be >= 1\n");
	printf("	   uright       -- upper right has to be >= 1\n");
	printf("	   uleft_lright -- upper left and lower right has to be >= 1\n\n");
	printf("  -d|--delta  <float number>              -- default: 0.1\n");
	printf("  -c|--rounds <number>                    -- default: 500\n\n");
	printf("  -p|--parent-max         <float number>  -- default: %.2f\n",   PARENT_MAX);
	printf("  -w|--matrix-width       <2 - %d>        -- default: 5\n",      MATRIX_WIDTH);
	printf("  -g|--plot-log\n");
	printf("	`- best         -- log only the best rating\n");
	printf("	   all          -- log all ratings 1\n");
	printf("  -x|--enable-maxima\n\n");
	printf("Rules should be supplied in the form:\n");
	printf("  X10X01X110X0011X or XbaXabXbbaXaabbX\n");
	printf("  |<--->|<------>|    |<--->|<------>|\n");
	printf("   first  second  rule  first  second\n\n");
	printf("  Meaning: BA < AB and BBA < AABB in this case A and B\n");
	printf("           are matrices of dimension (n,n). Parameter n is\n");
	printf("           supplied at compile time and is %d\n\n", MATRIX_WIDTH);
	printf("If the option --plot-log is given all ratings will be written in"
		" a '.dat' file plus a '.plot' file for gnuplot.\n\n");
	printf("If the option --enable-maxima is given the result will be written"
		" in an 'mg_XXXXXX' file and maxima recalculate and prints the "
		"result.\n\n");
}

static void parse_rules(struct pso_instance * const inst, const char *rules)
{
	inst->rules_count = 0;
	inst->rules_len  = strlen(rules);
	inst->rules = (int*)ya_malloc(sizeof(int) * inst->rules_len);

	uint8_t tmp = 0;
	for(int i = 0; i < inst->rules_len; i++) {
		if(rules[i] >= 'a')
			inst->rules[i] = (rules[i] == 'X') ? MUL_SEP : rules[i] - 'a';
		else
			inst->rules[i] = (rules[i] == 'X') ? MUL_SEP : rules[i] - '0';

		if(rules[i] == 'X') {
			tmp = (tmp + 1) % 2;
			if(!tmp) {
				inst->rules_count++;
			}
		}
	}
}

static void parse_configuration(struct pso_instance* const inst,
				struct matrix_option* const mopt,
				int argc, char** argv)
{
	int c;
	int idx;

	inst->match       = MATCH_ANY;
	inst->cond_left   = COND_UPPER_LEFT_LOWER_RIGHT;
	inst->cond_right  = COND_UPPER_RIGHT;
	inst->delta       = 0.1;
	inst->parent_max  = PARENT_MAX;

	mopt->rounds          = 500;
	mopt->enable_maxima   = 0;
	mopt->plot_log_enable = 0;
	mopt->matrix_width    = 5;
	mopt->w               = 0.7298;
//	mopt->c1              = 0.2;
//	mopt->c2              = 0.2;
	mopt->c1              = 2.05;
	mopt->c2              = 2.05;

	struct option opt[] =
	{
		{"match"             , required_argument, 0, 'm'},
		{"left-cond"         , required_argument, 0, 'l'},
		{"right-cond"        , required_argument, 0, 'r'},
		{"rounds"            , required_argument, 0, 'c'},
		{"delta"             , required_argument, 0, 'd'},
		{"help"              , no_argument,       0, 'h'},
		{"parent-max"        , required_argument, 0, 'p'},
		{"enable-maxima"     , no_argument,       0, 'x'},
		{"plot-log"          , required_argument, 0, 'g'},
		{"matrix-width"      , required_argument, 0, 'w'},
		{0, 0, 0, 0}
	};

	while((c = getopt_long(argc, argv, "m:l:r:c:d:hp:xg:w:",
			      opt, &idx)) != EOF) {
		switch(c) {
		case 'm':
			if(!strcmp(optarg, "all"))
				inst->match = MATCH_ALL;
			else if(!strcmp(optarg, "any"))
				inst->match = MATCH_ANY;
			else {
				print_usage();
				exit(EXIT_FAILURE);
			}
			break;
		case 'l':
			if(!strcmp(optarg, "uleft"))
				inst->cond_left = COND_UPPER_LEFT;
			else if(!strcmp(optarg, "uright"))
				inst->cond_left = COND_UPPER_RIGHT;
			else if(!strcmp(optarg, "uleft_lright"))
				inst->cond_left = COND_UPPER_LEFT_LOWER_RIGHT;
			else {
				print_usage();
				exit(EXIT_FAILURE);
			}

			break;
		case 'r':
			if(!strcmp(optarg, "uleft"))
				inst->cond_right = COND_UPPER_LEFT;
			else if(!strcmp(optarg, "uright"))
				inst->cond_right = COND_UPPER_RIGHT;
			else if(!strcmp(optarg, "uleft_lright"))
				inst->cond_right = COND_UPPER_LEFT_LOWER_RIGHT;
			else {
				print_usage();
				exit(EXIT_FAILURE);
			}

			break;
		case 'c':
			mopt->rounds = strtoul(optarg, NULL, 10);
			break;
		case 'd':
			inst->delta = strtod(optarg, NULL);
			break;
		case 'w':
			mopt->matrix_width = (int)strtod(optarg, NULL);
			if(mopt->matrix_width < 2 ||
			   mopt->matrix_width > MATRIX_WIDTH) {
				printf("matrix width was to small or to big!\n");
				print_usage();
				exit(EXIT_FAILURE);
			}
			break;
		case 'h':
			print_usage();
			exit(EXIT_FAILURE);
		case 'p':
			inst->parent_max = strtod(optarg, NULL);
			break;
		case 'x':
			mopt->enable_maxima = 1;
			break;
		case 'g': {
			mopt->plot_log_enable = 1;

			if(!strcmp(optarg, "all"))
				mopt->plot_log_best = 0;
			else if(!strcmp(optarg, "best"))
				mopt->plot_log_best = 1;
			else {
				print_usage();
				exit(EXIT_FAILURE);
			}
			break;
		}
		case '?':
			switch (optopt) {
			case 'm':
			case 'l':
			case 'r':
			case 'c':
			case 'd':
			case 'p':
			case 'g':
			case 'w':
				fprintf(stderr, "Option -%c requires an "
						"argument!\n", optopt);
				exit(EXIT_FAILURE);
				break;
			default:
				if (isprint(optopt)) {
					fprintf(stderr, "Unknown option "
							"character `0x%X\'!\n",
							optopt);
				}
				exit(EXIT_FAILURE);
				break;
			}
			break;

		default:
			printf("\n");
			print_usage();
			exit(EXIT_FAILURE);
		}
	}

	if(optind == argc) {
		printf("Rules are missing!\n\n");
		print_usage();
		exit(EXIT_FAILURE);
	}

	parse_rules(inst, argv[optind]);
}



int main(int argc, char** argv)
{
	struct pso_instance inst;
	struct matrix_option mopt;
	struct pso_instance *dev_inst;
	int* dev_rules;

	parse_configuration(&inst, &mopt, argc, argv);

	pso_inst_init(&inst, mopt.matrix_width);
	dev_inst = pso_inst_create_dev_inst(&inst, &dev_rules);
//	int evo_threads = get_evo_threads(&inst);
//
	const dim3 blocks(BLOCKS, inst.dim.particles);
	const dim3 threads(inst.dim.matrix_width, inst.dim.matrix_height);
	const dim3 setup_threads(inst.dim.matrix_width * inst.dim.matrix_height);
//
	setup_particle_kernel<<<BLOCKS, setup_threads>>>(dev_inst, false);
	cudaThreadSynchronize();
	CUDA_CALL(cudaGetLastError());

	setup_param<<<BLOCKS, inst.dim.particles>>>(dev_inst,
			mopt.w, mopt.c1, mopt.c2, false);
	cudaThreadSynchronize();
	CUDA_CALL(cudaGetLastError());

	setup_rating<<<BLOCKS, PARTICLE_COUNT>>>(dev_inst);
	CUDA_CALL(cudaGetLastError());

	// Prepare
	cudaEvent_t start, stop;
	float elapsedTime;
	float elapsedTimeTotal = 0.f;

	double * const rating = (double*)ya_malloc(BLOCKS * sizeof(double));
//	struct plot_log* pl = init_plot_log(mopt.plot_log_enable,
//					    mopt.plot_log_best);

	int rounds = -1;
	int block = 0; int thread = 0;

	pso_calc_res<<<blocks, threads>>>(dev_inst);
	CUDA_CALL(cudaGetLastError());
	cudaThreadSynchronize();
	CUDA_CALL(cudaGetLastError());

	pso_evaluation_lbest<<<blocks, threads>>>(dev_inst);
	CUDA_CALL(cudaGetLastError());
	cudaThreadSynchronize();
	CUDA_CALL(cudaGetLastError());

	pso_evaluation_gbest<<<BLOCKS, threads>>>(dev_inst);
	CUDA_CALL(cudaGetLastError());
	cudaThreadSynchronize();
	CUDA_CALL(cudaGetLastError());

	for(unsigned long i = 0; i < mopt.rounds; i++) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		// Start record
		cudaEventRecord(start, 0);

		pso_swarm_step_ccpso<<<blocks, threads>>>(dev_inst);
		CUDA_CALL(cudaGetLastError());
		cudaThreadSynchronize();
		CUDA_CALL(cudaGetLastError());

		pso_calc_res<<<blocks, threads>>>(dev_inst);
		CUDA_CALL(cudaGetLastError());
		cudaThreadSynchronize();
		CUDA_CALL(cudaGetLastError());

		pso_evaluation_lbest<<<blocks, threads>>>(dev_inst);
		CUDA_CALL(cudaGetLastError());
		cudaThreadSynchronize();
		CUDA_CALL(cudaGetLastError());

		pso_evaluation_gbest<<<BLOCKS, threads>>>(dev_inst);
		CUDA_CALL(cudaGetLastError());
		cudaThreadSynchronize();
		CUDA_CALL(cudaGetLastError());

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
		elapsedTimeTotal += elapsedTime;

		// Clean up:
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		if(i % 1000 == 0)
			print_gbest_particle_ratings(&inst);
		copy_gb_rating_dev_to_host(&inst, rating);
//		plot_log(pl, i, rating);

		for(int j = 0; j < BLOCKS; j++) {
			if(rating[j] == 0.) {
				block = j;
				rounds = i;
				i = mopt.rounds;
				break;
			}
		}
	}

	free(rating);
//	clean_plot_log(pl);
	pso_inst_copy_dev_to_host(dev_inst, &inst);

//	print_sparam(&inst);
//	print_parent_ratings(&inst);

	printf("Time needed: %f\n", elapsedTimeTotal);
	printf("Needed rounds: %d\n", rounds);
	printf("Result is block: %d, parent: %d\n", block, thread);
	printf("Result was in block: %d, child: %d, selection: %d\n",
		inst.res_child_block, inst.res_child_idx, inst.res_parent);

	print_particle_ratings(&inst);
	print_gbest_particle_ratings(&inst);
	print_global_matrix_pretty(stdout, &inst, block);
	print_rules(stdout, &inst);
//	print_parents(&inst, &mopt, block, thread, rounds);

	printf("Clean up and exit.\n");
	pso_inst_cleanup(&inst, dev_inst);
	cudaFree(dev_rules);
	cudaThreadExit();

	if(rounds == -1)
		return 0;

	return 1;
}
