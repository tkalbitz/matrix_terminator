/*
 * c_matrix_generator.cu
 *
 *  Created on: Sep 28, 2011
 *      Author: tkalbitz
 */
#include <getopt.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>

#include <sys/wait.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "custom/c_instance.h"
#include "custom/c_setup.h"
#include "custom/c_rating.h"
#include "custom/c_print.h"

#include "ya_malloc.h"
//#include "plot_log.h"

struct matrix_option {
	int      matrix_dim;
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
	printf("  -w|--matrix-dim       <2 - %d>        -- default: 5\n",        MATRIX_WIDTH);
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

static void parse_rules(struct c_instance& inst, const char *rules)
{
	inst.rules_count = 0;
	inst.rules_len  = strlen(rules);
	inst.rules = (int*)ya_malloc(sizeof(int) * inst.rules_len);

	uint8_t tmp = 0;
	for(size_t i = 0; i < inst.rules_len; i++) {
		if(rules[i] >= 'a')
			inst.rules[i] = (rules[i] == 'X') ? MUL_SEP : rules[i] - 'a';
		else
			inst.rules[i] = (rules[i] == 'X') ? MUL_SEP : rules[i] - '0';

		if(rules[i] == 'X') {
			tmp = (tmp + 1) % 2;
			if(!tmp) {
				inst.rules_count++;
			}
		}
	}
}

static void parse_configuration(struct c_instance&    inst,
				struct matrix_option& mopt,
				int argc, char** argv)
{
	int c;
	int idx;

	inst.match       = MATCH_ANY;
	inst.cond_left   = COND_UPPER_LEFT_LOWER_RIGHT;
	inst.cond_right  = COND_UPPER_RIGHT;
	inst.delta       = 0.1;
	inst.parent_max  = PARENT_MAX;
	inst.icount      = 100;
	inst.scount      = 100;

	mopt.rounds          = 500;
	mopt.enable_maxima   = 0;
	mopt.plot_log_enable = 0;
	mopt.matrix_dim    = 5;

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
		{"matrix-dim"        , required_argument, 0, 'w'},
		{"instances"         , required_argument, 0, 'i'},
		{"search-instances"  , required_argument, 0, 's'},

		{0, 0, 0, 0}
	};

	while((c = getopt_long(argc, argv, "m:l:r:c:d:hp:xg:w:s:i:",
			      opt, &idx)) != EOF) {
		switch(c) {
		case 'm':
			if(!strcmp(optarg, "all"))
				inst.match = MATCH_ALL;
			else if(!strcmp(optarg, "any"))
				inst.match = MATCH_ANY;
			else {
				print_usage();
				exit(EXIT_FAILURE);
			}
			break;
		case 'l':
			if(!strcmp(optarg, "uleft"))
				inst.cond_left = COND_UPPER_LEFT;
			else if(!strcmp(optarg, "uright"))
				inst.cond_left = COND_UPPER_RIGHT;
			else if(!strcmp(optarg, "uleft_lright"))
				inst.cond_left = COND_UPPER_LEFT_LOWER_RIGHT;
			else {
				print_usage();
				exit(EXIT_FAILURE);
			}

			break;
		case 'r':
			if(!strcmp(optarg, "uleft"))
				inst.cond_right = COND_UPPER_LEFT;
			else if(!strcmp(optarg, "uright"))
				inst.cond_right = COND_UPPER_RIGHT;
			else if(!strcmp(optarg, "uleft_lright"))
				inst.cond_right = COND_UPPER_LEFT_LOWER_RIGHT;
			else {
				print_usage();
				exit(EXIT_FAILURE);
			}

			break;
		case 'i':
			inst.icount = strtoul(optarg, NULL, 1000);
			break;
		case 's':
			inst.scount = strtoul(optarg, NULL, 100);
			break;
		case 'c':
			mopt.rounds = strtoul(optarg, NULL, 10);
			break;
		case 'd':
			inst.delta = strtod(optarg, NULL);
			break;
		case 'w':
			mopt.matrix_dim = (int)strtod(optarg, NULL);
			if(mopt.matrix_dim < 2 ||
			   mopt.matrix_dim > MATRIX_WIDTH) {
				printf("matrix width was to small or to big!\n");
				print_usage();
				exit(EXIT_FAILURE);
			}
			break;
		case 'h':
			print_usage();
			exit(EXIT_FAILURE);
		case 'p':
			inst.parent_max = strtod(optarg, NULL);
			break;
		case 'x':
			mopt.enable_maxima = 1;
			break;
		case 'g': {
			mopt.plot_log_enable = 1;

			if(!strcmp(optarg, "all"))
				mopt.plot_log_best = 0;
			else if(!strcmp(optarg, "best"))
				mopt.plot_log_best = 1;
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
			case 's':
			case 'i':
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

void setup_rating(struct c_instance& inst)
{
	dim3 threads(inst.mdim, inst.mdim);
	dim3 blocks(BLOCKS, min(inst.icount, 512));

	int i = 0;
	do{
		setup_rating<<<blocks, threads>>>(inst, i);
		CUDA_CALL(cudaGetLastError());
		i += 512;
	} while((i + 512) <= inst.icount);

	if(blocks.y == 512 && (inst.icount - i) != 0) {
		dim3 rest(BLOCKS, inst.icount - i);
		setup_rating<<<rest, threads>>>(inst, i);
		CUDA_CALL(cudaGetLastError());
	}
}

int main(int argc, char** argv)
{
	struct c_instance inst;
	struct matrix_option mopt;
	struct c_instance *dev_inst;
	struct c_instance host_inst;
	int* dev_rules;
	size_t freeBefore, freeAfter, total;

	srand(time(0));
	parse_configuration(inst, mopt, argc, argv);

	CUDA_CALL(cudaMemGetInfo(&freeBefore, &total));
	c_inst_init(inst, mopt.matrix_dim);
	host_inst = inst;
	dev_inst = c_inst_create_dev_inst(inst, &dev_rules);

	int3* stack;
	unsigned int* top;
	const size_t slen = BLOCKS * inst.rules_count * inst.width_per_matrix;
	CUDA_CALL(cudaMalloc(&stack, 2 * slen * sizeof(*stack)));
	CUDA_CALL(cudaMalloc(&top, BLOCKS * sizeof(*top)));

	dim3 blocks(BLOCKS, inst.scount);
	dim3 threads(inst.mdim, inst.mdim);

	CUDA_CALL(cudaMemGetInfo(&freeAfter, &total));
	printf("Allocated %.2f MiB from %.2f MiB\n",
			(freeBefore - freeAfter) / 1024 / 1024.f,
			total / 1024 / 1024.f);

	setup_instances_kernel<<<1, 320>>>(inst);
	CUDA_CALL(cudaGetLastError());

	setup_best_kernel<<<1, BLOCKS>>>(inst);
	CUDA_CALL(cudaGetLastError());

	setup_rating(inst);

	// Prepare
	cudaEvent_t start, stop;
	float elapsedTime;
	float elapsedTimeTotal = 0.f;

	double* rating   = (double*)ya_malloc(BLOCKS * sizeof(double));
	int* best_idx = (int*)ya_malloc(BLOCKS * sizeof(best_idx));

	int rounds = -1;
	int block = 0; int pos = 0;

	CUDA_CALL(cudaMemcpy(rating, inst.best, BLOCKS * sizeof(*rating), cudaMemcpyDeviceToHost));
	for(int i = 0; i < BLOCKS; i++) {
		printf("%.2e ", rating[i]);
	}
	printf("\n");

	for(unsigned long i = 0; i < mopt.rounds; i++) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		// Start record
		cudaEventRecord(start, 0);

		copy_parent_kernel<<<BLOCKS, 128>>>(inst);
		CUDA_CALL(cudaGetLastError());

		path_mutate_kernel_p1<<<BLOCKS, threads>>>(inst, stack, top);
		CUDA_CALL(cudaGetLastError());

		path_mutate_kernel_p2<<<BLOCKS, 1>>>(inst, stack, top);
		CUDA_CALL(cudaGetLastError());

		mutate_kernel<<<blocks, 128>>>(inst);
		CUDA_CALL(cudaGetLastError());

		rate_mutated_kernel<<<blocks, threads>>>(inst);
		CUDA_CALL(cudaGetLastError());

		reduce_rating_kernel<<<BLOCKS, 512>>>(inst);
		CUDA_CALL(cudaGetLastError());

		copy_to_child_kernel<<<BLOCKS, 192>>>(inst);
		CUDA_CALL(cudaGetLastError());

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
		elapsedTimeTotal += elapsedTime;

		// Clean up:
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		if(i % 1000 == 0) {
			printf("%6d: ", i / 1000);
			CUDA_CALL(cudaMemcpy(rating, inst.best, BLOCKS * sizeof(*rating), cudaMemcpyDeviceToHost));
			for(int j = 0; j < BLOCKS; j++) {
				printf("%.2e ", rating[j]);

				if(rating[j] == 0.) {
					printf("drin!\n");
					rounds = i;
					block = j;
					i = mopt.rounds;
					CUDA_CALL(cudaMemcpy(best_idx, inst.best_idx,
							BLOCKS * sizeof(*best_idx),
							cudaMemcpyDeviceToHost));
					pos = best_idx[j];
					break;
				}
			}
			printf("\n");
		}
	}

	free(rating);
	free(best_idx);
////	clean_plot_log(pl);

	printf("Time needed: %f\n", elapsedTimeTotal);
	printf("Needed rounds: %d\n", rounds);
	printf("Result is block: %d, pos: %d\n", block, pos);

	print_matrix_pretty(stdout, inst, block, pos);
	print_rules(stdout, host_inst);

	printf("Clean up and exit.\n");
	c_inst_cleanup(inst, dev_inst);
	cudaFree(dev_rules);
	cudaThreadExit();

	if(rounds == -1)
		return 0;

	return 1;
}
