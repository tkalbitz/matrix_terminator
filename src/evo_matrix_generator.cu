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

#include "instance.h"

#include "evo.h"
#include "evo_rating.h"
#include "evo_setup.h"

#include "matrix_print.h"
#include "matrix_copy.h"
#include "ya_malloc.h"
#include "plot_log.h"

struct matrix_option {
	double mut_rate;
	double recomb_rate;
	double sparam;
	int    matrix_width;
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
	printf("  -m|--mutation-rate      <float number>  -- default: %3.2f\n",  MUT_RATE);
	printf("  -e|--recombination-rate <float number>  -- default: %3.2f\n",  RECOMB_RATE);
	printf("  -p|--parent-max         <float number>  -- default: %.2f\n",   PARENT_MAX);
	printf("  -s|--strategy-param     <float number>  -- default: %.2f\n",   SPARAM);
	printf("  -w|--matrix-width       <2 - %d>        -- default: 5\n",      MATRIX_WIDTH);
	printf("  -g|--plot-log\n");
	printf("	`- best         -- log only the best rating\n");
	printf("	   all          -- log all ratings 1\n");
	printf("  -x|--enable-maxima\n\n");
	printf("Rules should be supplied in the form:\n");
	printf("  X10X01X110X0011X or XbaXabYbbaXaabbX\n");
	printf("  |<--->|<------>|    |<--->|<------>|\n");
	printf("   first  second  rule  first  second\n\n");
	printf("  Meaning: BA < AB and BBA < AABB in this case A and B\n");
	printf("           are matrices of dimension (n,n). Parameter n is\n");
	printf("           supplied at compile time and is %d\n\n", MATRIX_WIDTH);
	printf("           If Y is supplied instead of X at the beginning of a\n");
	printf("           rule right-cond can but must not apply. Between to\n");
	printf("           rule sides X and Y are interchangeable.\n\n");

	printf("If the option --plot-log is given all ratings will be written in"
		" a '.dat' file plus a '.plot' file for gnuplot.\n\n");
	printf("If the option --enable-maxima is given the result will be written"
		" in an 'mg_XXXXXX' file and maxima recalculate and prints the "
		"result.\n\n");
}

void print_parents(struct instance* const inst,
		   struct matrix_option* const mopt,
		   const int block,
		   const int thread,
		   const int rounds)
{
	FILE* f = stdout;
	char* const fname = strdup("mg_XXXXXX");

	printf("Parents:\n");

	if(mopt->enable_maxima) {
		int fd = mkstemp(fname);
		if(fd == -1) {
			perror("mkstemp: Failed fallback to stdout.");
			f = stdout;
			mopt->enable_maxima = 0;
		}

		f = fdopen(fd, "wb");
		if(f == NULL) {
			perror("fdopen: Failed fallback to stdout.");
			close(fd);
			f = stdout;
			mopt->enable_maxima = 0;
		}
	}

	print_parent_matrix_pretty(f, inst, block, thread);
	print_rules(f, inst);

	if(rounds != -1 && mopt->enable_maxima) {
		fprintf(f, "quit();\n");
		fflush(f);
		fclose(f);

		int r = fork();

		if(r > 0) {
			int status;
			wait(&status);
		} else if(r == 0){
			execlp("maxima", "maxima", "--very-quiet", "-b", fname, NULL);
		}

		if(r == -1) {
			perror("fork failed");
		}
	}

	free(fname);
}

static void parse_rules(struct instance * const inst, const char *rules)
{
	inst->rules_count = 0;
	inst->rules_len  = strlen(rules);
	inst->rules = (int*)ya_malloc(sizeof(int) * inst->rules_len);

	uint8_t tmp = 0;
	for(int i = 0; i < inst->rules_len; i++) {
		switch(rules[i]) {
		case 'X': {
			inst->rules[i] = MUL_SEP;
			break;
		}
		case 'Y': {
			inst->rules[i] = MUL_MARK;
			break;
		}
		default:
			if(rules[i] >= 'a')
				inst->rules[i] = rules[i] - 'a';
			else
				inst->rules[i] = rules[i] - '0';
			break;
		}

		if(rules[i] == 'X' || rules[i] == 'Y') {
			tmp = (tmp + 1) % 2;
			if(!tmp) {
				inst->rules_count++;
			}
		}
	}
}

static void parse_configuration(struct instance* const inst,
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

	mopt->mut_rate        = MUT_RATE;
	mopt->recomb_rate     = RECOMB_RATE;
	mopt->sparam          = SPARAM;
	mopt->rounds          = 500;
	mopt->enable_maxima   = 0;
	mopt->plot_log_enable = 0;
	mopt->matrix_width    = 5;

	struct option opt[] =
	{
		{"match"             , required_argument, 0, 'm'},
		{"left-cond"         , required_argument, 0, 'l'},
		{"right-cond"        , required_argument, 0, 'r'},
		{"rounds"            , required_argument, 0, 'c'},
		{"delta"             , required_argument, 0, 'd'},
		{"help"              , no_argument,       0, 'h'},
		{"mutation-rate"     , required_argument, 0, 'u'},
		{"recombination-rate", required_argument, 0, 'e'},
		{"parent-max"        , required_argument, 0, 'p'},
		{"strategy-param"    , required_argument, 0, 's'},
		{"enable-maxima"     , no_argument,       0, 'x'},
		{"plot-log"          , required_argument, 0, 'g'},
		{"matrix-width"      , required_argument, 0, 'w'},
		{0, 0, 0, 0}
	};

	while((c = getopt_long(argc, argv, "m:l:r:c:d:hu:e:p:s:xg:w:",
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
		case 'u':
			mopt->mut_rate = strtod(optarg, NULL);
			break;
		case 'e':
			mopt->recomb_rate = strtod(optarg, NULL);
			break;
		case 'p':
			inst->parent_max = strtod(optarg, NULL);
			break;
		case 's':
			mopt->sparam     = strtod(optarg, NULL);
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
			case 'u':
			case 'e':
			case 'p':
			case 's':
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
	/* there is no runtime limit for kernels */
//	CUDA_CALL(cudaSetDevice(0));

	struct instance inst;
	struct matrix_option mopt;
	struct instance *dev_inst;
	int* dev_rules;

	parse_configuration(&inst, &mopt, argc, argv);

	inst_init(&inst, mopt.matrix_width);
	dev_inst = inst_create_dev_inst(&inst, &dev_rules);
	int evo_threads = get_evo_threads(&inst);

	const dim3 blocks(BLOCKS, PARENTS*CHILDS);
	const dim3 threads(inst.dim.matrix_width, inst.dim.matrix_height);
	const dim3 setup_threads(inst.dim.matrix_width * inst.dim.matrix_height);

	setup_childs_kernel<<<BLOCKS, setup_threads>>>(dev_inst, false);
	cudaThreadSynchronize();
	CUDA_CALL(cudaGetLastError());

	setup_sparam<<<BLOCKS, evo_threads>>>(dev_inst,
			mopt.sparam, mopt.mut_rate, mopt.recomb_rate, false);
	cudaThreadSynchronize();
	CUDA_CALL(cudaGetLastError());

	// Prepare
	cudaEvent_t start, stop;
	float elapsedTime;
	float elapsedTimeTotal = 0.f;

	const int width = inst.dim.parents * inst.dim.blocks;
	double * const rating = (double*)ya_malloc(width * sizeof(double));
	struct plot_log* pl = init_plot_log(mopt.plot_log_enable,
					    mopt.plot_log_best);

	int rounds = -1;
	int block = 0; int thread = 0;

	evo_calc_res<<<blocks, threads>>>(dev_inst);
	CUDA_CALL(cudaGetLastError());
	cudaThreadSynchronize();
	CUDA_CALL(cudaGetLastError());

	evo_kernel_part_two<<<BLOCKS, threads>>>(dev_inst);
	CUDA_CALL(cudaGetLastError());
	cudaThreadSynchronize();
	CUDA_CALL(cudaGetLastError());

	for(unsigned long i = 0; i < mopt.rounds; i++) {
		if(i % 300 == 0) {
			setup_childs_kernel<<<BLOCKS, setup_threads>>>(dev_inst, true);
			CUDA_CALL(cudaGetLastError());
			evo_calc_res<<<blocks, threads>>>(dev_inst);
			CUDA_CALL(cudaGetLastError());
			evo_kernel_part_two<<<BLOCKS, threads>>>(dev_inst);
			CUDA_CALL(cudaGetLastError());
			setup_sparam<<<BLOCKS, evo_threads>>>(dev_inst,
					mopt.sparam, mopt.mut_rate, mopt.recomb_rate, true);
			CUDA_CALL(cudaGetLastError());
		}

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		// Start record
		cudaEventRecord(start, 0);

		evo_kernel_part_one<<<BLOCKS, evo_threads>>>(dev_inst);
		CUDA_CALL(cudaGetLastError());
		cudaThreadSynchronize();
		CUDA_CALL(cudaGetLastError());

		evo_calc_res<<<blocks, threads>>>(dev_inst);
		CUDA_CALL(cudaGetLastError());
		cudaThreadSynchronize();
		CUDA_CALL(cudaGetLastError());

		evo_kernel_part_two<<<BLOCKS, threads>>>(dev_inst);
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

		copy_parent_rating_dev_to_host(&inst, rating);
		plot_log(pl, i, rating);

		for(int j = 0; j < width; j += PARENTS) {
			if(rating[j] == 0.) {
				block = j / PARENTS;
				thread = j % PARENTS;
				rounds = i;
				i = mopt.rounds;
				break;
			}
		}
	}

	free(rating);
	clean_plot_log(pl);
	inst_copy_dev_to_host(dev_inst, &inst);

	print_parent_ratings(&inst);

#ifdef DEBUG
	double *crat = (double*)ya_malloc(2 * get_evo_threads(&inst) *
			               inst.dim.blocks * sizeof(double));
	int child = block * inst.dim.parents * inst.dim.childs * 2;
	copy_child_rating_dev_to_host(&inst, crat);
	printf("Output block: %d child: %d\n", block, (int)crat[child + 1]);
	print_debug_pretty(stdout, &inst, block, (int)crat[child + 1]);
	print_child_matrix_pretty(stdout, &inst, block, (int)crat[child + 1]);
	free(crat);
#endif

	print_parents(&inst, &mopt, block, thread, rounds);

	printf("Time needed: %f\n", elapsedTimeTotal);
	printf("Needed rounds: %d\n", rounds);
	printf("Result is block: %d, parent: %d\n", block, thread);
	printf("Result was in block: %d, child: %d, selection: %d\n",
		inst.res_child_block, inst.res_child_idx, inst.res_parent);

	printf("Clean up and exit.\n");
	inst_cleanup(&inst, dev_inst);
	cudaFree(dev_rules);
	cudaThreadExit();

	if(rounds == -1)
		return 0;

	return 1;
}
