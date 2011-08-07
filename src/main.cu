#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <getopt.h>
#include <ctype.h>
#include <errno.h>

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
	printf("  -s|--strategy-param     <float number>  -- default: %.2f\n", SPARAM);
	printf("  -g|--plot-log\n");
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

void print_parents(struct instance* const inst,
		   const int block,
		   const int thread,
		   const int rounds) {
	FILE* f = stdout;
	char* const fname = strdup("mg_XXXXXX");

	printf("Parents:\n");

	if(inst->maxima) {
		int fd = mkstemp(fname);
		if(fd == -1) {
			perror("mkstemp: Failed fallback to stdout.");
			f = stdout;
			inst->maxima = 0;
		}

		f = fdopen(fd, "wb");
		if(f == NULL) {
			perror("fdopen: Failed fallback to stdout.");
			close(fd);
			f = stdout;
			inst->maxima = 0;
		}
	}

	print_parent_matrix_pretty(f, inst, block, thread);
	print_rules(f, inst);

	if(rounds != -1 && inst->maxima) {
		fprintf(f, "quit();\n");
		fflush(f);
		fclose(f);

		int r = fork();

		if(r > 0)
			execlp("maxima", "maxima", "--very-quiet", "-b", fname, NULL);

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

static int parse_configuration(struct instance* const inst,
			       int argc, char** argv)
{
	int c;
	int idx;
	unsigned long int rounds = 500;

	inst->match       = MATCH_ANY;
	inst->cond_left   = COND_UPPER_LEFT_LOWER_RIGHT;
	inst->cond_right  = COND_UPPER_RIGHT;
	inst->delta       = 0.1;
	inst->mut_rate    = MUT_RATE;
	inst->recomb_rate = RECOMB_RATE;
	inst->parent_max  = PARENT_MAX;
	inst->def_sparam  = SPARAM;
	inst->maxima      = 0;
	inst->plot_log    = 0;
	
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
		{"plot-log"          , no_argument,       0, 'g'},
		{0, 0, 0, 0}
	};

	while((c = getopt_long(argc, argv, "m:l:r:c:d:hu:e:p:s:xg",
			      opt, &idx)) != EOF) {
		switch(c) {
		case 'm':
			if(!strcmp(optarg, "all"))
				inst->match = MATCH_ALL;
			else if(!strcmp(optarg, "any"))
				inst->match = MATCH_ANY;
			else
				return -1;

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
			rounds = strtoul(optarg, NULL, 10);
			break;
		case 'd':
			inst->delta = strtod(optarg, NULL);
			break;
		case 'h':
			print_usage();
			return 0;
		case 'u':
			inst->mut_rate = strtod(optarg, NULL);
			break;
		case 'e':
			inst->recomb_rate = strtod(optarg, NULL);
			break;
		case 'p':
			inst->parent_max = strtod(optarg, NULL);
			break;
		case 's':
			inst->def_sparam = strtod(optarg, NULL);
			break;
		case 'x':
			inst->maxima = 1;
			break;
		case 'g':
			inst->plot_log = 1;
			break;
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
	return rounds;
}

int main(int argc, char** argv)
{
	/* there is no runtime limit for kernels */
	CUDA_CALL(cudaSetDevice(0));

	struct instance inst;
	struct instance *dev_inst;

	unsigned long max_rounds = 500;

	max_rounds = parse_configuration(&inst, argc, argv);
	if(max_rounds == 0)
		return 1;

	inst_init(&inst);
	dev_inst = inst_create_dev_inst(&inst);
	int evo_threads = get_evo_threads(&inst);

	setup_parent_kernel<<<BLOCKS, inst.dim.matrix_height>>>(dev_inst);
	cudaThreadSynchronize();
	CUDA_CALL(cudaGetLastError());

	setup_sparam<<<BLOCKS, evo_threads>>>(dev_inst);
	cudaThreadSynchronize();
	CUDA_CALL(cudaGetLastError());

	// Prepare
	cudaEvent_t start, stop;
	float elapsedTime;
	float elapsedTimeTotal = 0.f;

	const dim3 blocks(BLOCKS, PARENTS*CHILDS);
	const dim3 threads(MATRIX_WIDTH, MATRIX_HEIGHT);
	const dim3 copy_threads(MATRIX_WIDTH, MATRIX_HEIGHT);

	const int width = inst.dim.parents * inst.dim.blocks;
	double * const rating = (double*)ya_malloc(width * sizeof(double));
	struct plot_log* pl = init_plot_log(&inst);

	int rounds = -1;
	int block = 0; int thread = 0;

	for(unsigned long i = 0; i < max_rounds; i++) {
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

		evo_kernel_part_two<<<BLOCKS, copy_threads>>>(dev_inst);
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
				i = max_rounds;
				break;
			}
		}
	}

	free(rating);
	clean_plot_log(pl);
	inst_copy_dev_to_host(dev_inst, &inst);

//	print_sparam(&inst);
	print_parent_ratings(&inst);

	printf("Time needed: %f\n", elapsedTimeTotal);
	printf("Needed rounds: %d\n", rounds);
	printf("Result is block: %d, parent: %d\n", block, thread);
	printf("Result was in block: %d, child: %d, selection: %d\n",
		inst.res_child_block, inst.res_child_idx, inst.res_parent);

	print_parents(&inst, block, thread, rounds);

	#ifdef DEBUG
	if(rounds != -1) {
		printf("Result Matrix:\n");
		print_result_matrix_pretty(&inst, block, 0);
		print_rules(stdout, &inst);
		print_result_matrix_pretty(&inst, block, 1);
		print_rules(stdout, &inst);
	}
	#endif

	printf("Clean up and exit.\n");
	inst_cleanup(&inst, dev_inst);

	if(rounds == -1)
		return 0;

	return 1;
}
