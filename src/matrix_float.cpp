// this program searches a matrix individuumpretation
// over the naturals that is compatible with
// z001 = ( RULES a a b b -> b b b a a a) .

// compile: gcc -O6 -std=gnu9x -o matrix matrix.c
// run (example): ./matrix 5 1000 100
// should give a result within 10 seconds 
// (but it depends on the RNG initializiation).
// see end of file for description of cmd line args

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

float *
make(int rows, int cols)
{
	return (float *) malloc(rows * cols * sizeof(float));
}

void
copy(int rows, int cols, const float * src, float * dest)
{
	memcpy(dest, src, rows * cols * sizeof(float));
}

void
show(FILE * f, int mcount, int mwidth, const float * mat)
{
	for (int c = 0; c < mcount; c++) {
		for (int j = 0; j < mwidth; j++) {
			fprintf(f, "----");
		}
		fprintf(f, "letter %d\n", c);
		for (int i = 0; i < mwidth; i++) {
			for (int j = 0; j < mwidth; j++) {
				fprintf(
						f,
						"%4.2f ",
						mat[c * mwidth * mwidth
								+ i * mwidth + j]);
			}
			fprintf(f, "\n");
		}
	}
}

void
plus(int rows, int cols, const float * a, const float * b, float * c)
{
	for (int i = 0; i < rows * cols; i++) {
		c[i] = a[i] + b[i];
	}
}

void
times(int rows, int mid, int cols, const float * a, const float * b, float * c)
{
	for (int i = 0; i < rows; i++) {
		for (int k = 0; k < cols; k++) {
			int s = 0;
			for (int j = 0; j < mid; j++) {
				s += a[i * mid + j] * b[j * cols + k];
			}
			c[i * cols + k] = s;
		}
	}
}

// write individuumpretation of word w
// into pre-allocated result matrix
void
eval(int * rule, int mwidth, float * individuum, float * result)
{
	assert(rule[0] >= 0);
	copy(mwidth, mwidth, individuum + mwidth * mwidth * rule[0], result);
	for (int i = 1; rule[i] >= 0; i++) {
		float * accu = make(mwidth, mwidth);
		times(mwidth, mwidth, mwidth, result,
				individuum + mwidth * mwidth * rule[i], accu);
		copy(mwidth, mwidth, accu, result);
		free(accu);
	}
}

float
penalty(int * lhs, int * rhs,
// lhs, rhs are strings 0,1,2,..,
// terminated by negative number.
// FIXME: empty string not handled correctly
		int mwidth, float * individuum
		// individuumpretation  is array of matrices
		)
{
	float * l = make(mwidth, mwidth);
	eval(lhs, mwidth, individuum, l);
	float * r = make(mwidth, mwidth);
	eval(rhs, mwidth, individuum, r);
	float s = 0;
	for (int row = 0; row < mwidth; row++) {
		for (int col = 0; col < mwidth; col++) {
			float x = l[row * mwidth + col];
			float y = r[row * mwidth + col];
			float p = 0;
			if ((row == 0) && (col == mwidth - 1)) {
				p = (x > y) ? 0 : 10000 * (y - x + 1);
			} else {
				p = (x >= y) ? 0 : y * y - x * x;
			}
			if (p > 1000000)
				p = 1000000;
			s += p;
		}
	}
	free(l);
	free(r);
	return s;
}

// ensure special shape: 
// first column is (1,0..0)^T, last row is (0..0,1)
void
patch(int mwidth, float * m)
{
	for (int i = 0; i < mwidth; i++) {
		m[i * mwidth + 0] = 0;
		m[(mwidth - 1) * mwidth + i] = 0;
	}
	m[0 * mwidth + 0] = 1;
	m[(mwidth - 1) * mwidth + mwidth - 1] = 1;
}

// fill randomly with 0,1
void
fill(int mwidth, float * m)
{
	for (int i = 0; i < mwidth; i++) {
		for (int j = 0; j < mwidth; j++) {
			m[i * mwidth + j] = random() % 2;
		}
	}
	patch(mwidth, m);
}

// change one position (downwards only)
void
mutate(int mcount, int mwidth, float * individuum)
{
	int letter = random() % mcount;
	int row = random() % (mwidth - 1);
	int col = 1 + random() % (mwidth - 1);
	float * pos = individuum + letter * mwidth * mwidth + row * mwidth + col;
	int newP = *pos - 1;
	if (newP < 0)
		newP = 0;
	*pos = newP;
}

// this is the main trick, stolen from Dieter Hofbauer, 
// who had this in MultumNonMulta already in 2006:

// increase weights on path that corresponds to some
// error (= weight increase) in some position.

// find an index pair (p,q) such that lhs[p,q] < rhs[p,q],
// then find a random (!) path (sequence of indices)
// p = p_0 , p_1, p_2, .. , p_n = q,
// then for each i, increase the value of 
// individuum[p_i, p_i+1] in the individuumpretation of
// letter  lhs[i].

void
path_mutate(int * lhs, int * rhs, int mcount, int mwidth, float * individuum)
{
	float * lres = make(mwidth, mwidth);
	float * rres = make(mwidth, mwidth);

	eval(lhs, mwidth, individuum, lres);
	eval(rhs, mwidth, individuum, rres);

	int top = 0;
	int * rows = (int*)malloc(mwidth * mwidth * sizeof(float));
	int * cols = (int*)malloc(mwidth * mwidth * sizeof(float));

	for (int row = 0; row < mwidth; row++) {
		for (int col = 0; col < mwidth; col++) {

			int special = (row == 0) && (col == mwidth - 1);
			int l = lres[row * mwidth + col];
			int r = rres[row * mwidth + col];
			int ok = special ? (l > r) : (l >= r);

			if (!ok) {
				rows[top] = row;
				cols[top] = col;
				top++;
			}
		}
	}

	if (0 == top) {
		fprintf(stdout, "what");
		show(stdout, mcount, mwidth, individuum);
		// exit (0);
		return;
	}

	int i = random() % top;
	int l = rows[i];
	int r = cols[i];

	free(rows);
	free(cols);
	free(lres);
	free(rres);

	// now we have the position of the error in (l,q)

	for (int i = 0; lhs[i] >= 0; i++) {
		int c = lhs[i];
		int goal = lhs[i + 1] < 0 ? r : 1 + random() % (mwidth - 2);
		float * pos = individuum + c * mwidth * mwidth + l * mwidth
				+ goal;
		int newP = *pos + 1;
		if (newP < 1)
			newP = 1;
		*pos = newP;
		l = goal;
	}

}

// try to overwrite this individuum with a better one
int
anneal(int total, int * lhs, int * rhs, int mcount, int mwidth,
		float * individuum)
{
	float best = penalty(lhs, rhs, mwidth, individuum);
	if (0 == best)
		return 0;
	// fprintf (stdout, "anneal start %d\n", best);

	size_t s = mcount * mwidth * mwidth * sizeof(float);
	float * candidate = (float*)malloc(s);

	for (int steps = 0; steps < total; steps++) {
		memcpy(candidate, individuum, s);
		mutate(mcount, mwidth, candidate);
		float p = penalty(lhs, rhs, mwidth, candidate);
		int luck = 0 == random() % total;
		if (p <= best || luck) {
			memcpy(individuum, candidate, s);
			best = p;
		}
	}
	// fprintf (stdout, "anneal end %d\n", best);

	free(candidate);
	return best;
}

int
bits(int x)
{
	int c = 0;
	while (x > 0) {
		x >>= 1;
		c++;
	}
	return c;
}

void
census(char * s, int size, int * data)
{
	int maxbits = 32;
	int * tab = (int*)malloc(maxbits * sizeof(int));
	for (int i = 0; i < maxbits; i++) {
		tab[i] = 0;
	}
	for (int i = 0; i < size; i++) {
		tab[bits(data[i])]++;
	}
	printf("census: number of items with %s of given bit width\n", s);
	for (int i = 0; i < maxbits; i++) {
		if (tab[i] > 0) {
			printf("%d: %d, ", i, tab[i]);
		}
	}
	printf("\n");
	free(tab);
}

void
census(char * s, int size, float * data)
{
	int maxbits = 32;
	int * tab = (int*)malloc(maxbits * sizeof(int));
	for (int i = 0; i < maxbits; i++) {
		tab[i] = 0;
	}
	for (int i = 0; i < size; i++) {
		tab[bits((int)data[i])]++;
	}
	printf("census: number of items with %s of given bit width\n", s);
	for (int i = 0; i < maxbits; i++) {
		if (tab[i] > 0) {
			printf("%d: %d, ", i, tab[i]);
		}
	}
	printf("\n");
	free(tab);
}

static inline int timespec_subtract(struct timespec *result,
                                    struct timespec *after,
                                    struct timespec *before)
{
        result->tv_nsec = after->tv_nsec - before->tv_nsec;

        if (result->tv_nsec < 0) {
                /* Borrow 1sec from 'tv_sec' if subtraction -ve */
                result->tv_nsec += 1000000000;
                result->tv_sec = after->tv_sec - before->tv_sec - 1;

                return 1;
        } else {
                result->tv_sec = after->tv_sec - before->tv_sec;
                return 0;
        }
}

void
evolution(int size, int asteps, int * lhs, int * rhs, int mcount, int mwidth)
{
	int s = mcount * mwidth * mwidth;
	float * pop = (float*)malloc(size * s * sizeof(float));
	float * pen = (float*)malloc(size * sizeof(float));
	int * age = (int*)malloc(size * sizeof(int));

	for (int p = 0; p < size; p++) {
		float * individuum = pop + p * mcount * mwidth * mwidth;

		for (int c = 0; c < mcount; c++) {
			fill(mwidth, individuum + c * mwidth * mwidth);
		}

		pen[p] = penalty(lhs, rhs, mwidth, individuum);
		age[p] = 0;
	}

	int globally_best = pen[0];

	struct timespec before, after, elapsed;

	clock_gettime(CLOCK_MONOTONIC, &before);
	for (int step = 0;; step++) {
		int parent = random() % size;
		float * individuum = (float*)malloc(s * sizeof(float));
		memcpy(individuum, pop + parent * s, s * sizeof(int));

		path_mutate(lhs, rhs, mcount, mwidth, individuum);
		int best = anneal(asteps, lhs, rhs, mcount, mwidth, individuum);

		int child = random() % size;

		if (best < pen[child]) {
			if (best < globally_best) {
				fprintf(
						stdout,
						"step %5d: parent from %d with fit %d age %d replaces child at %d with fit %d age %d\n",
						step, parent, best, age[parent],
						child, pen[child], age[child]);
				globally_best = best;
			}

			memcpy(pop + child * s, individuum, s * sizeof(int));
			pen[child] = best;
			age[child] = age[parent] + 1;
		}

		if (0 == best) {
			show(stdout, mcount, mwidth, individuum);
			exit(0);
		}

		if (0 == step % 1000) {
			clock_gettime(CLOCK_MONOTONIC, &after);
			timespec_subtract(&elapsed, &after, &before);
			printf("step %d need %ld.%03lds\n", step, elapsed.tv_sec, elapsed.tv_nsec / (long int)1e6);
			census("penalty", size, pen);
			census("age", size, age);
			clock_gettime(CLOCK_MONOTONIC, &before);
		}
		free(individuum);
	}
	free(age);
	free(pop);
}

void
init_random_generator()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	srandom(tv.tv_usec);
}

int
main(int argc, char ** argv)
{
	if (4 != argc) {
		fprintf(stderr, "cmd line arguments:\n");
		fprintf(stderr, "  int mwidthension of matrices,\n");
		fprintf(stderr, "  int size of population,\n");
		fprintf(stderr, "  int number of annealing steps.\n");
		fprintf(stderr, "example: ./matrix 5 100 100\n");
		exit(-1);
	}

	int mwidth;
	sscanf(argv[1], "%d", &mwidth);
	int pop;
	sscanf(argv[2], "%d", &pop);
	int ann;
	sscanf(argv[3], "%d", &ann);

	int mcount = 2;

	int lhs[] = { 0, 0, 1, 1, -1 };
	int rhs[] = { 1, 1, 1, 0, 0, 0, -1 };

	// int lhs [] = { 0,0, -1};
	// int rhs [] = { 0,1,0, -1};

	init_random_generator();

	evolution(pop, ann, lhs, rhs, mcount, mwidth);
}
