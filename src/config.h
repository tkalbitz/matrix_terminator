#ifndef __M_CONFIG_H__
#define __M_CONFIG_H__

#define BLOCKS  32

#define PARENTS 30
#define CHILDS   7 

#define RECOMB_RATE 0.7
#define MUT_RATE    0.3 

#define MATRIX_HEIGHT 5
#define MATRIX_WIDTH  5

/* HACK: we can"t allocate memory dynamically, this should be enough */
#define MUL_ROW_LEN MATRIX_WIDTH

/* how many positions in the matrix should contain a value !+ 0 */
#define MATRIX_TAKEN_POS 0.3f

#endif
