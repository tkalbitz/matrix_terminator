#ifndef __M_CONFIG_H__
#define __M_CONFIG_H__

#define BLOCKS  5
#define THREADS 5
#define CHILDS   7 

#define MATRIX_HEIGHT 5
#define MATRIX_WIDTH  5

#if(THREADS < MATRIX_HEIGHT)
	#error Init parents my be not working
#endif

/* how many positions in the matrix should contain a value !+ 0 */
#define MATRIX_TAKEN_POS 0.3f

#endif
