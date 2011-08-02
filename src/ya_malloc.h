/*
 * ya_malloc.h
 *
 *  Created on: Aug 2, 2011
 *      Author: tkalbitz
 */

#ifndef YA_MALLOC_H_
#define YA_MALLOC_H_

#include <stdlib.h>
#include <stdio.h>

inline static void* ya_malloc(size_t size)
{
	void* m = malloc(size);

	if(!m) {
		fprintf(stderr, "The end of the world. Malloc failed and I will "
				"crash and burn.\n");
		fflush(stderr);
		abort();
	}

	return m;
}



#endif /* YA_MALLOC_H_ */
