/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
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
