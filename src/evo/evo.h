/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#ifndef __EVO_H__
#define __EVO_H__

#include "instance.h"

__global__ void evo_kernel_part_one(struct instance *inst);
__global__ void evo_kernel_part_two(struct instance *inst);

#endif
