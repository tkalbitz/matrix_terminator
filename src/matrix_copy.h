#ifndef MATRIX_COPY_H_
#define MATRIX_COPY_H_

#include "instance.h"

void copy_parents_dev_to_host(struct instance* inst, void* parent_cpy);
void copy_parent_rating_dev_to_host(struct instance* inst, void* parent_rat_cpy);
void copy_results_dev_to_host(struct instance* inst, void* result_cpy);

#endif /* MATRIX_COPY_H_ */
