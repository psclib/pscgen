#ifndef PSCGEN_H
#define PSCGEN_H

#include <stdlib.h>
#include <string.h>
#include "half.h"

#define RANGE_16  1 << 16

/* NNU dictionary (half float implementation) */
typedef struct _NNUDictionary NNUDictionary;
NNUDictionary* new_dict(const int alpha, const int beta,
                        const char *input_csv_path, const char *delimiters);
/* NNUDictionary* new_dict(const int alpha, const int beta, double *input_buf, */
/*                         int input_rows, int input_cols); */
void delete_dict(NNUDictionary *dict);


void atom_lookup(double *x, uint16_t ***iram, int **idx_arr,
                 const int alpha, const int beta);


#endif /*PSCGEN_H*/
