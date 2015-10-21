#ifndef NNU_H
#define NNU_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "half.h"
#include "util.h"
#include "linalg/linalg.h"

#define RANGE_16  1 << 16

/* NNU dictionary (half float implementation) */
typedef struct NNUDictionary {
    int alpha; //number of tables
    int beta;  //width of tables
    half* tables; //nnu lookup tables

    double* ldict; //learned dictionary
    int ldict_rows; //rows in ldict
    int ldict_cols; //cols in ldict
    double* ldict_vt; //eigen vectors of ldict
} NNUDictionary;

NNUDictionary* new_dict(const int alpha, const int beta,
                        const char *input_csv_path, const char *delimiters);
/* NNUDictionary* new_dict(const int alpha, const int beta, double *input_buf, */
/*                         int input_rows, int input_cols); */
void delete_dict(NNUDictionary *dict);


void atom_lookup(double *x, uint16_t ***iram, int **idx_arr,
                 const int alpha, const int beta);


#endif /*NNU_H*/
