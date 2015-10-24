#ifndef NNU_H
#define NNU_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "half.h"
#include "util.h"
#include "linalg/linalg.h"

#define RANGE_16  1 << 16
#define RANGE_32 1 << 32

/* NNU dictionary (uint16 implementation) */
typedef struct NNUDictionary {
    int alpha; //number of tables
    int beta;  //width of tables
    uint16_t* tables; //nnu lookup tables

    double* D; //learned dictionary
    int D_rows; //rows in ldict
    int D_cols; //cols in ldict
    double* VD; //vtd
} NNUDictionary;

NNUDictionary* new_dict(const int alpha, const int beta,
                        const char *input_csv_path, const char *delimiters);
/* NNUDictionary* new_dict(const int alpha, const int beta, double *input_buf, */
/*                         int input_rows, int input_cols); */
void delete_dict(NNUDictionary *dict);


void atom_lookup(double *x, uint16_t ***iram, int **idx_arr,
                 const int alpha, const int beta);


#endif /*NNU_H*/
