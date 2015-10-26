#ifndef NNU_H
#define NNU_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "half.h"
#include "util.h"
#include "hashset.h"
#include "hashset_itr.h"
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
    double* Vt; //Vt from SVD(D) -- taking alpha columns
    double* VD; //dot(Vt, d)
} NNUDictionary;

NNUDictionary* new_dict(const int alpha, const int beta,
                        const char *input_csv_path, const char *delimiters);
void delete_dict(NNUDictionary *dict);
double* nnu(NNUDictionary *dict, double *X, int X_rows, int X_cols);
inline void atom_lookup(uint16_t *tables, double *x, unsigned int *atom_idxs,
                        int alpha, int beta);

#endif /*NNU_H*/
