#ifndef NNU_DICT_H
#define NNU_DICT_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#ifndef STANDALONE
#include "nnu_storage.h"
#include "util.h"
#endif

/* NNU dictionary */
typedef struct NNUDictionary {
    int alpha; /* height of tables */
    int beta;  /* width of tables */
    int gamma; /* depth of tables */
    Storage_Scheme storage; /*float representation of each index */
    
    uint16_t *tables; /* nnu lookup tables (stores candidates)*/
    double *D; /* learned dictionary */
    int D_rows; /* rows in ldict */
    int D_cols; /* cols in ldict */
    double *Vt; /* Vt from SVD(D) -- taking alpha columns */
    double *VD; /* dot(Vt, d) */
} NNUDictionary;

/* Search algorithms */
int* nnu(NNUDictionary *dict, int alpha, int beta, double *X, int X_rows,
         int X_cols, double *avg_ab);
double* nns(NNUDictionary *dict, double *X, int X_rows, int X_cols);
double* mp(NNUDictionary *dict, double *X, int X_rows, int X_cols, int K);

/* Helper functions */
inline void compute_max_dot(double *max_coeff, int *max_idx, double *D,
                            double *x, int D_rows, int D_cols);
inline void compute_max_dot_set(double *max_coeff, int *max_idx, int *total_ab,
                                double *D, double *x, int *candidate_set,
                                int D_rows, int N);
void atom_lookup(NNUDictionary *dict, double *x, word_t *atom_idxs,
                 int *candidate_set, int *N, int alpha, int beta,
                 int s_stride);

/* Analysis functions */
int* table_histogram(NNUDictionary *dict, double *X, int X_rows, int X_cols);
int* table_histogram2(NNUDictionary *dict, double *X, int X_rows, int X_cols);
double* table_distance(NNUDictionary *dict, double *X, int X_rows, int X_cols);

#endif /*NNU_DICT_H*/
