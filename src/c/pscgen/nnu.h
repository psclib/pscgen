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
    double* Vt; //Vt from SVD(D) -- taking alpha columns
    double* VD; //dot(Vt, d)
    int* beta_scale; //number of beta values for each alpha
} NNUDictionary;

NNUDictionary* new_dict(const int alpha, const int beta,
                        const char *input_csv_path, const char *delimiters);
NNUDictionary* new_dict_from_buffer(const int alpha, const int beta,
                                    double *D, int rows, int cols);
void delete_dict(NNUDictionary *dict);
void save_dict(char *filepath, NNUDictionary *dict);
NNUDictionary* load_dict(char *filepath);

double* nnu(NNUDictionary *dict, double *X, int X_rows, int X_cols,
            double *avg_ab);
void atom_lookup(uint16_t *tables, double *x, word_t *atom_idxs,
                 int *candidate_set, int *N, int alpha, int beta,
                 int *beta_scale);
int* compute_beta_scale(double *s_values, int alpha, int beta);
#endif /*NNU_H*/
