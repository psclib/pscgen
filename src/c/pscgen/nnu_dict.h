#ifndef NNU_DICT_H
#define NNU_DICT_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "linalg/linalg.h"
#include "nnu_storage.h"
#include "util.h"

typedef enum
{
    pca, /* use eigvectors */
    uniform_sub,
    random_linear_comb,
    random_sub,
    random_proj /* random gaussian projections */
} Compression_Scheme;

/* NNU dictionary */
typedef struct NNUDictionary {
    int alpha; /* curr alpha */
    int beta;  /* curr beta */
    int max_alpha; /* height of tables */
    int max_beta; /* width of tables */
    int gamma; /* depth of tables */
    Storage_Scheme storage; /* float representation of each index */
    Compression_Scheme comp_scheme;
    int D_rows; /* rows in D */
    int D_cols; /* curr cols in D */
    int max_atoms; /* max cols in D */
    
    uint16_t *tables; /* nnu lookup tables (stores candidates)*/
    double *D; /* learned dictionary */
    double *D_mean; /*colwise mean of D */
    double *Vt; /* Vt from SVD(D) -- taking alpha columns */
    double *VD; /* dot(Vt, d) */
} NNUDictionary;

/* Dynamically allocated NNUDictionary functionality */
double* build_sensing_mat(double *Dt, int rows, int cols,
                          Compression_Scheme comp_scheme, int alpha,
                          int s_stride);
NNUDictionary* new_dict(const int alpha, const int beta, const int max_atoms,
                        Storage_Scheme storage, Compression_Scheme comp_scheme,
                        const char *input_csv_path, const char *delimiters);
NNUDictionary* new_dict_from_buffer(const int alpha, const int beta,
                                    Storage_Scheme storage, 
                                    Compression_Scheme comp_scheme,
                                    double *D, int rows, int cols,
                                    int max_atoms);
void delete_dict(NNUDictionary *dict);
/* void save_dict(char *filepath, NNUDictionary *dict); */
/* NNUDictionary* load_dict(char *filepath); */


/* Search algorithms */
int* nnu(NNUDictionary *dict, int alpha, int beta, double *X, int X_rows,
         int X_cols, double *avg_ab);
int nnu_single(NNUDictionary *dict, double *X, int X_rows);
void nnu_single_candidates(NNUDictionary *dict, double *X, int X_rows, 
                           int** candidate_set, double** magnitudes, int* N);
int nnu_single_nodot(NNUDictionary *dict, double *X, int X_rows, 
                      int *candidate_set);
int nnu_pca_single(NNUDictionary *dict, double *X, int X_rows);
double* nns(NNUDictionary *dict, double *X, int X_rows, int X_cols);
int nns_single(NNUDictionary *dict, double *X, int X_rows);
double* mp(NNUDictionary *dict, double *X, int X_rows, int X_cols, int K);

/* Helper functions */
void compute_max_dot(double *max_coeff, int *max_idx, double *D,
                            double *x, int D_rows, int D_cols);
void compute_max_dot_set(double *max_coeff, int *max_idx, int *total_ab,
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
