/*
 * Wrapper around CLAPACK 3.2.1
 *
*/

#ifndef LINALG_H
#define LINALG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef STANDALONE
#include "../util.h"
#endif

/* BLAS/LAPACK Prototypes */
void dgemv_(char *TRANS, const int *M, const int *N, double *alpha,
            double *A, const int *LDA, double *X, const int *INCX,
            double *beta, double *C, const int *INCY);
void dgemm_(char *TRANSA, char *TRANSB, const int *M, const int *N, 
            const int *K, double *alpha, double *A, const int *LDA, double *B,
            const int *LDB, double *beta, double *C, int *LDC);
int dgesvd_(char *jobz, char *jobvt, int *m, int *n, double *A, int *lda, 
            double *S, double *U, int *ldu, double *VT, int *ldvt,
            double *work, int *lwork, int *info);
double ddot_(int *N, double *DX, int *INCX, double *DY, int *INCY);

/* Wrapper functions */
double* blas_dmv_prod(double *A, double *x, int rows, int cols);
double* blas_dmm_prod(double *A, double *B, int A_rows, int A_cols, int B_rows,
                      int B_cols);
void lapack_d_SVD(double *input, int rows, int cols, double **U, double **S,
                  double **VT);
double blas_d_dot(double *X, double *Y, int N);

#endif /* LINALG_H */
