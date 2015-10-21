/*
 * Wrapper around CLAPACK 3.2.1
 *
*/

#ifndef LINALG_H
#define LINALG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//BLAS/LAPACK Prototypes
void dgemv_(char *TRANS, const int *M, const int *N, double *alpha,
            double *A, const int *LDA, double *X, const int *INCX,
            double *beta, double *C, const int *INCY);
int dgesvd_(char *jobz, char *jobvt, int *m, int *n, double *A, int *lda, 
            double *S, double *U, int *ldu, double *VT, int *ldvt,
            double *work, int *lwork, int *info);

//Wrapper functions
double* dmv_prod(double *A, double *x, int rows, int cols);
double* deig_vec(double *A, int rows, int cols);

#endif /* LINALG_H */
