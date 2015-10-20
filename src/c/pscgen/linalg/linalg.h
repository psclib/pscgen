/*
 * CLAPACK 3.2.1 
 *
*/

#ifndef LINALG_H
#define LINALG_H


void dgemv_(char *TRANS, const int *M, const int *N, double *alpha,
            double *A, const int *LDA, double *X, const int *INCX,
            double *beta, double *C, const int *INCY);


#endif /* LINALG_H */
