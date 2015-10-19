#ifndef PSCGEN_H
#define PSCGEN_H

#include <stdlib.h>
#include <string.h>
#include "half.h"

#define RANGE_16  1 << 16

/* NNU dictionary (half float implementation) */
typedef struct _NNUDictionary NNUDictionary;
NNUDictionary* new_dict(const int alpha, const int beta);
void build_dict(NNUDictionary *dict, double *input_buf, int input_rows,
                int input_cols);
void delete_dict(NNUDictionary *dict);

void dgemv_(char *TRANS, const int *M, const int *N, double *alpha,
            double *A, const int *LDA, double *X, const int *INCX,
            double *beta, double *C, const int *INCY);

int dgesvd_(char *jobz, char *jobvt, int *m, int *n, double *A, int *lda, 
            double *S, double *U, int *ldu, double *VT, int *ldvt,
            double *work, int *lwork, int *info);

void atom_lookup(double *x, uint16_t ***iram, int **idx_arr,
                 const int alpha, const int beta);


#endif /*PSCGEN_H*/
