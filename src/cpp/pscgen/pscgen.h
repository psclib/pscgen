#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdint.h>
#include <vector>
#include <set>
#include "half.hpp"
#include "sortindex.hpp"
using half_float::half;


const int RANGE_16 = 1 << 16;

extern "C" {
    void dgemv_(char *TRANS, const int *M, const int *N, double *alpha,
                double *A, const int *LDA, double *X, const int *INCX,
                double *beta, double *C, const int *INCY);


    int dgesvd_(char *jobz, char *jobvt, int *m, int *n, double *A, int *lda, 
                double *S, double *U, int *ldu, double *VT, int *ldvt,
                double *work, int *lwork, int *info);


    uint16_t*** new_dict(const int alpha, const int beta, const int range);

    void build_dict(double **vtd, uint16_t ***iram, const int alpha, 
                    const int range);

    void delete_dict(uint16_t ***dict, const int alpha, const int range);

    void atom_lookup(double *x, uint16_t ***iram, int **idx_arr,
                     const int alpha, const int beta);
}

double* read_csv(std::string file, int &rows, int &cols);
