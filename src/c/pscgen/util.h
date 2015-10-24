#ifndef CSV_H
#define CSV_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)

void read_csv(const char *filepath, const char *delimiters, double **buf,
              int *rows, int* cols);

void print_mat(double *buf, int rows, int cols);

inline double* new_dvec(int N);
inline void zero_dvec(double *vec, int N);
inline int idx2d(int i, int j, int rows);
inline int idx3d(int i, int j, int k, int rows, int cols);
void d_argsort(double *vec, int *idxs, int N);
double* d_transpose(double* mat, int rows, int cols);
double* d_trim(double* mat, int rows, int cols, int new_rows, int new_cols);

#endif /*CSV_H*/
