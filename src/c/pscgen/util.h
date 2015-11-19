#ifndef UTIL_H
#define UTIL_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)

typedef uint32_t word_t;
enum { WORD_SIZE = sizeof(word_t) * 8 };

/* bit-set functions */
word_t* bit_vector(int N);
int bindex(int b);
int boffset(int b);
void set_bit(word_t *data, int b);
void clear_bit(word_t *data, int b);
int get_bit(word_t *data, int b);
void clear_all_bit(word_t *data, int N);
void bit_set_idx(unsigned int *bitarray, size_t idx);

/* matrix functions */
void read_csv(const char *filepath, const char *delimiters, double **buf,
              int *rows, int* cols);
void print_mat(double *buf, int rows, int cols);
void print_mat_i(int *buf, int rows, int cols);
double* new_dvec(int N);
void zero_dvec(double *vec, int N);
void d_argsort(double *vec, int *idxs, int N);
double* d_transpose(double* mat, int rows, int cols);
double* d_trim(double* mat, int rows, int new_rows, int new_cols);
double* d_viewcol(double* mat, int col, int rows);
double d_dot(double *X, double *Y, int N);
double* dmm_prod(double *A, double *B, int A_rows, int A_cols, int B_rows,
                 int B_cols);

/* indexing functions */
int idx2d(int i, int j, int rows);
int idx3d(int i, int j, int k, int rows, int cols);

/* misc math */
int ipow(int base, int exp);

/* timing functions */
struct timespec t_diff(struct timespec start, struct timespec end);

#endif /*UTIL_H*/
