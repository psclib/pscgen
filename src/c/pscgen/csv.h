#ifndef CSV_H
#define CSV_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void read_csv(const char *filepath, const char *delimiters, int *rows,
              int* cols, double **buf);

void print_mat(double *buf, int rows, int cols);

#endif /*CSV_H*/
