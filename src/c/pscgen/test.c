#include "stdio.h"
#include "nnu.h"
#include "csv.h"
#include "linalg/linalg.h"



/* int dgesvd_(char *jobz, char *jobvt, int *m, int *n, double *A, int *lda, */ 
/*             double *S, double *U, int *ldu, double *VT, int *ldvt, */
/*             double *work, int *lwork, int *info); */


int main(int argc, char *argv[])
{
    /* int alpha = 10; */
    /* int beta = 10; */
    /* NNUDictionary *dict = new_dict(alpha, beta, "test.csv", ","); */
    /* delete_dict(dict); */

    int M, N;
    double *A;
    double alpha = 1.0;
    read_csv("test.csv", ",", &A, &M, &N);
    dgemv_("N", &M, &N, &alpha, A, 


    return 0;
}
