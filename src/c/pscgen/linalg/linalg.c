#include "linalg.h"

double* dmv_prod(double *A, double *x, int rows, int cols)
{
    double* y = calloc(cols, sizeof(double));
    char no = 'N';
    int lda = cols, incx = 1, incy = 1;
    double alpha = 1.0, beta = 0.0;

    dgemv_(&no, &rows, &cols, &alpha, A, &lda, x, &incx, &beta, y, &incy);

    return y;
}

/* A = M * K, B = K * N */
double* dmm_prod(double *A, double *B, int A_rows, int A_cols, int B_rows,
                 int B_cols)
{
    char no = 'N';
    double alpha = 1.0, beta = 0.0;
    double* C = calloc(A_rows * B_cols, sizeof(double));

    if(A_cols != B_rows) {
        printf("dgemm matrix mismatch:\nA:%d, %d\nB:%d, %d\n", A_rows, A_cols,
               B_rows, B_cols);
        exit(1);
    }

    dgemm_(&no, &no, &A_rows, &B_cols, &A_cols, &alpha, A, &A_rows, B, &B_rows,
           &beta, C, &A_rows);
           
    return C;
}

double d_dot(double *X, double *Y, int N)
{
    int incx = 1, incy = 1;
    return ddot_(&N, X, &incx, Y, &incy);
}

/* Returns Vt of a matrix A */
void d_SVD(double *input, int rows, int cols, double **U, double **S,
           double **VT)
{
    int info, lwork;
    double wkopt;
    double *work;
    const char *jobu = "All";
    const char *jobv = "All";

    /* TODO: switch to memcpy */
    double *A = malloc(sizeof(double) * rows * cols);
    int i;
    int min_rc = MIN(rows, cols);

    for(i = 0; i < rows*cols; i++)
        A[i] = input[i];

    *S = malloc(sizeof(double) * min_rc);
    *U = malloc(sizeof(double) * rows * rows);
    *VT = malloc(sizeof(double) * cols * cols);

    /* Query and allocate the optimal workspace */
    lwork = -1;
    dgesvd_((char *)jobu, (char *)jobv, &rows, &cols, A, &rows, *S, *U, &rows,
            *VT, &cols, &wkopt, &lwork, &info);
    lwork = (int)wkopt;
    work = malloc(sizeof(double) * lwork);

    /* Compute SVD */
    dgesvd_((char *)jobu, (char *)jobv, &rows, &cols, A, &rows, *S, *U, &rows,
            *VT, &cols, work, &lwork, &info);

    /* Check for convergence */
    if(info > 0) {
        printf("The algorithm computing SVD failed to converge.\n");
        exit(1);
    }

    /* clean-up */
    free(A);
    free(work);
}
