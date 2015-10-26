#include "linalg.h"

double* dmv_prod(double *A, double *x, int rows, int cols)
{
    double* y = malloc(sizeof(double) * cols);

    //TODO: confirm initalization is needed
    int i;
    for(i = 0; i < cols; i++) {
        y[i] = 0.0;
    }

    char no = 'N', tr = 'T';
    int lda = cols, incx = 1, incy = 1;
    double alpha = 1.0, beta = 0.0;
    printf("%d,%d\n", rows, cols);

    dgemv_(&no, &rows, &cols, &alpha, A, &lda, x, &incx, &beta, y, &incy);

    return y;
}

//A = M * K, B = K * N
double* dmm_prod(double *A, double *B, int A_rows, int A_cols, int B_rows,
                 int B_cols)
{
    if(A_cols != B_rows) {
        printf("dgemm matrix mismatch:\nA:%d, %d\nB:%d, %d\n", A_rows, A_cols,
               B_rows, B_cols);
        exit(1);
    }

    double* C = malloc(sizeof(double) * A_rows * B_cols);

    int i, j;
    for(i = 0; i < A_rows; i++) {
        for(j = 0; j < B_cols; j++) {
            C[idx2d(i, j, A_rows)] = 0.0;
        }
    }

    char no = 'N';
    double alpha = 1.0, beta = 0.0;

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
double* deig_Vt(double *input, int rows, int cols)
{
    int info, lwork;
    double wkopt;
    double *work;

    //TODO: switch to memcpy
    double *A = malloc(sizeof(double) * rows * cols);
    int i;
    for(i = 0; i < rows*cols; i++)
        A[i] = input[i];

    double *S = malloc(sizeof(double) * MIN(rows, cols));
    double *U = malloc(sizeof(double) * rows * rows);
    double *VT = malloc(sizeof(double) * cols * cols);

    /* Query and allocate the optimal workspace */
    lwork = -1;
    dgesvd_("All", "All", &rows, &cols, A, &rows, S, U, &rows, VT,
            &cols, &wkopt, &lwork, &info);
    lwork = (int)wkopt;
    work = malloc(sizeof(double) * lwork);

    /* Compute SVD */
    dgesvd_("All", "All", &rows, &cols, A, &rows, S, U, &rows, VT,
            &cols, work, &lwork, &info);

    /* Check for convergence */
    if(info > 0) {
        printf("The algorithm computing SVD failed to converge.\n");
        exit(1);
    }

    //clean-up
    free(A);
    free(work);
    free(S);
    free(U);

    return VT;
}
