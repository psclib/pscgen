#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <pscgen/pscgen.h>

/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, int m, int n, double* a, int lda );

int main(int argc, char** argv) {
    uint16_t*** dict = new_dict(5, 5, RANGE_16);
    int rows, cols;
    double* learned_dict = read_csv(argv[1], rows, cols);

    int info, lwork;
    double wkopt;
    double *work;
    double *S = new double[rows];
    double *U = new double[cols*cols];
    double *VT = new double[rows*rows];

    /* Query and allocate the optimal workspace */
    lwork = -1;
    dgesvd_("All", "All", &cols, &rows, learned_dict, &cols, S, U, &cols, VT,
            &rows, &wkopt, &lwork, &info);
    lwork = (int)wkopt;
    work = new double[lwork];
    /* Compute SVD */
    dgesvd_("All", "All", &cols, &rows, learned_dict, &cols, S, U, &cols, VT,
            &rows, work, &lwork, &info);
    /* Check for convergence */
    if( info > 0 ) {
        printf( "The algorithm computing SVD failed to converge.\n" );
        exit( 1 );
    }
    /* Print singular values */
    print_matrix( "Singular values", 1, rows, S, 1 );
    /* Print left singular vectors */
    print_matrix( "Left singular vectors (stored columnwise)", cols, rows, U, cols );
    /* Print right singular vectors */
    print_matrix( "Right singular vectors (stored rowwise)", rows, rows, VT, cols );

    //clean-up
    delete [] learned_dict;
    delete [] work;
    delete [] S;
    delete [] U;
    delete [] VT;
    delete_dict(dict, 5, RANGE_16);
}


/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, int m, int n, double* a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
                printf( "\n" );
        }
}
