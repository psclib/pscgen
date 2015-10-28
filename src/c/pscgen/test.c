#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include "nnu.h"
#include "util.h"
#include "linalg/linalg.h"

int main(int argc, char *argv[])
{
    time_t start, end;
    int alpha = 10;
    int beta = 10;
    int alpha_beta = alpha * beta;
    double *A;
    int rA, cA;

    time(&start);
    NNUDictionary *dict = new_dict(alpha, beta, "test.csv", ",");
    time(&end);

    printf("%.5f secs for training\n", difftime(end, start));
    read_csv("nnutest/large_test.csv", ",", &A, &rA, &cA);
    printf("%d, %d\n", rA, cA);

    time(&start);
    double *ret = nnu(dict, A, rA, cA);
    time(&end);

    printf("%.5f secs for nnu\n", difftime(end, start));
    /* print_mat(ret, cA, 1); */
    /* print_mat(A, rA, cA); */
    /* print_mat(B, 1, 4); */
    /* print_mat(C, rA, cB); */

    //clean-up
    delete_dict(dict);
    free(A);
    free(ret);

    return 0;
}
