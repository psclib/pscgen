#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "nnu.h"
#include "util.h"
#include "linalg/linalg.h"

int main(int argc, char *argv[])
{
    int alpha = 10;
    int beta = 10;
    int alpha_beta = alpha * beta;
    double *A;
    int rA, cA;
    NNUDictionary *dict = new_dict(alpha, beta, "test.csv", ",");
    read_csv("test.csv", ",", &A, &rA, &cA);
    double *ret = nnu(dict, A, rA, cA);


    /* print_mat(A, rA, cA); */
    /* print_mat(B, 1, 4); */
    /* print_mat(C, rA, cB); */

    //clean-up
    delete_dict(dict);

    return 0;
}
