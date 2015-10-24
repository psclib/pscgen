#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nnu.h"
#include "util.h"
#include "linalg/linalg.h"

int main(int argc, char *argv[])
{
    int alpha = 10;
    int beta = 10;
    NNUDictionary *dict = new_dict(alpha, beta, "test.csv", ",");
    delete_dict(dict);

    /* double *A; */
    /* int rA, cA, rB, cB; */
    /* read_csv("test2.csv", ",", &A, &rA, &cA); */
    /* read_csv("vec2.csv", ",", &B, &rB, &cB); */

    /* double *C = dmm_prod(A, B, rA, cA, rB, cB); */
    /* double *B = d_trim(A, rA, cA, 1, 4); */

    /* print_mat(A, rA, cA); */
    /* print_mat(B, 1, 4); */
    /* print_mat(C, rA, cB); */

    return 0;
}
