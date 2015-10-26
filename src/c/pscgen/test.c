#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "util.h"
#include "linalg/linalg.h"

int main(int argc, char *argv[])
{
    int alpha = 10;
    int beta = 10;
    int alpha_beta = alpha * beta;
    /* NNUDictionary *dict = new_dict(alpha, beta, "test.csv", ","); */
    /* delete_dict(dict); */
    unsigned int *atom_idxs = (int *)calloc(alpha_beta / 8 + 1,
                                            sizeof(unsigned int));
    bit_set_idx(atom_idxs, 1);
    bit_set_idx(atom_idxs, 44);
    bit_set_idx(atom_idxs, 25);
    int j;
    
    for(j = 0; j < alpha_beta; j++) {
        //skip missing values
        if(atom_idxs[j] == 0) {
            continue;
        }

        printf("%d\n", j);
    }
   

    /* double *A; */
    /* int rA, cA, rB, cB; */
    /* read_csv("test2.csv", ",", &A, &rA, &cA); */

    /* int i; */
    /* for(i = 0; i < cA; i++) */
    /*     printf("%d\n", A1[i]); */

    /* read_csv("vec2.csv", ",", &B, &rB, &cB); */

    /* double *C = dmm_prod(A, B, rA, cA, rB, cB); */
    /* double *B = d_trim(A, rA, cA, 1, 4); */

    /* print_mat(A, rA, cA); */
    /* print_mat(B, 1, 4); */
    /* print_mat(C, rA, cB); */

    return 0;
}
