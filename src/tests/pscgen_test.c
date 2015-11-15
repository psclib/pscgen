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
    int beta = 500;
    int gamma_pow = 16;
    int alpha_beta = alpha * beta;
    double *A, ab;
    int rA, cA;

    NNUDictionary *dict = new_dict(alpha, beta, gamma_pow, "/home/brad/data/D1500_hog.csv", ",");
    /* NNUDictionary *dict = new_dict(alpha, beta, "/home/brad/data/D2000.csv", ","); */
    /* NNUDictionary *dict = new_dict(alpha, beta, "/home/brad/data/notredame/tiny.csv", ","); */
    save_dict("/home/brad/data/dict1500hog.nnu", dict);
    /* exit(1); */

    /* NNUDictionary *dict = load_dict("/home/brad/data/dict1500hog.nnu"); */
    /* read_csv("/home/brad/data/kth_test_hog.csv", ",", &A, &rA, &cA); */
    read_csv("/home/brad/data/D1500_hog.csv", ",", &A, &rA, &cA);
    /* read_csv("/home/brad/data/kth_test.csv", ",", &A, &rA, &cA); */
    /* print_mat(A, rA, cA); */
    /* double *ret = nns(dict, A, rA, cA); */
    /* int* lookup_hist = table_histogram2(dict, A, rA, cA); */
    /* double* lookup_dist = table_distance(dict, A, rA, cA); */


    /* print_mat_i(lookup_hist, alpha, beta); */
    /* print_mat(lookup_dist, alpha, beta); */
    /* print_mat(ret, 1, cA); */
    /* print_mat(ret2, 1, cA); */

    //clean-up
    delete_dict(dict);
    free(A);
    /* free(lookup_hist); */
    /* free(lookup_dist); */

    return 0;
}
