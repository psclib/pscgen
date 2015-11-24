#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include "nnu_generator.h"
#include "nnu_storage.h"
#include "nnu_dict.h"
#include "util.h"

int main(int argc, char *argv[])
{
    time_t start, end;
    int alpha = 5;
    int beta = 5;
    double *A, ab;
    int rA, cA;

    NNUDictionary *dict = new_dict(alpha, beta, mini, "/home/brad/data/D1500_hog.csv", ",");
    /* NNUDictionary *dict = new_dict(alpha, beta, "/home/brad/data/notredame/tiny.csv", ","); */
    /* save_dict("/home/brad/data/dict1500hog.nnu", dict); */
    /* exit(1); */

    /* NNUDictionary *dict = load_dict("/home/brad/data/dict1500hog.nnu"); */
    read_csv("/home/brad/data/kth_test_hog.csv", ",", &A, &rA, &cA);
    /* read_csv("/home/brad/data/D1500_hog.csv", ",", &A, &rA, &cA); */
    /* read_csv("/home/brad/data/kth_test.csv", ",", &A, &rA, &cA); */
    /* print_mat(A, rA, cA); */
    int *ret = nnu(dict, 20, 20, A, rA, cA, &ab);
    /* int* lookup_hist = table_histogram2(dict, A, rA, cA); */
    /* double* lookup_dist = table_distance(dict, A, rA, cA); */


    print_mat_i(ret, 1, cA);
    /* print_mat(lookup_dist, alpha, beta); */
    /* print_mat(ret, 1, cA); */
    /* print_mat(ret2, 1, cA); */

    //clean-up
    delete_dict(dict);
    free(A);
    free(ret);
    /* free(lookup_dist); */

    return 0;
}
