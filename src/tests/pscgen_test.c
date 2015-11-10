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
    int alpha = 5;
    int beta = 10;
    int alpha_beta = alpha * beta;
    double *A, ab;
    int rA, cA;

    NNUDictionary *dict = new_dict(alpha, beta, "/home/brad/data/D.csv", ",");
    /* NNUDictionary *dict = new_dict(alpha, beta, "/home/brad/data/D2000.csv", ","); */
    /* NNUDictionary *dict = new_dict(alpha, beta, "/home/brad/data/notredame/tiny.csv", ","); */
    save_dict("/home/brad/data/dict1500alpha30.nnu", dict);


    /* NNUDictionary *dict = load_dict("/home/brad/data/dict5kalpha20.nnu"); */
    read_csv("/home/brad/data/kth_test.csv", ",", &A, &rA, &cA);
    /* read_csv("/home/brad/data/notredame/tiny.csv", ",", &A, &rA, &cA); */
    /* print_mat(A, rA, cA); */
    double *ret = nnu(dict, A, rA, cA, &ab);

    print_mat(ret, cA, 1);

    //clean-up
    delete_dict(dict);
    free(A);
    free(ret);

    return 0;
}
