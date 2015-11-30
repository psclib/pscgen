#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include "nnu_generator.h"
#include "nnu_storage.h"
#include "nnu_dict.h"
#include "nnu_classifier.h"
#include "util.h"

int main(int argc, char *argv[])
{
    int i;
    int N = 10;
    int ws = 4;
    int ss = 1;
    double X[10] = {1,2,3,4,5,6,7,8,9,10};
    double Y[7] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
    int Z[22] = {1,4,7,3,2,2,6,6,9,2,1,2,4,2,3,2,5,6,5,4,3,5};
    int num_windows = (int *)compute_num_sliding_windows(N, ws, ss);
    double *window_X = (double *)calloc(num_windows * ws, sizeof(double));
    double *bag_rep = (double *)calloc(11, sizeof(double));

    sliding_window(X, window_X, N, ws, ss);
    print_mat(window_X, ws, num_windows);

    printf("------\n");

    normalize_colwise(window_X, ws, num_windows);
    print_mat(window_X, ws, num_windows);

    printf("------\n");

    subtract_rowwise(window_X, Y, ws, num_windows);
    print_mat(window_X, ws, num_windows);
    
    printf("------\n");
    bag_of_words(Z, bag_rep, 22, 11);
    print_mat(bag_rep, 1, 11);

    printf("------\n");

    return 0;
}
