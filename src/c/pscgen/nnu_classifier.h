#ifndef NNU_CLASSIFIER_H
#define NNU_CLASSIFIER_H

#include <math.h>
#include <stdlib.h>
#include "util.h"

/* linear SVM */
typedef struct SVM {
    int num_features;
    int num_classes;
    int num_clfs;

    double *coefs;
    double *intercepts;
} SVM;


int compute_num_sliding_windows(int N, int ws, int ss);
void sliding_window(double *X, double *window_X, int N, int ws, int ss);
void normalize_rowwise(double *X, int rows, int cols);
double norm(double *X, int N);
void subtract_rowwise(double *X, double *Y, int rows, int cols);
void bag_of_words(int *X, double *bag_X, int N, int max_len);

void class_idxs(int idx, int num_classes, int *c1, int *c2);
int classify(double *X, SVM *svm);
//int command = classify(bag_rep, dict->svm_weights);


#endif /* NNU_CLASSIFIER_H */
