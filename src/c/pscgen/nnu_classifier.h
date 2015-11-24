#ifndef NNU_CLASSIFIER_H
#define NNU_CLASSIFIER_H

#include <math.h>
#include "util.h"

//Sliding window the input signal
int compute_num_sliding_windows(int N, int ws, int ss);
void sliding_window(double *X, double *window_X, int N, int ws, int ss);

//Normalize the windows
void normalize_rowwise(double *X, int rows, int cols);
double norm(double *X, int N);

//Zero mean the windows
void subtract_rowwise(double *X, double *Y, int rows, int cols);

//idxs = nnu(windows, dict);
//bag_rep = bow(idxs);
void bag_of_words(int *X, double *bag_X, int N, int max_len);
//int command = classify(bag_rep, dict->svm_weights);


#endif /* NNU_CLASSIFIER_H */
