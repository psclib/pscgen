#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <math.h>
#include <stdlib.h>
#include "util.h"

/* linear SVM */
typedef struct SVM {
    int num_features;
    int num_classes;
    int num_clfs;
    
    int *wins;
    double *coefs;
    double *intercepts;
} SVM;


SVM* new_svm(int num_features, int num_classes, double *coefs,
             double *intercepts);
void delete_svm(SVM *svm);
void class_idxs(int idx, int num_classes, int *c1, int *c2);
int classify(double *X, SVM *svm);


#endif /* CLASSIFIER_H */
