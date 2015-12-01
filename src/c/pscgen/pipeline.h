#ifndef PIPELINE_H
#define PIPELINE_H

#include "util.h"
#include "classifier.h"
#include "nnu_dict.h"

typedef struct Pipeline {
    int ws;
    int ss;

    double *window_X;
    double *bag_X;
    NNUDictionary *nnu;
    SVM *svm;
} Pipeline;


Pipeline* new_pipeline(NNUDictionary *nnu, SVM *svm, int ws, int ss);
void delete_pipeline(Pipeline *pipeline);
int classification_pipeline(double *X, int x_len, Pipeline *pipeline);

#endif /* PIPELINE_H */
