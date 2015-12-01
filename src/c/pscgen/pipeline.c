#include "pipeline.h"


Pipeline* new_pipeline(NNUDictionary *nnu, SVM *svm, int ws, int ss)
{
    Pipeline *pipeline;
    pipeline = (Pipeline *)malloc(sizeof(Pipeline));
    pipeline->nnu = nnu;
    pipeline->svm = svm;
    pipeline->ws = ws;
    pipeline->ss = ss;
    pipeline->window_X = (double *)calloc(ws, sizeof(double));
    pipeline->bag_X = (double *)calloc(pipeline->nnu->D_cols, sizeof(double));

    return pipeline;
}

void delete_pipeline(Pipeline *pipeline)
{
    free(pipeline->nnu);
    free(pipeline->svm);
    free(pipeline->window_X);
    free(pipeline);
}
