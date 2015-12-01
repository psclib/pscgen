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
    free(pipeline->window_X);
    free(pipeline);
}

int classification_pipeline(double *X, int x_len, Pipeline *pipeline)
{
    int i, idx;
    double l2_norm;

    memset(pipeline->bag_X, 0, pipeline->nnu->D_cols);

    for(i = 0; i*pipeline->ss < x_len - pipeline->ws + 1; i++) {
        ith_window(X, pipeline->window_X, i, pipeline->ws, pipeline->ss);
        normalize_colwise(pipeline->window_X, pipeline->ws, 1);
        subtract_colwise(pipeline->window_X, pipeline->nnu->D_mean,
                         pipeline->ws, 1);

        idx = nnu_single(pipeline->nnu, pipeline->window_X, pipeline->ws);
        pipeline->bag_X[idx] += 1.0;
    }

    
    l2_norm = norm(pipeline->bag_X, pipeline->nnu->D_cols);

    for(i = 0; i < pipeline->nnu->D_cols; i++) {
        pipeline->bag_X[i] /= l2_norm;
    }

    return classify(pipeline->bag_X, pipeline->svm);
}
