#include "pipeline.h"


Pipeline* new_pipeline(NNUDictionary *nnu, SVM *svm, int N, int ws, int ss)
{
    Pipeline *pipeline;
    pipeline = (Pipeline *)malloc(sizeof(Pipeline));
    pipeline->nnu = nnu;
    pipeline->svm = svm;
    pipeline->num_windows = compute_num_sliding_windows(N, ws, ss);
    pipeline->ws = ws;
    pipeline->ss = ss;
    pipeline->N = N;
    pipeline->window_X = (double *)malloc(pipeline->num_windows*ws
                                          *sizeof(double));
    pipeline->bag_X = (double *)malloc(pipeline->nnu->D_cols*sizeof(double));

    return pipeline;
}

void delete_pipeline(Pipeline *pipeline)
{
    free(pipeline->nnu);
    free(pipeline->svm);
    free(pipeline->window_X);
    free(pipeline);
}

int classification_pipeline(double *X, Pipeline *pipeline)
{
    int *idxs;
    double avg_ab;

    sliding_window(X, pipeline->window_X, pipeline->N, pipeline->ws,
                   pipeline->ss);
    normalize_colwise(pipeline->window_X, pipeline->ws, pipeline->num_windows);
    subtract_colwise(pipeline->window_X, pipeline->nnu->D_mean, pipeline->ws,
                     pipeline->num_windows);
    idxs = nnu(pipeline->nnu, pipeline->nnu->alpha, pipeline->nnu->beta,
               pipeline->window_X, pipeline->num_windows, pipeline->ws,
               &avg_ab);
    bag_of_words(idxs, pipeline->bag_X, pipeline->N, pipeline->nnu->D_cols);

    return classify(pipeline->bag_X, pipeline->svm);
}
