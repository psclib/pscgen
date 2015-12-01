#include <stdio.h>
#include "generator.h"
#include "classifier.h"
#include "pipeline.h"
#include "nnu_storage.h"
#include "nnu_dict.h"

int main(int argc, char **argv)
{
    int alpha = 10;
    int beta = 10;

    NNUDictionary *dict = new_dict(alpha, beta, mini, argv[1], ",");
    double *coefs = calloc(100*78, sizeof(double));
    double *intercepts = calloc(78, sizeof(double));
    SVM *svm = new_svm(100, 13, coefs, intercepts);
    Pipeline *pipeline = new_pipeline(dict, svm, 200, 100, 50);


    char *p = pipeline_to_str(pipeline);
    printf("%s\n", p);

    free(p);
    delete_pipeline(pipeline);


    return 0;
}
