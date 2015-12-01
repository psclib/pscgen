#include <stdio.h>
#include "generator.h"
#include "classifier.h"
#include "pipeline.h"
#include "nnu_storage.h"
#include "nnu_dict.h"
#include "util.h"

int main(int argc, char **argv)
{
    int alpha = 10;
    int beta = 10;
    int num_clfs, ws, inter_r, inter_c, ss, num_classes, num_features;
    double *coefs, *intercepts;

    ss = 12;
    ws = 100;
    num_classes = 13;

    NNUDictionary *dict = new_dict(alpha, beta, mini, argv[1], ",");
    read_csv(argv[2], ",", &coefs, &num_features, &num_clfs);
    read_csv(argv[3], ",", &intercepts, &inter_r, &inter_c);
    SVM *svm = new_svm(num_features, num_classes, coefs, intercepts);
    Pipeline *pipeline = new_pipeline(dict, svm, ws, ss);


    char *p = pipeline_to_str(pipeline);
    printf("%s\n", p);

    free(p);
    delete_pipeline(pipeline);


    return 0;
}
