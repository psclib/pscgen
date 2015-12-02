#include "classifier.h"


/* coefs and intercepts are pre-trained.
 * coefs (num_clfs * num_features)
 * intercepts (num_clfs)
 */
SVM* new_svm(int num_features, int num_classes, double *coefs,
             double *intercepts)
{
    SVM *svm;
    svm = (SVM *)malloc(sizeof(SVM));
    svm->num_features = num_features;
    svm->num_classes = num_classes;
    svm->num_clfs = num_classes;

    svm->coefs = coefs;
    svm->intercepts = intercepts;

    return svm;
}

void delete_svm(SVM *svm)
{
    free(svm);
}

int classify(double *X, SVM *svm)
{
    int i, max_class_idx;
    double *coef_col, max_coef, tmp_coef;

    /* Do one v. rest classification */
    for(i = 0; i < svm->num_clfs; i++) {
        coef_col = d_viewcol(svm->coefs, i, svm->num_features);
        tmp_coef = d_dot(coef_col, X, svm->num_features) + svm->intercepts[i];
        if(i == 0 || tmp_coef > max_coef) {
            max_coef = tmp_coef;
            max_class_idx = i;
        }
    }

    return max_class_idx;
}
