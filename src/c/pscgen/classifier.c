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
    svm->num_clfs = (num_classes * (num_classes - 1)) / 2;
    svm->wins = (int *)calloc(num_classes, sizeof(int));
    svm->coefs = coefs;
    svm->intercepts = intercepts;

    return svm;
}

void delete_svm(SVM *svm)
{
    free(svm->wins);
    free(svm);
}


void class_idxs(int idx, int num_classes, int *c1, int *c2)
{
    int i, j, curr_idx;

    curr_idx = 0;
    for(i = 0; i < num_classes; i++) {
        for(j = i + 1; j < num_classes; j++) {
            if(idx == curr_idx) {
                *c1 = i;
                *c2 = j;
                return;
            }
            curr_idx++;
        }
    }
}

int classify(double *X, SVM *svm)
{
    int i, c1, c2, max_wins, max_class_idx;
    double *coef_col;

    c1 = c2 = max_wins = max_class_idx = 0;

    /* Clear wins */
    memset(svm->wins, 0, svm->num_classes);

    /* Do one v. one classification */
    for(i = 0; i < svm->num_clfs; i++) {
        class_idxs(i, svm->num_classes, &c1, &c2);
        coef_col = d_viewcol(svm->coefs, i, svm->num_features);
        if(d_dot(coef_col, X, svm->num_features )+ svm->intercepts[i] > 0) {
            svm->wins[c1]++;
        }
        else {
            svm->wins[c2]++;
        }
    }

    /* Find winner */
    for(i = 0; i < svm->num_classes; i++) {
        if(svm->wins[i] > max_wins) {
            max_wins = svm->wins[i];
            max_class_idx = i;
        }
    }

    return max_class_idx;
}
