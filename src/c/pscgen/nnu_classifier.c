#include "nnu_classifier.h"

/* Given dim of data N, window size ws and stride size ss, compute number
 * of windows that would be generated
 */
int compute_num_sliding_windows(int N, int ws, int ss)
{
    int i;
    int windows = 0;

    for(i = 0; i < N - ws + 1; i += ss) {
        windows++;
    }

    return windows;
}

/* Takes X of length N and returns sliding window of size ws with stride ss 
 * STORED COULMN-WISE
 * */
void sliding_window(double *X, double *window_X, int N, int ws, int ss)
{
    int i, j;

    for(i = 0; i < N - ws + 1; i += ss) {
        for(j = 0; j < ws; j++) {
            window_X[idx2d(j, i, ws)] = X[i + j];
        }
    }
}

void normalize_colwise(double *X, int rows, int cols)
{
    int i, j;
    double l2_norm, *x;

    for(i = 0; i < cols; i++) {
        x = d_viewcol(X, i, rows);
        l2_norm = norm(x, rows);

        /* if norm is eps 0, then continue */
        if(fabs(l2_norm) < 1e-7) {
            continue;
        }

        for(j = 0; j < rows; j++) {
            x[j] /= l2_norm;
        }
    }
}

double norm(double *X, int N)
{
    int i;
    double l2_norm = 0.0;

    for(i = 0; i < N; i++) {
        l2_norm += X[i] * X[i];
    }

    l2_norm = sqrt(l2_norm);

    return l2_norm;
}

void subtract_rowwise(double *X, double *Y, int rows, int cols)
{
    int i, j;

    for(i = 0; i < cols; i++) {
        for(j = 0; j < rows; j++) {
            X[idx2d(j, i, rows)] -= Y[i];
        }
    }
}

void bag_of_words(int *X, double *bag_X, int N, int max_len)
{
    int i;
    double l2_norm;
    
    for(i = 0; i < max_len; i++) {
        bag_X[i] = 0.0;
    }

    for(i = 0; i < N; i++) {
        bag_X[X[i]] += 1;
    }

    l2_norm = norm(bag_X, max_len);

    for(i = 0; i < max_len; i++) {
        bag_X[i] /= l2_norm;
    }
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
    int i, c1, c2, max_wins, max_class_idx, *wins;
    double *coef_col;
    wins = (int *)calloc(svm->num_classes, sizeof(int));
    max_wins = max_class_idx = 0;

    for(i = 0; i < svm->num_clfs; i++) {
        class_idxs(i, svm->num_classes, &c1, &c2);
        coef_col = d_viewcol(svm->coefs, i, svm->num_clfs);
        if(d_dot(coef_col, X, svm->num_features) + svm->intercepts[i] > 0) {
            wins[c1]++;
        }
        else {
            wins[c2]++;
        }
    }

    for(i = 0; i < svm->num_classes; i++) {
        if(wins[i] > max_wins) {
            max_wins = wins[i];
            max_class_idx = i;
        }
    }

    return max_class_idx;
}
