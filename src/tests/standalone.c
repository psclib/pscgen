#include <stdio.h>
#include "util.h"
#include "nnu_5_5.h"

int main()
{
    double *A;
    int rA, cA;

    read_csv("/home/brad/data/voice1.csv", ",", &A, &rA, &cA);
    int ret = classification_pipeline(A, cA, &pipeline);
    printf("%d\n", ret);

    return 0;
}
