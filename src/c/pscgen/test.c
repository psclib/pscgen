#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nnu.h"
#include "util.h"
#include "linalg/linalg.h"

int main(int argc, char *argv[])
{
    int alpha = 10;
    int beta = 10;
    NNUDictionary *dict = new_dict(alpha, beta, "test.csv", ",");

    /* printf("%f\n", dict->tables[idx3d(0, 0, 0, alpha, RANGE_16)]); */
    printf("%d,%d,%d\n", dict->alpha, dict->beta, RANGE_16);

    delete_dict(dict);

    return 0;
}
