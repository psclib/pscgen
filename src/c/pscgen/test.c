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
    delete_dict(dict);

    /* unsigned long long int i; */
    /* float f; */
    /* int vals = 0; */

    /* /1* for(i = 0; i < 4294967295; i++) { *1/ */
    /* /1*     f = *(float*)(&i); *1/ */
    /* /1*     if(f > 0 && f < 1){ *1/ */
    /* /1*         vals++; *1/ */
    /* /1*         /2* printf("%f\n", f); *2/ *1/ */
    /* /1*     } *1/ */
    /* /1* } *1/ */

    /* for(i = 0; i < 65535; i++) { */
    /*     unsigned int y = __gnu_h2f_ieee(i); */
    /*     float z = *(float*)(&y); */
    /*     z = half_to_float(i); */
    /*     if(z > 0 && z < 1){ */ 
    /*         vals++; */
    /*         printf("%g\n", z); */
    /*     } */
    /* } */

    return 0;
}
