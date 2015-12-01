#include <stdio.h>
#include "nnu.h"

int main()
{
    double A[200] = {  0.00999051,  0.05994308,  0.00749289,  0.05494782,  0.09740751,
        0.0349668 ,  0.04745494,  0.04495731,  0.02497628,  0.08991462,
        0.0524502 ,  0.01498577,  0.0524502 ,  0.06244071,  0.0349668 ,
        0.06743597,  0.03746443,  0.09241225,  0.09740751,  0.10739802,
        0.09740751,  0.08741699,  0.03246917,  0.10739802,  0.01998103,
        0.0349668 ,  0.0174834 ,  0.06244071,  0.11489091,  0.0524502 ,
        0.05744545,  0.06993359,  0.11489091,  0.14236482,  0.0524502 ,
        0.03246917,  0.10490039,  0.03746443,  0.04745494,  0.00499526,
        0.01998103,  0.04245968,  0.00249763,  0.00999051,  0.01248814,
        0.05994308,  0.06493834,  0.02247866,  0.07492885,  0.02497628,
        0.03996205,  0.09241225,  0.06993359,  0.1498577 ,  0.03746443,
        0.08242174,  0.04245968,  0.11738853,  0.07243122,  0.01498577,
        0.07742648,  0.04495731,  0.05994308,  0.19981027,  0.0349668 ,
        0.04495731,  0.04745494,  0.01498577,  0.00749289,  0.12488142,
        0.09740751,  0.09990514,  0.02747391,  0.05744545,  0.01248814,
        0.04995257,  0.01998103,  0.05994308,  0.04745494,  0.09990514,
        0.04745494,  0.05994308,  0.03996205,  0.08991462,  0.07492885,
        0.06493834,  0.02247866,  0.07492885,  0.0349668 ,  0.0174834 ,
        0.04745494,  0.0174834 ,  0.02997154,  0.09241225,  0.03746443,
        0.04995257,  0.02247866,  0.05994308,  0.01998103,  0.03246917,
        0.0349668 ,  0.08242174,  0.00999051,  0.00749289,  0.07492885,
        0.03746443,  0.05494782,  0.00249763,  0.08991462,  0.15235533,
        0.0174834 ,  0.01498577,  0.08242174,  0.01248814,  0.13986719,
        0.06743597,  0.05744545,  0.08991462,  0.0524502 ,  0.02497628,
        0.0174834 ,  0.06993359,  0.04745494,  0.01498577,  0.02247866,
        0.0174834 ,  0.03746443,  0.00749289,  0.00999051,  0.0349668 ,
        0.07243122,  0.07492885,  0.00999051,  0.11738853,  0.01498577,
        0.01998103,  0.00249763,  0.00249763,  0.06493834,  0.04745494,
        0.08491936,  0.04995257,  0.00749289,  0.07243122,  0.07992411,
        0.02997154,  0.1323743 ,  0.0524502 ,  0.03246917,  0.08491936,
        0.10490039,  0.00249763,  0.07992411,  0.09241225,  0.09490988,
        0.04745494,  0.14736007,  0.00249763,  0.06244071,  0.19481501,
        0.02997154,  0.08491936,  0.00499526,  0.02747391,  0.02997154,
        0.00999051,  0.01248814,  0.05994308,  0.07742648,  0.03246917,
        0.07742648,  0.12987668,  0.02747391,  0.1323743 ,  0.02997154,
        0.1323743 ,  0.0524502 ,  0.08741699,  0.14236482,  0.03746443,
        0.18232687,  0.01498577,  0.07742648,  0.07492885,  0.18732213,
        0.08741699,  0.06743597,  0.00999051,  0.1498577 ,  0.05994308,
        0.11988616,  0.02497628,  0.08242174,  0.        ,  0.00499526,
        0.04495731,  0.04245968,  0.05494782,  0.02747391,  0.08491936};

    int ret = classification_pipeline(A, 200, &pipeline);
    printf("%d\n", ret);

    return 0;
}
