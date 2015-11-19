#include "nnu_generator.h"
#include "nnu_storage.h"

int main()
{
    int alpha = 10;
    int beta = 10;

    /* Generate standalone nnu locally */
    generate_nnu("../../../data/D750_hog.csv", "nnu.h", alpha, beta, mini);

    return 0;
}
