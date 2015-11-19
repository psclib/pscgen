#include "nnu_generator.h"
#include "nnu_storage.h"

int main(int argc, char **argv)
{
    int alpha = 10;
    int beta = 10;

    /* Generate standalone nnu locally */
    generate_nnu(argv[1], "nnu.h", alpha, beta, mini);

    return 0;
}
