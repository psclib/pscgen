#include "stdio.h"
#include "pscgen.h"
#include "csv.h"

int main(int argc, char *argv[])
{
    int alpha = 10;
    int beta = 10;
    NNUDictionary *dict = new_dict(alpha, beta);
    delete_dict(dict);

    int input_rows, input_cols;
    double *input_buf;

    read_csv("test.csv", ",", &input_rows, &input_cols, &input_buf);
    print_mat(input_buf, input_rows, input_cols);

    free(input_buf);
    return 0;
}
