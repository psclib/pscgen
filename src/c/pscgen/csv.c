#include "csv.h"

void read_csv(const char *filepath, const char *delimiters,
              int *rows, int *cols, double **buf)
{
    int nchars = 10000;
    int nbytes = sizeof(char) * nchars;
    int idx = 0;
    FILE *fp = fopen(filepath, "r");
    char *line = malloc(nbytes);
    char *item;

    //initialize rows/cols
    *rows = 0;
    *cols = 0;

    //get dimensions of csv file
    while(fgets(line, nchars, fp) != NULL) {
        //count cols for first rows
        if((*cols) == 0) {
            item = strtok(line, delimiters);
            while(item != NULL) {
                (*cols)++;
                item = strtok(NULL, delimiters);
            }
        }
        (*rows)++;
    }

    //allocate the buffer
    *buf = malloc(sizeof(double) * (*rows) * (*cols));

    //rewind fp to start of file
    fseek(fp, 0, SEEK_SET);

    //read into buffer
    while(fgets(line, nchars, fp) != NULL) {
        item = strtok(line, delimiters);
        while(item != NULL) {
            (*buf)[idx] = atof(item);
            idx++;
            item = strtok(NULL, delimiters);
        }
    }

    //clean-up
    fclose(fp);
    free(line);
}


void print_mat(double *buf, int rows, int cols)
{
    int i, j;
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols-1; j++) {
            printf("%2.2f, ", buf[i*cols + j]);
        }
        printf("%2.2f", buf[i*cols + j]);
        printf("\n");
    }
}
