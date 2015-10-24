#include "util.h"

static double* _dvalues;

int d_argsort_compare(const void* a, const void* b)
{
    if(_dvalues[*(int*)b] > _dvalues[*(int*)a]) return 1;
    if(_dvalues[*(int*)b] < _dvalues[*(int*)a]) return -1;
    return 0;
}


void read_csv(const char *filepath, const char *delimiters,
              double **buf, int *rows, int *cols)
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

inline double* new_dvec(int N)
{
    int i;
    double *vec = malloc(sizeof(double) * N);

    for(i = 0; i < N; i++) {
        vec[i] = 0.0;
    }

    return vec;
}

inline void zero_dvec(double *vec, int N)
{
    int i;

    for(i = 0; i < N; i++) {
        vec[i] = 0.0;
    }
}


inline int idx2d(int i, int j, int cols)
{
    return i * cols + j;
}


inline int idx3d(int i, int j, int k, int rows, int cols) {
    return i * rows * cols + j * rows + k;
}

int* d_argsort(double* vec, int N)
{
    int i;

    //create idx array in sorted order
    int *idxs = malloc(sizeof(int) * N);
    for(i = 0; i < N; i++) {
        idxs[i] = i;
    }

    //set scoped pointer to vec
    _dvalues = vec;

    //perform qsort
    qsort(idxs, N, sizeof(int), d_argsort_compare);

    return idxs;
}


void d_transpose_inplace(double *mat, int rows, int cols)
{
    int start;
    for(start = 0; start <= cols * rows - 1; ++start) {
        int next = start;
        int i = 0;

        do {
            ++i;
            next = (next % rows) * cols + next / rows;
        } while (next > start);

    if (next >= start && i != 1) {
        const double tmp = mat[start];
        next = start;
        do {
            i = (next % rows) * cols + next / rows;
            mat[next] = (i == start) ? tmp : mat[i];
            next = i;
        } while (next > start);
      }
    }
}


void print_mat(double *buf, int rows, int cols)
{
    int i, j;
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols-1; j++) {
            printf("%2.2f, ", buf[i*cols + j]);
        }
        printf("%2.2f", buf[idx2d(i, j, cols)]);
        printf("\n");
    
    }
}
