#ifndef STANDALONE
#include "util.h"
#endif


int d_argsort_compare(const void* a, const void* b, void *thunk)
{
    double *dvalues = (double *)thunk;

    if(dvalues[*(int*)b] < dvalues[*(int*)a]) return 1;
    if(dvalues[*(int*)b] > dvalues[*(int*)a]) return -1;
    return 0;
}


void read_csv(const char *filepath, const char *delimiters,
              double **buf, int *rows, int *cols)
{
    int nchars = 10000000;
    int nbytes = sizeof(char) * nchars;
    int i = 0;
    int j = 0;
    FILE *fp = fopen(filepath, "r");
    char *line = (char *)malloc(nbytes);
    char *item;

    /* initialize rows/cols */
    *rows = 0;
    *cols = 0;

    /* get dimensions of csv file */
    while(fgets(line, nchars, fp) != NULL) {
        /* count cols for first rows */
        if((*cols) == 0) {
            item = strtok(line, delimiters);
            while(item != NULL) {
                (*cols)++;
                item = strtok(NULL, delimiters);
            }
        }
        (*rows)++;
    }

    /* allocate the buffer */
    *buf = (double *)malloc(sizeof(double) * (*rows) * (*cols));

    /* rewind fp to start of file */
    fseek(fp, 0, SEEK_SET);

    /* read into buffer */
    while(fgets(line, nchars, fp) != NULL) {
        item = strtok(line, delimiters);
        while(item != NULL) {
            (*buf)[idx2d(i, j, *rows)] = atof(item);
            item = strtok(NULL, delimiters);
            j++;
        }
        j = 0;
        i++;
    }

    /* clean-up */
    fclose(fp);
    free(line);
}

double* new_dvec(int N)
{
    int i;
    double *vec = (double *)malloc(sizeof(double) * N);

    for(i = 0; i < N; i++) {
        vec[i] = 0.0;
    }

    return vec;
}

void zero_dvec(double *vec, int N)
{
    int i;

    for(i = 0; i < N; i++) {
        vec[i] = 0.0;
    }
}


int idx2d(int i, int j, int rows)
{
    return j * rows + i;
}


int idx3d(int i, int j, int k, int rows, int cols)
{
    return i * rows * cols + j * rows + k;
}

void d_argsort(double *vec, int *idxs, int N)
{
    int i;
    for(i = 0; i < N; i++) {
        idxs[i] = i;
    }

    /* perform qsort */
    qsort_r(idxs, N, sizeof(int), d_argsort_compare, (void *)vec);
}


/* bit-set fuctions */
word_t* bit_vector(int N)
{
    return (word_t *)calloc(N / 32 + 1, sizeof(word_t));
}

int bindex(int b)
{
    return b / WORD_SIZE;
}
int boffset(int b)
{
    return b % WORD_SIZE;
}

void set_bit(word_t *data, int b)
{ 
    data[bindex(b)] |= 1 << (boffset(b)); 
}

void clear_bit(word_t *data, int b)
{ 
    data[bindex(b)] &= ~(1 << (boffset(b)));
}

int get_bit(word_t *data, int b)
{ 
    return data[bindex(b)] & (1 << (boffset(b)));
}

void clear_all_bit(word_t *data, int N)
{
    memset(data, 0, (N/32 + 1) * sizeof(word_t));
}

double* d_transpose(double *mat, int rows, int cols)
{
    double *mat_t = (double *)malloc(sizeof(double) * rows * cols);
    int i, j;
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            mat_t[idx2d(j, i, cols)] = mat[idx2d(i, j, rows)];
        }
    }

    return mat_t;
}

double* d_trim(double* mat, int rows, int new_rows, int new_cols)
{
    double *ret_mat = (double *)malloc(sizeof(double) * new_rows * new_cols);
    int i, j;
    for(i = 0; i < new_rows; i++) {
        for(j = 0; j < new_cols; j++) {
            ret_mat[idx2d(i, j, new_rows)] = mat[idx2d(i, j, rows)];
        }
    }

    return ret_mat;
}

double* d_viewcol(double *mat, int col, int rows)
{
    return mat + idx2d(0, col, rows);
}

void print_mat(double *buf, int rows, int cols)
{
    int i, j;
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols-1; j++) {
            printf("%2.6f,", buf[idx2d(i, j, rows)]);
        }
        printf("%2.6f", buf[idx2d(i, j, rows)]);
        printf("\n");
    
    }
}

void print_mat_i(int *buf, int rows, int cols)
{
    int i, j;
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols-1; j++) {
            printf("%d,", buf[idx2d(i, j, rows)]);
        }
        printf("%d", buf[idx2d(i, j, rows)]);
        printf("\n");
    
    }
}

double d_dot(double *X, double *Y, int N)
{
    int i;
    double ret = 0;

    for(i = 0; i < N; i++) {
        ret += X[i] * Y[i]; 
    }

    return ret;
}

/* naive matrix-matrix product implementation */
double* dmm_prod(double *A, double *B, int A_rows, int A_cols, int B_rows,
                 int B_cols)
{
    int i, j, k;
    double *ret = (double *)calloc(A_rows*B_cols, sizeof(double));
    for(i = 0; i < A_rows; i++) {
        for(j = 0; j < B_cols; j++) {
            for(k = 0; k < A_cols; k++) {
                ret[idx2d(i, j, A_rows)] += A[idx2d(i, k , A_rows)] *
                                            B[idx2d(k, j, B_rows)];
            }
        }
    }

    return ret;
}


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

void ith_window(double *X, double *window_X, int win_num, int ws, int ss)
{
    int i, s_amount;
    s_amount = ss * win_num;

    for(i = 0; i < ws; i++) {
        window_X[i] = X[i + s_amount];
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

void subtract_colwise(double *X, double *Y, int rows, int cols)
{
    int i, j;

    for(i = 0; i < cols; i++) {
        for(j = 0; j < rows; j++) {
            X[idx2d(j, i, rows)] -= Y[i];
        }
    }
}


double* mean_colwise(double *X, int rows, int cols)
{
    int i, j;
    double *mean_X = (double *)calloc(cols, sizeof(double));

    for(i = 0; i < cols; i++) {
        for(j = 0; j < rows; j++) {
            mean_X[i] += X[idx2d(j, i, rows)];
        }
    }

    for(i = 0; i < cols; i++) {
        mean_X[i] /= rows;
    }

    return mean_X;
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


int ipow(int base, int exp)
{
    int result = 1;
    while (exp)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }

    return result;
}

int rand_int(int n)
{
    int limit = RAND_MAX - RAND_MAX % n;
    int rnd;

    do {
        rnd = rand();
    } while (rnd >= limit);

    return rnd % n;
}

void shuffle(int *array, int n)
{
    int i, j, tmp;

    for (i = n - 1; i > 0; i--) {
        j = rand_int(i + 1);
        tmp = array[j];
        array[j] = array[i];
        array[i] = tmp;
    }
}

double randn (double mu, double sigma)
{
    double U1, U2, W, mult;
    static double X1, X2;
    static int call = 0;

    if (call == 1) {
        call = !call;
        return (mu + sigma * (double) X2);
    }

    do {
        U1 = -1 + ((double) rand () / RAND_MAX) * 2;
        U2 = -1 + ((double) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    } while (W >= 1 || W == 0);

    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;

    return (mu + sigma * (double) X1);
}

struct timespec t_diff(struct timespec start, struct timespec end)
{
	struct timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}
