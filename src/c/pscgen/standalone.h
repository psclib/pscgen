/* Static version of Pipeline -- used for standalone embedded apps */
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


typedef uint32_t word_t;
enum { WORD_SIZE = sizeof(word_t) * 8 };

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

int get_bit(word_t *data, int b)
{ 
    return data[bindex(b)] & (1 << (boffset(b)));
}

void clear_all_bit(word_t *data, int N)
{
    memset(data, 0, (N/32 + 1) * sizeof(word_t));
}

int idx2d(int i, int j, int rows)
{
    return j * rows + i;
}


int idx3d(int i, int j, int k, int rows, int cols)
{
    return i * rows * cols + j * rows + k;
}

double* d_viewcol(double *mat, int col, int rows)
{
    return mat + idx2d(0, col, rows);
}


void dmm_prod(double *A, double *B, double *C, int A_rows, int A_cols,
              int B_rows, int B_cols)
{
    int i, j, k;

    for(i = 0; i < A_rows; i++) {
        for(j = 0; j < B_cols; j++) {
            for(k = 0; k < A_cols; k++) {
                C[idx2d(i, j, A_rows)] += A[idx2d(i, k , A_rows)] *
                                          B[idx2d(k, j, B_rows)];
            }
        }
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

void subtract_rowwise(double *X, double *Y, int rows, int cols)
{
    int i, j;

    for(i = 0; i < cols; i++) {
        for(j = 0; j < rows; j++) {
            X[idx2d(j, i, rows)] -= Y[i];
        }
    }
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


typedef enum
{
    half,
    mini,
    micro,
    nano,
    two_mini,
    four_micro
} Storage_Scheme;


uint8_t float_to_nano(float i)
{
    uint8_t ret = 0;
    int mask = 0x007f0000;
    int fi, *fi_ptr;
    if(i < 0) {
        ret |= 1 << 1;
        i = -i;
    }

    i += 1;
    fi_ptr = (int *)&i;
    fi = *fi_ptr;
    ret |= (mask & fi) >> 22;

    return ret;
}

float nano_to_float(uint8_t y)
{
    float *f;
    int s;
    int b = 0x3f800000;

    if(y == 0) {
        return 0.0;
    }

    s = (y >> 1) << 31;
    b |= ((y << 1) >> 1) << 22;
    f = (float *)&b;
    *f = *f - 1;

    if(s == 0){
        return *f;    
    }else{
        return -*f;    
    }
}

uint8_t float_to_micro(float i)
{
    uint8_t ret = 0;
    int mask = 0x007f0000;
    int fi, *fi_ptr;

    if(i < 0){
        ret |= 1 << 3;
        i = -i;
    }

    i += 1;
    fi_ptr = (int *)&i;
    fi = *fi_ptr;
    ret |= (mask & fi) >> 20;

    return ret;
}

float micro_to_float(uint8_t y)
{
    float *f;
    int b = 0x3f800000;
    int s;

    if(y == 0) {
        return 0.0;
    }

    s = (y >> 3) << 31;
    b |= ((y << 1) >> 1) << 20;
    f = (float *)&b;
    *f = *f - 1;

    if(s == 0){
        return *f;    
    }else{
        return -*f;    
    }
}



uint8_t float_to_mini(float i)
{
    int fi, *fi_ptr;
    int mask = 0x007f0000;
    uint8_t ret = 0;

    if(i < 0) {
        ret |= 1 << 7;
        i = -i;
    }

    i += 1;
    fi_ptr = (int *)&i;
    fi = *fi_ptr;

    ret |= (mask & fi) >> 16;

    return ret;
}

float mini_to_float(uint8_t y)
{
    float *f;
    int s;
    int b = 0x3f800000;

    if(y == 0) {
        return 0.0;
    }

    s = (y >> 7) << 31;
    b |= ((y << 1) >> 1) << 16;
    f = (float *)&b;
    *f = *f - 1;

    if(s == 0) {
      return *f;    
    }
    else{
      return -*f;    
    }
}

static uint32_t  half_to_float_I(uint16_t y)
{
    int s = (y >> 15) & 0x00000001;  /* sign */
    int e = (y >> 10) & 0x0000001f;  /* exponent */
    int f =  y        & 0x000003ff;  /* fraction */

    /* need to handle 7c00 INF and fc00 -INF? */
    if (e == 0) {
        /* need to handle +-0 case f==0 or f=0x8000? */
        if (f == 0)  /* Plus or minus zero */
            return s << 31;
        else {       /* Denormalized number -- renormalize it */
            while (!(f & 0x00000400)) {
                f <<= 1;
                e -=  1;
            }
            e += 1;
            f &= ~0x00000400;
        }
    } else if (e == 31) {
        /* Inf */
        if (f == 0)
            return (s << 31) | 0x7f800000;
        /* NaN */
        else
            return (s << 31) | 0x7f800000 | (f << 13);
    }

    e = e + (127 - 15);
    f = f << 13;

    return ((s << 31) | (e << 23) | f);
}

static uint16_t float_to_half_I(uint32_t i)
{
    int s =  (i >> 16) & 0x00008000;                   /* sign */
    int e = ((i >> 23) & 0x000000ff) - (127 - 15);     /* exponent */
    int f =   i        & 0x007fffff;                   /* fraction */

    /* need to handle NaNs and Inf? */
    if (e <= 0) {
        if (e < -10) {
            if (s)                                             /* handle -0.0 */
               return 0x8000;
            else
               return 0;
        }
        f = (f | 0x00800000) >> (1 - e);
        return s | (f >> 13);
    } else if (e == 0xff - (127 - 15)) {
        if (f == 0)                                             /* Inf */
            return s | 0x7c00;
        else {                                                  /* NAN */
            f >>= 13;
            return s | 0x7c00 | f | (f == 0);
        }
    } else {
        if (e > 30)                                             /* Overflow */
            return s | 0x7c00;
        return s | (e << 10) | (f >> 13);
    }
}

float half_to_float(uint16_t y)
{
    union { float f; uint32_t i; } v;
    v.i = half_to_float_I(y);
    return v.f;
}

uint16_t float_to_half(float i)
{
    union { float f; uint32_t i; } v;
    v.f = i;
    return float_to_half_I(v.i);
}


int storage_gamma_pow(Storage_Scheme s)
{
    switch (s) {
        case half: return 16;
        case mini: return 8;
        case micro: return 4;
        case nano: return 2;
        case two_mini: return 16;
        case four_micro: return 16;
    }

    return -1;

}

int storage_stride(Storage_Scheme s)
{
    switch (s) {
        case half: return 1;
        case mini: return 1;
        case micro: return 1;
        case nano: return 1;
        case two_mini: return 2;
        case four_micro: return 4;
    }

    return -1;
}

const char* print_storage(Storage_Scheme s)
{
    switch (s) {
        case half: return "half";
        case mini: return "mini";
        case micro: return "micro";
        case nano: return "nano";
        case two_mini: return "two_mini";
        case four_micro: return "four_micro";
    }

    return "";
}

uint16_t float_to_storage(float i, Storage_Scheme s)
{
    switch (s) {
        case half: return float_to_half(i);
        case mini: return float_to_mini(i);
        case micro: return float_to_micro(i);
        case nano: return float_to_nano(i);
        case two_mini: return float_to_mini(i);
        case four_micro: return float_to_micro(i);
    }

    return -1;
}

void storage_to_float(float *i, uint16_t y, Storage_Scheme s)
{
    switch (s) {
        case half:
            i[0] = half_to_float(y);
            break;
        case mini:
            i[0] = mini_to_float(y);
            break;
        case micro:
            i[0] = micro_to_float(y);
            break;
        case nano:
            i[0] = nano_to_float(y);
            break;
        case two_mini: 
            i[0] = mini_to_float(y >> 8);
            i[1] = mini_to_float(y);
            break;
        case four_micro: 
            i[0] = micro_to_float(y >> 12);
            i[1] = micro_to_float(y >> 8);
            i[2] = micro_to_float(y >> 4);
            i[3] = micro_to_float(y);
            break;
    }
}


/* NNU dictionary */
typedef struct NNUDictionary {
    int alpha; /* height of tables */
    int beta;  /* width of tables */
    int gamma; /* depth of tables */
    Storage_Scheme storage; /*float representation of each index */
    
    uint16_t *tables; /* nnu lookup tables (stores candidates)*/
    double *D; /* learned dictionary */
    double *D_mean; /*colwise mean of D */
    int D_rows; /* rows in ldict */
    int D_cols; /* cols in ldict */
    double *Vt; /* Vt from SVD(D) -- taking alpha columns */
    double *VD; /* dot(Vt, d) */
} NNUDictionary;

/* Computes the max dot product from candidate set with input sample x */
inline void compute_max_dot_set(double *max_coeff, int *max_idx, int *total_ab,
                                double *D, double *x, int *candidate_set,
                                int D_rows, int N)
{
    int i;
    double tmp_coeff = 0.0;
    *max_coeff = 0.0;
    (*total_ab) += N;

	for(i = 0; i < N; i++) {
        tmp_coeff = d_dot(x, d_viewcol(D, candidate_set[i], D_rows), D_rows);
        tmp_coeff = fabs(tmp_coeff);
        if(tmp_coeff > *max_coeff) {
            *max_coeff = tmp_coeff;
            *max_idx = candidate_set[i];
        }
    }
}

/* NNU candidate lookup using the generated tables */
void atom_lookup(NNUDictionary *dict, double *x, word_t *atom_idxs,
                 int *candidate_set, int *N, int alpha, int beta, int s_stride)
{
    int i, j, table_idx, table_key, shift_amount, shift_bits;
    uint16_t *beta_neighbors;
    *N = 0;
    
    for(i = 0; i < alpha; i++) {
        table_key = 0;
        for(j = 0; j < s_stride; j++) {
            shift_amount = (16 / s_stride) * (s_stride - j - 1);
            shift_bits = float_to_storage(x[i*s_stride + j], dict->storage);
            table_key |= (shift_bits <<  shift_amount);
        }
        table_idx = idx3d(i, table_key, 0, dict->beta, dict->gamma);
        beta_neighbors = &dict->tables[table_idx];
        for(j = 0; j < beta; j++) {
            if(get_bit(atom_idxs, beta_neighbors[j]) == 0) {
                set_bit(atom_idxs, beta_neighbors[j]);
                candidate_set[*N] = beta_neighbors[j];
                (*N)++;
            }
        }
    }
}

/* NNU lookup for input vector X */
int nnu(NNUDictionary *dict, double *X, int X_rows)
{
    int N;
    int max_idx = 0;
    int total_ab = 0;
    int D_rows = dict->D_rows;
    int s_stride = storage_stride(dict->storage);
    int X_cols = 1;  /* fixes X_cols to single vector case */
    double max_coeff = 0.0;

    word_t atom_idxs[ATOMS/32 + 1] = {0};
    int candidate_set[ALPHA*BETA] = {0};
    double *D = dict->D;
    double VX[ALPHA*S_STRIDE] = {0};

    dmm_prod(dict->Vt, X, VX, dict->alpha*s_stride, dict->D_rows, X_rows,
             X_cols); 
    atom_lookup(dict, d_viewcol(VX, 0, dict->alpha*s_stride), atom_idxs,
                candidate_set, &N, ALPHA, BETA, s_stride);
    compute_max_dot_set(&max_coeff, &max_idx, &total_ab, D,
                        d_viewcol(X, 0, X_rows), candidate_set, D_rows, N);

	return max_idx;
}

/* linear SVM */
typedef struct SVM {
    int num_features;
    int num_classes;
    int num_clfs;
    
    int *wins;
    double *coefs;
    double *intercepts;
} SVM;

void class_idxs(int idx, int num_classes, int *c1, int *c2)
{
    int i, j, curr_idx;

    curr_idx = 0;
    for(i = 0; i < num_classes; i++) {
        for(j = i + 1; j < num_classes; j++) {
            if(idx == curr_idx) {
                *c1 = i;
                *c2 = j;
                return;
            }
            curr_idx++;
        }
    }
}

int classify(double *X, SVM *svm)
{
    int i, c1, c2, max_wins, max_class_idx;
    double *coef_col;

    c1 = c2 = max_wins = max_class_idx = 0;

    /* Clear wins */
    memset(svm->wins, 0, svm->num_classes);

    /* Do one v. one classification */
    for(i = 0; i < svm->num_clfs; i++) {
        class_idxs(i, svm->num_classes, &c1, &c2);
        coef_col = d_viewcol(svm->coefs, i, svm->num_clfs);
        if(d_dot(coef_col, X, svm->num_features) + svm->intercepts[i] > 0) {
            svm->wins[c1]++;
        }
        else {
            svm->wins[c2]++;
        }
    }

    /* Find winner */
    for(i = 0; i < svm->num_classes; i++) {
        if(svm->wins[i] > max_wins) {
            max_wins = svm->wins[i];
            max_class_idx = i;
        }
    }

    return max_class_idx;
}



typedef struct Pipeline {
    int ws;
    int ss;
    int num_windows;
    int N;

    double *window_X;
    double *bag_X;
    NNUDictionary *nnu;
    SVM *svm;
} Pipeline;

int classification_pipeline(double *X, Pipeline *pipeline)
{
    int i, idx;
    double l2_norm;

    memset(pipeline->bag_X, 0, pipeline->nnu->D_cols);

    sliding_window(X, pipeline->window_X, pipeline->N, pipeline->ws,
                   pipeline->ss);
    normalize_colwise(pipeline->window_X, pipeline->ws, pipeline->num_windows);
    subtract_rowwise(pipeline->window_X, pipeline->nnu->D_mean, pipeline->ws,
                     pipeline->num_windows);

    for(i = 0; i < pipeline->num_windows; i++) {
        idx = nnu(pipeline->nnu,
                  d_viewcol(pipeline->window_X, i, pipeline->ws),
                  pipeline->ws);
        pipeline->bag_X[idx] += 1.0;
    }

    
    l2_norm = norm(pipeline->bag_X, pipeline->nnu->D_cols);

    for(i = 0; i < pipeline->nnu->D_cols; i++) {
        pipeline->bag_X[i] /= l2_norm;
    }

    return classify(pipeline->bag_X, pipeline->svm);
}
