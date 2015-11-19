/* Static version of NNU -- used for standalone embedded apps */
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


/* NNU dictionary */
typedef struct NNUDictionary {
    int alpha; /* height of tables */
    int beta;  /* width of tables */
    int gamma; /* depth of tables */
    Storage_Scheme storage; /*float representation of each index */
    
    uint16_t *tables; /* nnu lookup tables (stores candidates)*/
    double *D; /* learned dictionary */
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
