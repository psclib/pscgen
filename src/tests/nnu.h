#define STANDALONE
#define ALPHA 5
#define BETA 5
#define ATOMS 10
#define S_STRIDE 1
#ifndef NNU_STORAGE_H
#define NNU_STORAGE_H

#include <stdio.h>
#include <inttypes.h>

typedef enum
{
    half,
    mini,
    micro,
    nano,
    two_mini,
    four_micro
} Storage_Scheme;

const char* print_storage(Storage_Scheme);
int storage_gamma_pow(Storage_Scheme);
int storage_stride(Storage_Scheme);
uint16_t float_to_storage(float, Storage_Scheme);
void storage_to_float(float*, uint16_t, Storage_Scheme);

uint8_t float_to_nano(float i);
float nano_to_float(uint8_t y);
uint8_t float_to_micro(float i);
float micro_to_float(uint8_t y);
float half_to_float(uint16_t);
uint16_t float_to_half(float);
uint8_t float_to_mini(float i);
float mini_to_float(uint8_t y);

#endif /* NNU_STORAGE_H */

#ifndef STANDALONE
#include "nnu_storage.h"
#endif

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

uint16_t nnu_table[6400] = {2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,2,3,9,0,5,6,1,5,4,7,6,1,5,4,7,6,1,5,4,7,6,1,5,7,4,6,1,7,5,4,6,7,1,5,4,7,6,1,5,4,7,6,1,5,4,7,6,1,5,8,7,6,1,8,5,7,6,8,1,0,7,8,0,6,1,7,8,0,6,1,7,8,0,6,1,7,8,0,6,1,7,8,0,6,1,8,7,0,6,1,8,0,7,6,1,8,0,7,6,1,8,0,7,6,1,8,0,7,6,1,8,0,7,6,1,0,8,7,6,1,0,8,7,6,1,0,8,7,6,1,0,8,7,6,1,0,8,7,6,1,0,8,7,6,1,0,8,7,6,1,0,8,7,6,1,0,8,7,6,1,0,8,7,6,1,0,8,7,2,6,0,8,7,2,6,0,8,7,2,6,0,8,7,2,6,0,8,7,2,6,0,8,7,2,6,0,8,2,7,6,0,8,2,7,6,0,8,2,7,6,0,8,2,7,6,0,8,2,7,6,0,2,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,2,0,8,7,6,6,1,5,4,7,1,6,5,4,7,1,5,6,4,7,5,1,6,4,7,5,4,1,6,7,4,5,1,6,7,4,5,1,6,7,4,5,1,6,7,4,5,1,6,7,4,5,1,6,7,4,5,1,6,7,4,5,1,6,7,4,5,1,6,7,4,5,1,6,7,4,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,9,1,6,4,9,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,1,6,9,4,5,1,3,9,4,5,3,1,9,4,3,5,1,9,3,4,5,1,9,3,4,5,1,9,3,4,5,1,9,3,4,5,1,9,3,4,5,1,9,3,4,5,1,9,3,4,5,1,9,3,4,5,1,9,3,4,5,1,9,3,4,5,1,9,3,4,5,1,9,3,4,5,1,9,3,4,5,1,9,3,4,5,1,9,3,4,5,1,9,3,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,3,9,4,5,1,0,8,6,1,7,0,8,6,7,1,8,0,7,6,1,8,0,7,6,1,8,7,0,6,1,7,8,0,6,3,7,8,0,3,6,7,8,0,3,6,7,8,3,0,6,7,3,8,0,6,7,3,8,0,6,3,7,8,0,6,3,7,8,0,6,3,7,8,0,6,3,7,8,0,6,3,7,8,0,6,3,7,8,0,6,3,7,8,0,2,3,7,8,0,2,3,7,8,2,0,3,7,2,8,0,3,2,7,8,0,3,2,7,8,0,3,2,7,8,0,3,2,7,8,0,3,2,7,8,0,2,3,7,8,9,2,3,7,9,8,2,3,7,9,8,2,3,9,7,8,2,3,9,7,8,2,3,9,7,8,2,3,9,7,8,2,9,3,7,8,2,9,3,7,8,2,9,3,7,8,2,9,3,7,8,2,9,3,7,8,2,9,3,7,8,2,9,3,7,8,2,9,3,7,8,2,9,3,7,8,2,9,3,7,8,2,9,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,9,2,3,7,8,0,8,6,1,7,0,6,8,1,7,6,0,1,8,7,6,1,0,8,7,6,1,0,8,7,1,6,0,8,7,1,6,0,8,7,1,6,0,8,7,1,6,0,8,7,1,6,0,8,7,1,6,0,8,7,1,6,0,8,7,1,6,0,8,7,1,6,0,8,4,1,6,0,8,4,1,6,0,4,8,1,6,4,0,8,1,6,4,0,8,1,4,6,0,8,4,1,6,0,8,4,1,6,0,8,4,1,6,0,8,4,1,6,0,8,4,1,6,0,8,4,1,6,0,8,4,1,6,0,8,4,1,6,0,8,4,1,6,0,8,4,1,6,0,8,4,1,6,0,8,4,1,6,0,8,4,1,6,0,5,4,1,6,5,0,4,1,6,5,0,4,1,6,5,0,4,1,5,6,0,4,5,1,6,0,4,5,1,6,0,4,5,1,6,0,4,5,1,6,0,4,5,1,6,0,4,5,1,6,0,4,5,1,6,0,4,5,1,6,0,4,5,1,6,0,4,5,1,6,0,4,5,1,6,0,4,5,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,5,4,1,6,0,8,7,4,2,5,7,4,8,2,5,4,7,8,2,5,4,7,8,2,5,2,4,7,8,5,2,4,7,5,8,2,5,4,7,8,2,5,4,7,8,2,5,4,7,8,5,2,4,7,8,5,2,4,7,8,5,2,9,4,7,5,2,9,4,7,5,2,9,4,7,5,9,2,4,7,5,9,2,4,7,9,5,2,4,7,9,5,2,4,7,9,5,2,4,7,9,5,2,4,7,9,5,2,4,7,9,5,2,4,7,9,5,2,4,7,9,5,2,4,7,9,5,2,4,7,9,5,2,4,7,9,5,2,1,4,9,5,2,1,4,9,5,2,1,4,9,5,1,2,4,9,1,5,2,4,9,1,5,2,4,9,1,5,2,4,9,1,5,2,4,9,1,5,2,4,9,1,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,1,9,5,2,4,8,7,4,2,5,8,7,4,2,5,8,7,4,2,5,8,7,4,2,5,8,7,4,2,5,8,7,4,2,5,8,7,4,2,3,8,7,4,2,3,8,7,4,3,2,8,7,4,3,2,8,7,3,4,6,3,8,7,4,6,3,8,7,6,4,3,6,8,7,4,3,6,8,7,4,3,6,8,7,4,3,6,8,7,4,3,6,8,7,4,3,6,8,7,4,3,6,8,7,4,3,6,8,7,4,3,6,8,7,0,3,6,0,8,7,3,6,0,8,7,6,3,0,8,7,6,3,0,8,7,6,3,0,8,7,6,3,0,8,7,6,3,0,8,7,6,3,0,8,7,6,3,0,8,7,6,3,0,8,7,6,3,0,8,7,6,0,3,8,7,6,0,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,0,6,3,8,7,2,5,9,3,8,5,2,9,3,8,5,9,2,3,8,5,9,3,8,2,9,5,3,8,2,9,3,8,5,2,3,8,9,5,2,8,3,9,5,2,8,3,9,5,7,8,3,9,7,5,8,3,9,7,5,8,7,3,9,4,7,8,3,4,0,7,4,0,8,3,7,4,0,8,3,7,4,0,8,3,7,4,0,8,3,7,4,0,8,3,4,0,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,0,4,7,8,3,2,5,9,3,8,2,5,9,3,8,2,5,9,3,8,2,5,9,3,8,2,5,9,1,3,2,5,1,9,3,2,1,5,9,3,2,1,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,3,1,2,5,9,6,1,2,5,6,9,1,2,6,5,9,1,6,2,5,9,1,6,2,5,9,1,6,2,5,9,1,6,2,5,9,1,6,2,5,9,1,6,2,5,9,1,6,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9,6,1,2,5,9};
double D[960] = {-0.063812,0.010963,-0.061043,0.003572,-0.057976,0.009001,-0.052650,0.009088,-0.053329,0.002671,-0.052841,0.000210,-0.040138,0.011367,-0.045170,0.007313,-0.063232,0.027118,-0.062084,0.011864,-0.042813,-0.011766,-0.063286,0.007990,-0.057999,0.003520,-0.059483,0.018461,-0.070602,0.013065,-0.056709,0.022735,0.155197,0.161499,0.134491,0.151382,0.163330,0.154221,0.119797,0.123779,0.179810,0.148773,0.143509,0.145218,0.140411,0.145416,0.141449,0.158569,0.222534,0.212006,0.233521,0.218811,0.198273,0.143936,0.130692,0.168671,0.206635,0.135437,0.133514,0.215607,0.230133,0.162720,0.147522,0.247894,0.179810,0.162994,0.155273,0.187683,0.183746,0.143036,0.120193,0.145615,0.169510,0.160797,0.148407,0.131119,0.144684,0.160934,0.152084,0.141922,0.201508,0.180817,0.213440,0.216156,0.177673,0.120560,0.120651,0.176392,0.191132,0.130875,0.159088,0.240997,0.201721,0.140533,0.178986,0.279663,0.096649,-0.006517,0.094727,0.000364,0.038570,-0.003595,0.089752,0.002571,0.083221,0.008884,0.076996,-0.007699,0.003387,0.001570,0.086639,0.003806,0.056427,-0.000893,0.029884,-0.016777,0.088226,-0.000778,-0.034531,-0.015228,0.008516,-0.014085,-0.003529,-0.017035,0.008471,-0.022666,0.030668,-0.009983,0.091095,0.085938,0.135498,0.109985,0.086243,0.134644,0.307983,0.199829,0.120361,0.187561,0.306519,0.183777,0.123505,0.177368,0.261536,0.196838,0.090485,0.083466,0.121857,0.090790,0.080597,0.132019,0.248291,0.162170,0.151001,0.180725,0.233093,0.181580,0.135681,0.175720,0.266357,0.200806,0.089355,0.091278,0.138367,0.081268,0.084076,0.130676,0.305054,0.195801,0.121887,0.179688,0.308472,0.188965,0.123047,0.156067,0.274048,0.221680,0.082916,0.084808,0.126160,0.085144,0.078522,0.137207,0.243835,0.160217,0.141846,0.178406,0.252014,0.178772,0.124237,0.174744,0.265503,0.199646,0.046377,-0.003132,0.043328,0.000826,0.045191,-0.006910,0.039432,0.004222,0.041737,-0.005018,0.043971,-0.004413,0.053080,-0.014017,0.065878,-0.007056,0.061166,-0.004543,0.062857,0.001055,0.050148,0.009638,0.065458,0.008412,0.047365,-0.002205,0.043813,-0.008305,0.037086,-0.008164,0.023123,-0.002815,0.077955,0.096391,0.111491,0.077315,0.068034,0.086528,0.135030,0.088352,0.083679,0.102590,0.129235,0.099186,0.099128,0.121582,0.163869,0.109287,0.241726,0.162428,0.128235,0.129340,0.243245,0.257107,0.210002,0.221842,0.333713,0.209595,0.149943,0.180603,0.338406,0.211141,0.143687,0.210720,0.071320,0.084076,0.113749,0.084052,0.071479,0.094876,0.139984,0.090963,0.079709,0.094527,0.128805,0.105767,0.096514,0.103141,0.163879,0.110311,0.233100,0.162598,0.125878,0.123830,0.238308,0.261990,0.214030,0.207235,0.326755,0.212999,0.168715,0.204692,0.336453,0.219618,0.141463,0.205716,-0.118286,-0.078552,-0.076904,-0.030136,-0.046509,-0.014992,-0.015007,-0.013008,-0.046448,-0.022858,-0.061096,-0.024780,-0.050629,0.009949,-0.007805,-0.006634,0.004593,-0.012222,0.019287,-0.014053,0.046448,0.020691,0.027267,0.021606,0.086487,-0.015320,0.074951,0.009918,0.108337,0.049225,0.094543,0.077271,0.285645,0.130127,0.126099,0.201172,0.261963,0.232544,0.241455,0.322021,0.239990,0.250732,0.240356,0.156494,0.146240,0.136597,0.186279,0.187256,0.068359,0.089478,0.161865,0.155151,0.119019,0.075378,0.083069,0.064514,0.186035,0.148560,0.128296,0.061401,0.040894,0.075256,0.151001,0.178223,0.174072,0.083008,0.141602,0.219238,0.161011,0.133301,0.319824,0.395996,0.196777,0.329346,0.181396,0.159424,0.153076,0.129272,0.179565,0.142578,0.061279,0.081726,0.164185,0.153809,0.124695,0.092773,0.099182,0.073792,0.173096,0.159424,0.189331,0.078369,0.041962,0.047699,0.157837,0.248169,-0.102905,0.006145,-0.139526,0.052139,-0.191589,0.031265,-0.094421,0.023544,-0.107269,-0.010276,-0.036880,0.023516,-0.016411,0.019913,-0.041901,0.004660,-0.035837,0.000522,0.003455,-0.013040,0.026161,-0.013359,0.015638,-0.000971,0.018684,-0.011807,0.050632,-0.007925,0.017342,-0.002321,0.011640,-0.000197,0.107880,0.128845,0.217529,0.127594,0.079926,0.152100,0.288086,0.203064,0.112244,0.162903,0.268433,0.174500,0.097015,0.132874,0.245850,0.191711,0.096497,0.132690,0.182495,0.148987,0.115662,0.171326,0.249573,0.142639,0.113647,0.172729,0.270020,0.184753,0.105896,0.152222,0.238586,0.158356,0.120758,0.155701,0.210266,0.170166,0.125427,0.159241,0.246948,0.198486,0.113159,0.164795,0.245544,0.172852,0.101532,0.155090,0.261292,0.191650,0.128082,0.133972,0.190186,0.158081,0.124695,0.149841,0.203247,0.174194,0.142151,0.165466,0.237732,0.179321,0.132416,0.157898,0.223083,0.178040,-0.121399,-0.000604,-0.092407,0.006924,-0.069702,-0.012306,-0.112427,-0.008148,-0.068237,0.005592,-0.057983,-0.019989,-0.168945,-0.017868,-0.032715,-0.001529,-0.107910,-0.005707,-0.069946,-0.003435,-0.031342,0.000143,-0.017868,-0.000416,-0.006081,0.000032,-0.011696,-0.000186,-0.012856,-0.000150,-0.011963,0.000309,0.080078,0.088806,0.131714,0.097473,0.078491,0.105835,0.182983,0.129028,0.077148,0.112244,0.301758,0.262451,0.135498,0.141846,0.292236,0.246338,0.093872,0.107239,0.134277,0.113098,0.104492,0.119751,0.166504,0.142456,0.103088,0.120239,0.273193,0.296631,0.082825,0.120972,0.324219,0.292969,0.092346,0.094543,0.143555,0.101868,0.083069,0.106384,0.179932,0.124817,0.085327,0.132446,0.324951,0.276367,0.107605,0.143677,0.318359,0.266846,0.104187,0.113464,0.131714,0.118286,0.105469,0.120178,0.170654,0.136597,0.080505,0.115601,0.211060,0.221802,0.082214,0.123718,0.335938,0.302734,0.045258,0.006691,0.022659,-0.010925,0.019104,-0.018356,-0.002792,0.002609,-0.029831,0.011803,-0.051910,0.019257,-0.074524,0.000879,-0.046936,-0.007336,-0.137207,0.247437,-0.028046,-0.114319,-0.025726,-0.096802,-0.006599,-0.063782,0.005562,-0.011528,0.006874,0.011658,0.011185,0.034210,0.024384,0.057037,0.149292,0.105896,0.122131,0.147583,0.148315,0.219360,0.266113,0.183350,0.138672,0.130981,0.207397,0.246826,0.299805,0.159180,0.158081,0.137085,0.183472,0.121704,0.101318,0.080200,0.078247,0.095581,0.184937,0.249756,0.247437,0.215820,0.142456,0.121277,0.100220,0.125488,0.175659,0.253418,0.111450,0.097046,0.112183,0.158936,0.172852,0.234985,0.252930,0.185547,0.113647,0.111328,0.233765,0.289307,0.228394,0.171753,0.171509,0.132935,0.195190,0.130737,0.128418,0.114502,0.104980,0.130737,0.184570,0.240845,0.235840,0.174805,0.143188,0.111938,0.115540,0.100952,0.142456,0.241577,-0.030228,0.036041,0.063293,-0.018311,0.001310,0.008278,-0.041046,-0.012711,-0.008820,-0.001640,0.045654,0.082703,0.095764,0.181763,0.052856,0.182007,0.051971,0.118530,-0.015640,0.064270,0.036133,0.028534,0.013664,0.003489,-0.034210,-0.002899,-0.000518,0.003237,0.004658,0.015602,0.026764,-0.007320,0.159180,0.155640,0.169800,0.172363,0.161987,0.204590,0.208130,0.187988,0.127319,0.152954,0.189697,0.150269,0.121094,0.178955,0.249390,0.175903,0.182983,0.134033,0.199219,0.183105,0.160400,0.155396,0.245972,0.208740,0.121643,0.144287,0.199463,0.135254,0.130981,0.179810,0.233765,0.172485,0.157104,0.165894,0.173340,0.184692,0.168823,0.186768,0.217773,0.186401,0.128418,0.145874,0.162842,0.134277,0.125854,0.180542,0.237305,0.171387,0.197876,0.156006,0.209229,0.181641,0.171143,0.175903,0.240967,0.191772,0.126221,0.153442,0.193481,0.142944,0.116577,0.162476,0.247925,0.158691,0.101664,0.054321,0.000532,0.058706,0.039285,0.020851,0.022683,0.021131,0.000992,0.006743,0.030581,0.006841,0.008028,0.009481,-0.007085,0.016528,-0.003571,0.011121,0.021568,0.012199,-0.045654,-0.011960,-0.045308,-0.030174,-0.049906,-0.031237,-0.129395,-0.057302,-0.077296,-0.075073,-0.088074,-0.091471,0.153158,0.155884,0.178670,0.169515,0.161743,0.165527,0.191081,0.167074,0.160278,0.152588,0.184163,0.181234,0.157756,0.171427,0.208659,0.189697,0.151265,0.146444,0.186198,0.176147,0.154744,0.163940,0.221436,0.189779,0.163778,0.166829,0.206909,0.189982,0.165934,0.172852,0.210978,0.172241,0.157471,0.159139,0.183553,0.164510,0.156657,0.165365,0.188436,0.158895,0.172201,0.165894,0.184652,0.181519,0.178304,0.182129,0.197103,0.185465,0.153524,0.147949,0.191813,0.173503,0.150146,0.161499,0.213257,0.179688,0.161580,0.177409,0.205444,0.181315,0.162882,0.186401,0.211914,0.166585,0.058319,-0.009270,0.068787,0.011856,0.076538,0.017609,0.073120,-0.002546,0.074280,-0.008873,0.062683,-0.003925,0.070374,0.005116,0.047302,-0.017471,0.045715,0.004200,0.032379,-0.003113,0.044983,0.012665,0.057129,0.034454,0.066223,0.066040,0.040070,0.025970,0.064941,0.026642,0.057465,0.001466,0.326660,0.111755,0.120972,0.105347,0.107117,0.106628,0.168579,0.325928,0.342285,0.201660,0.270264,0.140015,0.108337,0.087769,0.115784,0.322266,0.117371,0.120728,0.139893,0.130249,0.118896,0.122437,0.155884,0.158325,0.155029,0.133057,0.148438,0.125366,0.124268,0.135986,0.146851,0.151489,0.297607,0.105286,0.097900,0.122986,0.149780,0.147583,0.180176,0.367432,0.284424,0.151245,0.249390,0.175903,0.101440,0.067749,0.111206,0.332764,0.111938,0.101685,0.136719,0.131348,0.120300,0.124634,0.161865,0.153198,0.211304,0.141113,0.153687,0.128662,0.115845,0.118408,0.136230,0.184570};
double Vt[480] = {0.006521,0.114053,0.182961,0.191888,-0.207898,-0.001681,0.082312,-0.020082,0.011175,0.011797,0.005848,0.065562,0.192329,0.191952,-0.156623,-0.005878,0.027308,-0.024772,0.024620,0.045249,0.011416,0.045566,0.208212,0.122897,-0.167397,-0.002499,0.005622,0.006114,0.009769,0.057689,0.007391,0.004948,0.197702,0.160970,-0.118242,-0.002089,0.018610,-0.005684,-0.002963,0.006837,0.008891,0.028542,0.163586,0.194003,-0.072299,0.000876,0.020195,-0.014827,0.007239,-0.029873,0.000003,0.055531,0.121390,0.206456,0.026527,-0.005797,0.029103,0.004615,-0.011563,0.015555,0.008897,0.059572,0.233873,0.106523,0.120727,-0.016254,0.007449,0.021340,-0.012868,0.095325,-0.005047,0.030245,0.109922,0.184597,0.020203,-0.013737,0.036213,-0.001621,-0.000679,0.094146,0.009809,0.017508,0.180041,0.194903,0.166812,-0.029341,0.028211,0.001277,-0.151163,-0.327530,0.000792,0.006536,0.115605,0.102775,0.002123,0.005471,0.028822,0.017578,0.037986,0.214231,-0.010681,-0.037609,0.076328,0.147408,0.016781,0.004840,-0.015341,0.030080,0.059735,0.156144,-0.000304,-0.022798,0.088274,0.043727,0.005821,0.002872,-0.029554,0.036972,0.017018,0.114511,-0.005607,-0.093070,0.077134,0.054339,-0.034797,0.001679,-0.024241,0.047367,0.022779,0.018733,-0.000206,-0.091875,0.046769,0.036441,-0.022028,0.001832,-0.032596,0.018728,-0.035945,-0.010340,-0.006173,-0.121297,0.083773,0.050466,-0.035766,-0.001965,-0.067535,0.024089,-0.066046,-0.035366,-0.007878,-0.113829,0.069768,0.055694,-0.056071,-0.003064,-0.079091,0.008766,-0.096925,-0.081967,-0.118048,-0.222615,0.209994,-0.080746,0.049673,-0.091391,0.004948,0.024818,-0.086211,0.080267,-0.108706,0.006686,-0.042072,0.003061,0.079912,-0.101871,-0.059262,0.015275,-0.109110,0.020504,-0.098398,-0.109518,0.062351,-0.179709,0.013524,-0.117062,-0.076113,0.005649,-0.133229,-0.070284,-0.158191,-0.092190,-0.069229,0.108177,-0.139402,-0.144150,-0.254294,0.128543,0.044492,0.010285,-0.117706,-0.186611,0.211564,-0.039681,0.052282,-0.119681,-0.132286,0.068134,0.011285,0.043768,-0.167674,-0.141066,-0.096910,0.184364,-0.032719,-0.130322,-0.036990,-0.138270,-0.003729,-0.145652,-0.106818,-0.014486,-0.016925,-0.131338,-0.278777,-0.108870,0.027214,-0.035097,-0.000811,-0.031926,-0.151702,0.009049,-0.167432,0.125759,0.049593,-0.143062,-0.124317,0.013623,0.120724,0.079432,-0.107680,0.176469,0.096829,-0.128992,-0.032154,-0.097636,0.087808,0.035754,-0.109198,0.066679,-0.118905,0.016521,0.005779,-0.097657,0.172835,-0.106565,0.022280,0.033400,-0.112689,0.181178,-0.101900,0.128295,0.094874,-0.080048,0.136032,-0.106902,0.152955,0.051846,0.043754,0.083086,-0.142236,0.110083,-0.026971,0.150555,-0.010799,-0.127622,0.139013,0.044917,-0.009451,-0.142549,-0.131731,0.124817,0.162753,-0.107700,-0.158070,-0.121308,0.052638,0.037833,0.003622,-0.119537,-0.141434,0.020315,-0.157705,0.145492,0.065778,-0.126415,0.119174,-0.154846,0.051434,0.073353,-0.107596,0.263759,0.146673,-0.017906,0.065274,-0.112867,0.123894,0.043196,0.056490,0.034196,-0.152927,0.007145,-0.190410,0.148749,0.013633,-0.151848,0.049535,-0.089602,-0.067495,-0.106668,-0.108305,-0.122726,0.150603,-0.044936,0.108493,-0.090057,0.035502,-0.010756,-0.057923,0.103940,-0.110279,0.013312,-0.060281,-0.022102,0.104088,-0.110474,-0.068942,0.015235,-0.176247,0.063200,-0.101612,-0.045947,0.049776,-0.146772,0.000330,-0.112672,-0.016758,0.013137,-0.071752,-0.114379,-0.161014,-0.150466,-0.021160,0.080124,-0.120539,-0.152766,-0.323632,0.179045,0.002568,0.032099,-0.109242,-0.130953,0.154743,-0.019281,0.075309,-0.122084,-0.172502,0.027418,-0.051879,0.091978,-0.162238,-0.088291,-0.137526,0.188942,-0.106920,-0.135832,-0.056339,-0.125264,0.009642,-0.224242,-0.101791,-0.011037,0.006081,-0.109562,-0.156206,-0.108886,0.033463,-0.068963,-0.051214,-0.025226,-0.154919,0.012326,-0.204867,0.128961,0.029410,-0.141841,-0.100648,-0.008161,0.193952,0.060693,-0.109469,0.172528,0.065544,-0.120578,-0.036839,-0.096562,0.096982,0.015819,-0.086645,0.042818,-0.121177,0.011242,0.000790,-0.091916,0.126978,-0.108889,0.016439,0.021066,-0.132941,0.131333,-0.103653,0.113966,0.088941,-0.079447,0.093475,-0.109747,0.139412,0.067179,0.045491,0.013712,-0.138714,0.095904,-0.001721,0.150336,-0.037392,-0.126565,0.118965,0.026377,-0.022755,-0.115774,-0.132421,0.100509,0.205054,-0.071348,-0.123899,-0.119952,0.046590,0.057913,0.025389,-0.051258,-0.143232,-0.005500,-0.065336,0.104150,0.057897,-0.124465,0.131268,-0.067011,0.010936,0.099541,-0.105755,0.253686,0.127496,-0.015911,0.036576,-0.106960,0.153970,0.024920,0.095590,0.053228,-0.153104,0.015643,-0.197780,0.130878,0.081611,-0.161086,-0.019247,-0.072774,-0.107277,-0.051708};
double VD[50] = {-1.324738,0.174352,0.008671,-0.338726,0.148316,-1.361690,-0.010970,-0.047227,0.388921,-0.108742,-1.237898,0.499326,0.280532,0.049516,-0.008134,-1.300101,-0.526590,0.110898,-0.165740,0.047036,-1.386485,-0.046395,-0.246094,0.012745,0.147715,-1.330367,-0.028128,-0.503732,0.078517,0.016241,-1.347771,-0.002612,-0.028992,-0.196978,-0.460485,-1.396900,0.086766,0.047328,0.007654,0.124319,-1.391079,0.159871,0.021337,-0.000836,0.051631,-1.308511,-0.296410,0.394348,0.158304,0.037634};
NNUDictionary dict = {5,5,256,mini,nnu_table,D,96,10,Vt,VD};