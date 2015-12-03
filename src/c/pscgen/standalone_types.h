/* Static version of Pipeline -- used for standalone embedded apps */
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

typedef uint32_t word_t;
enum { WORD_SIZE = sizeof(word_t) * 8 };

typedef enum
{
    half,
    mini,
    micro,
    nano,
    two_mini,
    four_micro
} Storage_Scheme;

/* NNU dictionary */
typedef struct NNUDictionary {
    int alpha; /* curr alpha */
    int beta;  /* curr beta */
    int max_alpha; /* height of tables */
    int max_beta; /* width of tables */
    int gamma; /* depth of tables */
    Storage_Scheme storage; /*float representation of each index */
    int D_rows; /* rows in D */
    int D_cols; /* curr cols in D */
    int max_atoms; /* max cols in D */
    
    uint16_t *tables; /* nnu lookup tables (stores candidates)*/
    FLOAT_T *D; /* learned dictionary */
    FLOAT_T *D_mean; /*colwise mean of D */
    FLOAT_T *Vt; /* Vt from SVD(D) -- taking alpha columns */
    FLOAT_T *VD; /* dot(Vt, d) */
} NNUDictionary;

/* linear SVM */
typedef struct SVM {
    int num_features;
    int num_classes;
    int num_clfs;
    int max_classes;
    
    FLOAT_T *coefs;
    FLOAT_T *intercepts;
} SVM;

typedef struct Pipeline {
    int ws;
    int ss;

    FLOAT_T *window_X;
    FLOAT_T *bag_X;
    NNUDictionary *nnu;
    SVM *svm;
} Pipeline;

Pipeline PIPELINE;
