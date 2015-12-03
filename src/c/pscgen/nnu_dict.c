#include "nnu_dict.h"

NNUDictionary* new_dict(const int alpha, const int beta,
                        const int max_atoms, Storage_Scheme storage,
                        const char *input_csv_path,
                        const char *delimiters)
{
    double *D;
    int rows, cols;

    /* read in input dictionary file */
    read_csv(input_csv_path, delimiters, &D, &rows, &cols);

    /* create NNUDictionary using read in file */
    return new_dict_from_buffer(alpha, beta, storage, D, rows, cols, max_atoms);
}

NNUDictionary* new_dict_from_buffer(const int alpha, const int beta,
                                    Storage_Scheme storage, double *D,
                                    int rows, int cols, int max_atoms)
{
    int i, j, k, l, idx, gamma, table_idx, s_stride;
    int *idxs;
    double *Dt, *Vt, *Vt_full, *VD, *U, *S, *c, *D_mean;
    float *dv;
    uint16_t *tables;
    NNUDictionary *dict;

    /* make zero-mean and unit variance */
    normalize_colwise(D, rows, cols);
    D_mean = mean_colwise(D, rows, cols);
    subtract_colwise(D, D_mean, rows, cols);

    gamma = ipow(2, storage_gamma_pow(storage));
    s_stride = storage_stride(storage);
    tables = (uint16_t *)malloc(alpha * beta * gamma * sizeof(uint16_t));

    /* transpose D */
    Dt = d_transpose(D, rows, cols);

    /* get eigen vectors of input dictionary file */
    lapack_d_SVD(Dt, cols, rows, &U, &S, &Vt_full);

    /* trim to top alpha vectors */
    Vt = d_trim(Vt_full, rows, alpha*s_stride, rows);

    /* compute prod(V, D) */
    VD = dmm_prod(Vt, D, alpha*s_stride, rows, rows, cols);
    
    /* populate nnu tables */
    #pragma omp parallel private(dv, c, idxs, idx, table_idx, j, k, l)
    {
        idx = table_idx = j = k = l = 0;
        dv = (float *)malloc(sizeof(float) * s_stride);
        c = (double *)calloc(cols, sizeof(double));
        idxs = (int *)malloc(sizeof(int) * cols);

        #pragma omp for nowait
        for(i = 0; i < gamma; i++) {
            storage_to_float(dv, i, storage);
            for(j = 0; j < alpha; j++) {
                for(k = 0; k < cols; k++) {
                    for(l = 0; l < s_stride; l++) {
                        idx = idx2d(j*s_stride + l, k, alpha*s_stride);
                        c[k] += fabs(VD[idx] - dv[l]);
                    }
                }

                d_argsort(c, idxs, cols);
                for(k = 0; k < beta; k++) {
                    table_idx = idx3d(j, i, k, beta, gamma);
                    tables[table_idx] = idxs[k];
                }

                zero_dvec(c, cols);
            }
        }

        /* omp clean-up */
        free(idxs);
        free(dv);
        free(c);
    }

    /* Initialze NNUDictionary */
    dict = (NNUDictionary *)malloc(sizeof(NNUDictionary));
    dict->tables = tables;
    dict->alpha = alpha;
    dict->beta = beta;
    dict->gamma = gamma;
    dict->storage = storage;
    dict->D = D;
    dict->D_mean = D_mean; 
    dict->D_rows = rows;
    dict->D_cols = cols;
    dict->max_atoms = max_atoms;
    dict->VD = VD;
    dict->Vt = Vt;

    /* clean-up */
    free(Dt);
    free(U);
    free(S);
    free(Vt_full);

    return dict;
}

/* void save_dict(char *filepath, NNUDictionary *dict) */
/* { */
/*     int s_stride = storage_stride(dict->storage); */

/*     FILE *fp = fopen(filepath, "w+"); */
/*     fwrite(&dict->alpha, sizeof(int), 1, fp); */
/*     fwrite(&dict->beta, sizeof(int), 1, fp); */
/*     fwrite(&dict->gamma, sizeof(int), 1, fp); */
/*     fwrite(&dict->storage, sizeof(Storage_Scheme), 1, fp); */
/*     fwrite(dict->tables, sizeof(uint16_t), */
/*            dict->alpha * dict->beta * dict->gamma, fp); */
/*     fwrite(&dict->D_rows, sizeof(int), 1, fp); */
/*     fwrite(&dict->D_cols, sizeof(int), 1, fp); */
/*     fwrite(dict->D, sizeof(double), dict->D_rows * dict->D_cols, fp); */
/*     fwrite(dict->Vt, sizeof(double), dict->alpha*s_stride * dict->D_rows, fp); */
/*     fwrite(dict->VD, sizeof(double), dict->alpha*s_stride * dict->D_cols, fp); */
/*     fclose(fp); */
/* } */

/* NNUDictionary* load_dict(char *filepath) */
/* { */

/*     int s_stride; */
/*     NNUDictionary *dict = (NNUDictionary *)malloc(sizeof(NNUDictionary)); */
/*     FILE *fp = fopen(filepath, "r"); */
/*     fread(&dict->alpha, sizeof(int), 1, fp); */
/*     fread(&dict->beta, sizeof(int), 1, fp); */
/*     fread(&dict->gamma, sizeof(int), 1, fp); */
/*     fread(&dict->storage, sizeof(Storage_Scheme), 1, fp); */
/*     s_stride = storage_stride(dict->storage); */
/*     dict->tables = (uint16_t *)malloc(sizeof(uint16_t) * dict->alpha * */
/*                                       dict->beta * dict->gamma); */
/*     fread(dict->tables, sizeof(uint16_t), */
/*           dict->alpha * dict->beta * dict->gamma, fp); */
/*     fread(&dict->D_rows, sizeof(int), 1, fp); */
/*     fread(&dict->D_cols, sizeof(int), 1, fp); */
/*     dict->D = (double *)malloc(sizeof(double) * dict->D_rows * dict->D_cols); */
/*     dict->Vt = (double *)malloc(sizeof(double) * dict->alpha*s_stride * */
/*                                 dict->D_rows); */
/*     dict->VD = (double *)malloc(sizeof(double) * dict->alpha*s_stride * */
/*                                 dict->D_cols); */
/*     fread(dict->D, sizeof(double), dict->D_rows * dict->D_cols, fp); */
/*     fread(dict->Vt, sizeof(double), dict->alpha * dict->D_rows, fp); */
/*     fread(dict->VD, sizeof(double), dict->alpha * dict->D_cols, fp); */
/*     fclose(fp); */

/*     return dict; */
/* } */


void delete_dict(NNUDictionary *dict)
{
    free(dict->tables);
    free(dict->D);
    free(dict->D_mean);
    free(dict->Vt);
    free(dict->VD);
    free(dict);
}

int* nnu(NNUDictionary *dict, int alpha, int beta, double *X, int X_rows,
         int X_cols, double *avg_ab)
{
    int i, D_rows, D_cols, s_stride, N, max_idx, total_ab, thread_ab, *ret,
        *candidate_set;
    word_t *atom_idxs;
    double max_coeff, *D, *VX;

    /* zero mean and unit norm */
    normalize_colwise(X, X_rows, X_cols);
    subtract_colwise(X, dict->D_mean, X_rows, X_cols);


    D_rows = dict->D_rows;
    D_cols = dict->D_cols;
    s_stride = storage_stride(dict->storage);
    D = dict->D;
    total_ab = 0;
    ret = (int *)calloc(X_cols, sizeof(int));
    VX = blas_dmm_prod(dict->Vt, X, dict->alpha*s_stride, dict->D_rows, X_rows,
                  X_cols); 


    N = max_idx = thread_ab = 0;
    max_coeff = 0.0;
    atom_idxs = bit_vector(D_cols);
    candidate_set = (int *)calloc(alpha*beta, sizeof(int));

    for(i = 0; i < X_cols; i++) {
        atom_lookup(dict, d_viewcol(VX, i, dict->alpha*s_stride),
                    atom_idxs, candidate_set, &N, alpha, beta, s_stride);
        compute_max_dot_set(&max_coeff, &max_idx, &thread_ab, D,
                            d_viewcol(X, i, X_rows), candidate_set,
                            D_rows, N);
        ret[i] = max_idx;
        clear_all_bit(atom_idxs, D_cols);
    }

    total_ab += thread_ab;

    free(atom_idxs);
    free(candidate_set);

    /* update total ab used */
    *avg_ab = total_ab / (double)X_cols;

    /* clean-up */
    free(VX);

	return ret;
}

/* NNU lookup for input vector X */
int nnu_single(NNUDictionary *dict, double *X, int X_rows)
{
    double *VX;
    int N;
    int max_idx = 0;
    int total_ab = 0;
    int D_rows = dict->D_rows;
    int D_cols = dict->D_cols;
    int s_stride = storage_stride(dict->storage);
    int X_cols = 1;  /* fixes X_cols to single vector case */
    double max_coeff = 0.0;

    word_t* atom_idxs = bit_vector(D_cols);
    int *candidate_set = (int *)calloc(dict->alpha*dict->beta, sizeof(int));
    double *D = dict->D;

    /* zero mean and unit norm */
    normalize_colwise(X, X_rows, X_cols);
    subtract_colwise(X, dict->D_mean, X_rows, X_cols);


    VX = dmm_prod(dict->Vt, X, dict->alpha*s_stride, dict->D_rows, X_rows,
                  X_cols); 

    atom_lookup(dict, d_viewcol(VX, 0, dict->alpha*s_stride), atom_idxs,
                candidate_set, &N, dict->alpha, dict->beta, s_stride);
    compute_max_dot_set(&max_coeff, &max_idx, &total_ab, D,
                        d_viewcol(X, 0, X_rows), candidate_set, D_rows, N);


    /* clean-up */
    free(atom_idxs);
    free(candidate_set);
    free(VX);

	return max_idx;
}

int nnu_pca_single(NNUDictionary *dict, double *X, int X_rows)
{
    double *VX;
    int N;
    int max_idx = 0;
    int total_ab = 0;
    int D_cols = dict->D_cols;
    int s_stride = storage_stride(dict->storage);
    int X_cols = 1;  /* fixes X_cols to single vector case */
    double max_coeff = 0.0;

    word_t* atom_idxs = bit_vector(D_cols);
    int *candidate_set = (int *)calloc(dict->alpha*dict->beta, sizeof(int));

    /* zero mean and unit norm */
    normalize_colwise(X, X_rows, X_cols);
    subtract_colwise(X, dict->D_mean, X_rows, X_cols);


    VX = dmm_prod(dict->Vt, X, dict->alpha*s_stride, dict->D_rows, X_rows,
                  X_cols); 
    atom_lookup(dict, d_viewcol(VX, 0, dict->alpha*s_stride), atom_idxs,
                candidate_set, &N, dict->alpha, dict->beta, s_stride);
    compute_max_dot_set(&max_coeff, &max_idx, &total_ab, dict->VD,
                        VX, candidate_set, dict->alpha*s_stride, N);

    /* clean-up */
    free(atom_idxs);
    free(candidate_set);
    free(VX);

	return max_idx;
}


/* NNU lookup for input vector X */
int nnu_single_nodot(NNUDictionary *dict, double *X, int X_rows, 
                      int *candidate_set)
{
    double *VX;
    int N;
    int D_cols = dict->D_cols;
    int s_stride = storage_stride(dict->storage);
    int X_cols = 1;  /* fixes X_cols to single vector case */

    word_t* atom_idxs = bit_vector(D_cols);

    /* zero mean and unit norm */
    normalize_colwise(X, X_rows, X_cols);
    subtract_colwise(X, dict->D_mean, X_rows, X_cols);

    VX = dmm_prod(dict->Vt, X, dict->alpha*s_stride, dict->D_rows, X_rows,
                  X_cols); 

    atom_lookup(dict, d_viewcol(VX, 0, dict->alpha*s_stride), atom_idxs,
                candidate_set, &N, dict->alpha, dict->beta, s_stride);

    /* clean-up */
    free(atom_idxs);
    free(VX);

	return N;
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

int nns_single(NNUDictionary *dict, double *X, int X_rows)
{
    int D_rows = dict->D_rows;
    int D_cols = dict->D_cols;
    int max_idx = 0;
    double max_coeff;
    double *D = dict->D;

    compute_max_dot(&max_coeff, &max_idx, D, d_viewcol(X, 0, X_rows),
                    D_rows, D_cols);

	return max_idx;
}

double* nns(NNUDictionary *dict, double *X, int X_rows, int X_cols)
{
    int D_rows = dict->D_rows;
    int D_cols = dict->D_cols;
    int max_idx = 0;
    int i;
    double max_coeff;
    double *D = dict->D;
    double *ret = new_dvec(X_cols);

	for(i = 0; i < X_cols; i++) {
        compute_max_dot(&max_coeff, &max_idx, D, d_viewcol(X, i, X_rows),
                        D_rows, D_cols);
		ret[i] = max_idx;
	}

	return ret;
}

double* mp(NNUDictionary *dict, double *X, int X_rows, int X_cols, int K)
{
    int i, j, k;
    int D_rows = dict->D_rows;
    int D_cols = dict->D_cols;
    int max_idx = 0;
    double tmp_coeff;
    double max_coeff = 0.0;

    double *D = dict->D;
    double *ret = (double *)calloc(X_cols * D_cols, sizeof(double));

	for(i = 0; i < X_cols; i++) {
		for(j = 0; j < D_cols; j++) {
		    for(k = 0; k < K; k++) {
                tmp_coeff = d_dot(d_viewcol(X, i, X_rows),
                                  d_viewcol(D, j, D_rows), D_rows);
                tmp_coeff = fabs(tmp_coeff);
                if(tmp_coeff > max_coeff) {
                    max_coeff = tmp_coeff;
                    max_idx = j;
                }
            }
		    ret[idx2d(i, max_idx, X_cols)] = max_coeff;
    		max_coeff = 0.0;
        }
	}

	return ret;
}


/* Computes the max dot product from candidate set with input sample x */
void compute_max_dot_set(double *max_coeff, int *max_idx, int *total_ab,
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

/* Computes the max dot product index of a Dictionary D with input sample x */
void compute_max_dot(double *max_coeff, int *max_idx, double *D,
                            double *x, int D_rows, int D_cols)
{
    int i;
    double tmp_coeff;
    *max_coeff = 0.0;

    for(i = 0; i < D_cols; i++) {
        tmp_coeff = d_dot(x, d_viewcol(D, i, D_rows), D_rows);
        tmp_coeff = fabs(tmp_coeff);
        if(tmp_coeff > *max_coeff) {
            *max_coeff = tmp_coeff;
            *max_idx = i;
        }
    }
}
