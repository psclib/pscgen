#include "nnu.h"
#include <inttypes.h>

NNUDictionary* new_dict(const int alpha, const int beta,
                        const char *input_csv_path,
                        const char *delimiters)
{
    double *D;
    int rows, cols;

    /* read in input dictionary file */
    read_csv(input_csv_path, delimiters, &D, &rows, &cols);

    /* create NNUDictionary using read in file */
    return new_dict_from_buffer(alpha, beta, D, rows, cols);
}

NNUDictionary* new_dict_from_buffer(const int alpha, const int beta,
                                    double *D, int rows, int cols)
{
    int i, j, k, q, table_idx;
    float dv;
    int *beta_scale, *idxs;
    double *Dt, *Vt, *Vt_full, *VD, *U, *S, *s_values, *c;
    uint16_t *tables = malloc(sizeof(uint16_t) * alpha * beta * USHRT_MAX);
    NNUDictionary *dict;

    /* transpose D */
    Dt = d_transpose(D, rows, cols);

    /* get eigen vectors of input dictionary file */
    d_SVD(Dt, cols, rows, &U, &S, &Vt_full);

    /* compute scaling values for beta */
    s_values = d_viewcol(S, 0, cols);
    beta_scale = compute_beta_scale(s_values, alpha, beta);

    /* Turning off beta scale */
    for(q = 0; q < alpha; q++) {
        beta_scale[q] = beta;
    }

    /* trim to top alpha vectors */
    Vt = d_trim(Vt_full, rows, alpha, rows);

    /* compute prod(V, D) */
    VD = dmm_prod(Vt, D, alpha, rows, rows, cols);
    
    /* populate nnu tables */
    c = new_dvec(cols);
    idxs = malloc(sizeof(int) * cols);

    for(i = 0; i < USHRT_MAX; i++) {
        dv = half_to_float(i);
        for(j = 0; j < alpha; j++) {
            for(k = 0; k < cols; k++) {
                c[k] = fabs(VD[idx2d(j, k, alpha)] - dv);
            }

            d_argsort(c, idxs, cols);
            for(k = 0; k < beta; k++) {
                table_idx = idx3d(j, i, k, beta, USHRT_MAX);
                tables[table_idx] = idxs[k];
            }

            zero_dvec(c, cols);
        }
    }

    /* Initialze NNUDictionary */
    dict = malloc(sizeof(NNUDictionary));
    dict->tables = tables;
    dict->alpha = alpha;
    dict->beta = beta;
    dict->D = D;
    dict->D_rows = rows;
    dict->D_cols = cols;
    dict->VD = VD;
    dict->Vt = Vt;
    dict->beta_scale = beta_scale;

    /* clean-up */
    free(Dt);
    free(U);
    free(S);
    free(Vt_full);
    free(idxs);
    free(c);

    return dict;
}

void save_dict(char *filepath, NNUDictionary *dict)
{
    FILE *fp = fopen(filepath, "w+");
    fwrite(&dict->alpha, sizeof(int), 1, fp);
    fwrite(&dict->beta, sizeof(int), 1, fp);
    fwrite(dict->tables, sizeof(uint16_t),
           dict->alpha * dict->beta * USHRT_MAX, fp);
    fwrite(&dict->D_rows, sizeof(int), 1, fp);
    fwrite(&dict->D_cols, sizeof(int), 1, fp);
    fwrite(dict->D, sizeof(double), dict->D_rows * dict->D_cols, fp);
    fwrite(dict->Vt, sizeof(double), dict->alpha * dict->D_rows, fp);
    fwrite(dict->VD, sizeof(double), dict->alpha * dict->D_cols, fp);
    fwrite(dict->beta_scale, sizeof(int), dict->alpha, fp);
    fclose(fp);
}

NNUDictionary* load_dict(char *filepath)
{
    NNUDictionary *dict = malloc(sizeof(NNUDictionary));
    FILE *fp = fopen(filepath, "r");
    fread(&dict->alpha, sizeof(int), 1, fp);
    fread(&dict->beta, sizeof(int), 1, fp);
    dict->tables = malloc(sizeof(uint16_t) * dict->alpha *
                          dict->beta * USHRT_MAX);
    fread(dict->tables, sizeof(uint16_t),
          dict->alpha * dict->beta * USHRT_MAX, fp);
    fread(&dict->D_rows, sizeof(int), 1, fp);
    fread(&dict->D_cols, sizeof(int), 1, fp);
    dict->D = malloc(sizeof(double) * dict->D_rows * dict->D_cols);
    dict->Vt = malloc(sizeof(double) * dict->alpha * dict->D_rows);
    dict->VD = malloc(sizeof(double) * dict->alpha * dict->D_cols);
    dict->beta_scale = malloc(sizeof(int) * dict->alpha);
    fread(dict->D, sizeof(double), dict->D_rows * dict->D_cols, fp);
    fread(dict->Vt, sizeof(double), dict->alpha * dict->D_rows, fp);
    fread(dict->VD, sizeof(double), dict->alpha * dict->D_cols, fp);
    fread(dict->beta_scale, sizeof(int), dict->alpha, fp);
    fclose(fp);

    return dict;
}


void delete_dict(NNUDictionary *dict)
{
    free(dict->tables);
    free(dict->D);
    free(dict->Vt);
    free(dict->VD);
    free(dict->beta_scale);
    free(dict);
}

double* nnu(NNUDictionary *dict, double *X, int X_rows, int X_cols,
            double *avg_ab)
{
    int i, N;
    int max_idx = 0;
    int total_ab = 0;
    int D_rows = dict->D_rows;
    int D_cols = dict->D_cols;
    int alpha = dict->alpha;
    int beta = dict->beta;
    double max_coeff = 0.0;

    word_t *atom_idxs = bit_vector(D_cols);
    int *candidate_set = malloc(sizeof(int)*alpha*beta);
    double *D = dict->D;
    double *ret = new_dvec(X_cols);
    double *VX = dmm_prod(dict->Vt, X, dict->alpha, dict->D_rows,
                          X_rows, X_cols); 

	for(i = 0; i < X_cols; i++) {
		atom_lookup(dict->tables, d_viewcol(VX, i, alpha), atom_idxs,
                    candidate_set, &N, alpha, beta, dict->beta_scale);
        compute_max_dot_set(&max_coeff, &max_idx, &total_ab, D,
                            d_viewcol(X, i, X_rows), candidate_set, D_rows, N);
		ret[i] = max_idx;
        clear_all_bit(atom_idxs, D_cols);
	}

    *avg_ab = total_ab / (double)X_cols;

    /* clean-up */
    free(VX);
    free(atom_idxs);
    free(candidate_set);

	return ret;
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


inline void compute_max_dot_set(double *max_coeff, int *max_idx, int *total_ab,
                                double *D, double *x, int *candidate_set,
                                int D_rows, int N)
{
    int i;
    double tmp_coeff;
    *max_coeff = 0.0;
    (*total_ab) += N;

	for(i = 0; i < N; i++) {
        tmp_coeff = d_dot(x, d_viewcol(D, candidate_set[i], D_rows), D_rows);
        if(fabs(tmp_coeff) > *max_coeff) {
            *max_coeff = tmp_coeff;
            *max_idx = candidate_set[i];
        }
    }
}

inline void compute_max_dot(double *max_coeff, int *max_idx, double *D,
                            double *x, int D_rows, int D_cols)
{
    int i;
    double tmp_coeff;
    *max_coeff = 0.0;

    for(i = 0; i < D_cols; i++) {
        tmp_coeff = d_dot(x, d_viewcol(D, i, D_rows), D_rows);

        if(fabs(tmp_coeff) > *max_coeff) {
            *max_coeff = tmp_coeff;
            *max_idx = i;
        }
    }
}


/* void mp(NNUDictionary *dict, double *X, int X_rows, int X_cols, int k) */
/* { */
/*     int D_rows = dict->D_rows; */
/*     int D_cols = dict->D_cols; */
/*     int i, j; */
/*     double tmpcoeff; */
/*     double maxcoeff = 0.0; */

/*     double *D = dict->D; */
/*     double *ret = calloc(X_cols * D_cols, sizeof(double)); */

/* 	for(i = 0; i < X_cols; i++) { */
/* 		for(j = 0; j < D_cols; j++) { */
/*             tmpcoeff = d_dot(d_viewcol(X, i, X_rows), */
/*                              d_viewcol(D, j, D_rows), D_rows); */

/* 			if(fabs(tmpcoeff) > maxcoeff) { */
/* 				maxcoeff = tmpcoeff; */
/* 				maxidx = candidate_set[j]; */
/* 			} */
/* 		} */

/* 		ret[i] = maxidx; */
/* 		maxcoeff = 0.0; */
/* 	} */

/* 	return ret; */
/* } */


void atom_lookup(uint16_t *tables, double *x, word_t *atom_idxs,
                 int *candidate_set, int *N, int alpha, int beta,
                 int *beta_scale)
{
    int i, j, table_idx;
    uint16_t *beta_neighbors;
    *N = 0;
    
    for(i = 0; i < alpha; i++) {
        table_idx = idx3d(i, float_to_half(x[i]), 0, beta, USHRT_MAX);
        beta_neighbors = &tables[table_idx];
        for(j = 0; j < beta_scale[i]; ++j) {
            if(get_bit(atom_idxs, beta_neighbors[j]) == 0) {
                set_bit(atom_idxs, beta_neighbors[j]);
                candidate_set[*N] = beta_neighbors[j];
                (*N)++;
            }
        }
    }
}

/* Computes the scaling beta values based on the singular values */
int* compute_beta_scale(double *s_values, int alpha, int beta)
{
    int i;
    int *beta_scale = malloc(sizeof(int) * alpha);
    double c = (double)beta / s_values[0];

    for(i = 0; i < alpha; i++) {
        beta_scale[i] = (int)ceil(c * s_values[i]);
    }

    return beta_scale;
}
