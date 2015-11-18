#include "nnu_dict.h"

int* nnu(NNUDictionary *dict, int alpha, int beta, double *X, int X_rows,
         int X_cols, double *avg_ab)
{
    int i, N;
    int max_idx = 0;
    int total_ab = 0;
    int D_rows = dict->D_rows;
    int D_cols = dict->D_cols;
    int s_stride = storage_stride(dict->storage);
    double max_coeff = 0.0;

    word_t *atom_idxs = bit_vector(D_cols);
    int *candidate_set = (int *)calloc(alpha*beta, sizeof(int));
    int *ret;
    double *D = dict->D;
    double *VX;

    ret = (int *)calloc(X_cols, sizeof(int));
    VX = dmm_prod(dict->Vt, X, dict->alpha*s_stride, dict->D_rows, X_rows, X_cols); 

	for(i = 0; i < X_cols; i++) {
		atom_lookup(dict, d_viewcol(VX, i, dict->alpha*s_stride), atom_idxs,
                    candidate_set, &N, alpha, beta, s_stride);
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

/* Computes the max dot product index of a Dictionary D with input sample x */
inline void compute_max_dot(double *max_coeff, int *max_idx, double *D,
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
