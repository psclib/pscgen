#include "nnu.h"
#include <inttypes.h>

NNUDictionary* new_dict(const int alpha, const int beta,
                        const char *input_csv_path,
                        const char *delimiters)
{
    //init variables
    uint16_t *tables = malloc(sizeof(uint16_t) * alpha * beta * USHRT_MAX);
    double *D, *Dt, *Vt, *Vt_full, *VD;
    int rows, cols;

    //read in input dictionary file
    read_csv(input_csv_path, delimiters, &D, &rows, &cols);

    //transpose Dt
    Dt = d_transpose(D, rows, cols);

    //get eigen vectors of input dictionary file
    Vt_full = deig_Vt(Dt, cols, rows);

    //trim to top alpha vectors
    Vt = d_trim(Vt_full, rows, rows, alpha, rows);

    //compute prod(V, D)
    VD = dmm_prod(Vt, D, alpha, rows, rows, cols);
    
    //populate nnu tables
    int i, j, k, table_idx;
    float dv;
    double *c = new_dvec(cols);
    int *idxs = malloc(sizeof(int) * cols);

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

    //Initialze NNUDictionary
    NNUDictionary *dict = malloc(sizeof(NNUDictionary));
    dict->tables = tables;
    dict->alpha = alpha;
    dict->beta = beta;
    dict->D = D;
    dict->D_rows = rows;
    dict->D_cols = cols;
    dict->VD = VD;
    dict->Vt = Vt;

    //clean-up
    free(Dt);
    free(Vt_full);
    free(idxs);
    free(c);

    return dict;
}

void delete_dict(NNUDictionary *dict)
{
    free(dict->tables);
    free(dict->D);
    free(dict->Vt);
    free(dict->VD);
    free(dict);
}

double* nnu(NNUDictionary *dict, double *X, int X_rows, int X_cols)
{
    int i, j, maxidx;
    int alpha_beta = dict->alpha * dict->beta;
    int D_rows = dict->D_rows;
    int D_cols = dict->D_cols;
    int alpha = dict->alpha;
    int beta = dict->beta;
    double tmpcoeff;
    double maxcoeff = 0.0;

    word_t *atom_idxs = bit_vector(D_cols);

    double *D = dict->D;
    double *ret = new_dvec(X_cols);
    double *VX = dmm_prod(dict->Vt, X, dict->alpha, dict->D_rows,
                          X_rows, X_cols); 

	for(i = 0; i < X_cols; ++i) {
		atom_lookup(dict->tables, d_viewcol(VX, i, alpha), atom_idxs,
                    alpha, beta);

		for(j = 0; j < D_cols; j++) {
            //skip unselected values
            if(get_bit(atom_idxs, j) == 0) {
                continue;
            }
            
            tmpcoeff = d_dot(d_viewcol(X, i, X_rows),
                             d_viewcol(D, j, D_rows), D_rows);
			if(fabs(tmpcoeff) > maxcoeff) {
				maxcoeff = tmpcoeff;
				maxidx = j;
			}
		}

		ret[i] = maxidx;
		maxcoeff = 0.0;
        clear_all_bit(atom_idxs, D_rows);
	}

    //clean-up
    free(VX);
    free(atom_idxs);

	return ret;
}

inline void atom_lookup(uint16_t *tables, double *x, word_t *atom_idxs,
                        int alpha, int beta)
{
    int i, k, table_idx;
    uint16_t *beta_neighbors;
    
    for(i = 0; i < alpha; i++) {
        table_idx = idx3d(i, (int)float_to_half((float)x[i]), 0,
                          beta, USHRT_MAX);
        beta_neighbors = &tables[table_idx];
        for(k = 0; k < beta; ++k) {
            set_bit(atom_idxs, beta_neighbors[k]);
        }
    }
}
