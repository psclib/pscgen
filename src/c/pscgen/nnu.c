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
    half v;
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
                table_idx = idx3d(j, i, k, alpha, USHRT_MAX);
                tables[table_idx] = idxs[k];
            }

            zero_dvec(c, cols);
        }
    }

    //Initialze NNUDictionary
    NNUDictionary *dict = malloc(sizeof(NNUDictionary));
    dict->tables = tables;
    dict->alpha = alpha;
    dict->D = D;
    dict->VD = VD;

    //clean-up
    free(Dt);
    free(Vt_full);
    free(Vt);
    free(idxs);
    free(c);

    return dict;
}

void delete_dict(NNUDictionary *dict)
{
    free(dict->tables);
    free(dict->D);
    free(dict->VD);
    free(dict);
}

/* void build_dict(double **vtd, uint16_t ***dict, const int alpha, */
/*                 const int beta, const int cols, const int range) { */
/*     for (uint32_t dictvalue = 0; dictvalue < range; ++dictvalue) { */
/*         half v; */
/*         v.data_ = dictvalue; */
/*         double dictdouble = v; */
/*         for (int j = 0; j < alpha; ++j) { */

/*             std::vector<double> c(cols, 0.0); */
/*             for (int i = 0; i < cols; ++i) { */
/*                 double v = vtd[j][i]; */
/*                 c[i] = std::abs(v - dictdouble); */
/*             } */

/*             std::vector<size_t> idxs = sort_indexes(c); */
/*             for (int k = 0; k < beta; ++k) { */
/*                 dict[j][dictvalue][k] = idxs[k]; */
/*             } */
/*         } */
/*     } */
/* } */


/* void atom_lookup(double *x, uint16_t ***dict, int **idx_arr, const int alpha, */
/*                  const int beta) { */
/*     std::set<int> idxs; */
/*     for (int j = 0; j < alpha; ++j) { */
/*         uint16_t *beta_neighbors = dict[alpha][half(x[j]).data_]; */
/*         for (int k = 0; k < beta; ++k) { */
/*             idxs.insert(beta_neighbors[k]); */
/*         } */
/*     } */
/*     std::copy(idxs.begin(), idxs.end(), *idx_arr); */
/* } */


/* //read csv from file */
/* double* read_csv(std::string file, int &rows, int &cols) { */
/*     std::ifstream data(file.c_str()); */
/*     std::string line; */
/*     rows = cols = 0; */

/*     while(std::getline(data, line)) { */
/*         std::stringstream lineStream(line); */
/*         std::string cell; */
/*         cols = 0; */
/*         while(std::getline(lineStream, cell, ',')) { */
/*             cols++; */
/*         } */
/*         rows++; */
/*     } */

/*     double *ret = new double[rows*cols]; */

/*     //rewind file to beginning */
/*     data.clear(); */
/*     data.seekg(0, data.beg); */
/*     int i = 0; */
/*     int j = 0; */
/*     while(std::getline(data, line)) { */
/*         std::stringstream lineStream(line); */
/*         std::string cell; */
/*         j = 0; */
/*         while(std::getline(lineStream, cell, ',')) { */
/*             ret[i*rows + j] = std::stod(cell); */
/*             j++; */
/*         } */
/*         i++; */
/*     } */

/*   return ret; */
/* } */
