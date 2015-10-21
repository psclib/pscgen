#include "nnu.h"

NNUDictionary* new_dict(const int alpha, const int beta,
                        const char *input_csv_path,
                        const char *delimiters)
{
    //Initialze NNUDictionary
    NNUDictionary *dict = malloc(sizeof(NNUDictionary));
    half *tables = malloc(sizeof(half) * alpha * beta * RANGE_16);
    dict->alpha = alpha;
    dict->beta = beta;
    dict->tables = tables;

    //read in input dictionary file
    read_csv(input_csv_path, delimiters, &dict->ldict, &dict->ldict_rows,
             &dict->ldict_cols);

    //get eigen vectors of input dictionary file
    dict->ldict_vt = deig_vec(dict->ldict, dict->ldict_rows, dict->ldict_cols);

    //populate nnu tables
    int i, j, k;
    half v;
    int cols = dict->ldict_cols;
    double dv;
    double *c = new_dvec(cols);
    for(i = 0; i < 1; i++) {
        v = (half)i;
        dv = (double)v;
        for(j = 0; j < dict->alpha; j++) {
            for(k = 0; k < cols; k++) {
                 c[k] = fabs(dict->ldict_vt[idx2d(j, k, cols)] - dv);
            }

            int *idxs = d_argsort(c, cols);
            for(k = 0; k < dict->beta; k++) {
                dict->tables[idx3d(j, i, k, RANGE_16, dict->beta)] = float_to_half(idxs[k]);
            }

            free(idxs);
            zero_dvec(c, dict->ldict_cols);
        }
    }
    
    printf("here\n");
    return dict;
}

void delete_dict(NNUDictionary *dict)
{
    free(dict->tables);
    free(dict->ldict);
    free(dict->ldict_vt);
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
