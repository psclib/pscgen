#include "nnu.h"
#include <inttypes.h>

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


    //make easier to read
    int rows = dict->ldict_rows;
    int cols = dict->ldict_cols;

    //get eigen vectors of input dictionary file
    dict->ldict_v = deig_vec(dict->ldict, rows, cols);

    printf("%f\n", dict->ldict_v[idx2d(0, 0, rows)]);
    printf("%f\n", dict->ldict_v[idx2d(0, 1, rows)]);
    printf("%f\n", dict->ldict_v[idx2d(0, 2, rows)]);
    printf("%f\n", dict->ldict_v[idx2d(0, 3, rows)]);
    printf("%f\n", dict->ldict_v[idx2d(0, 4, rows)]);
    exit(1);
    /* printf("%f\n", dict->ldict[idx2d(5, 4, cols)]); */

    double *vtd = dmm_prod(dict->ldict_v, dict->ldict, rows, rows, rows, cols);

    printf("%f\n", vtd[idx2d(5, 4, cols)]);

    
    //populate nnu tables
    uint32_t i;
    int j, k, table_idx;
    half v;
    float dv;
    double *c = new_dvec(rows);
    for(i = 0; i < RANGE_16-1; i++) {
        dv = half_to_float(i);
        for(j = 0; j < dict->alpha; j++) {
            for(k = 0; k < rows; k++) {
                c[k] = fabs(dict->ldict_v[idx2d(j, k, rows)] - dv);
                if(i == 2 && j == 3 && k == 4)
                   printf("%g, %g, %g\n", dict->ldict_v[idx2d(j, k, rows)], dv, c[k]);
            }

            int *idxs = d_argsort(c, rows);
            for(k = 0; k < dict->beta; k++) {
                table_idx = idx3d(j, i, k, dict->alpha, RANGE_16);
                dict->tables[table_idx] = idxs[k];
            }

            free(idxs);
            zero_dvec(c, rows);
        }
    }

    uint16_t x = dict->tables[idx3d(2,3,4, dict->alpha, RANGE_16)];
    printf("%d\n", x);
    
    return dict;
}

void delete_dict(NNUDictionary *dict)
{
    free(dict->tables);
    free(dict->ldict);
    free(dict->ldict_v);
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
