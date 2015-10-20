#include "nnu.h"

struct _NNUDictionary {
    int alpha; //number of tables
    int beta;  //width of tables
    half* tables; //nnu lookup tables
};

NNUDictionary* new_dict(const int alpha, const int beta,
                        const char *input_csv_path,
                        const char *delimiters)
{
    //Initialze nnu dictionary
    NNUDictionary *dict = malloc(sizeof(NNUDictionary));
    half *tables = malloc(sizeof(half) * alpha * beta * RANGE_16);
    dict->alpha = alpha;
    dict->beta = beta;
    dict->tables = tables;

    //read in input csv file
    int input_rows, input_cols;
    double *input_buf;
    read_csv(input_csv_path, delimiters, &input_rows, &input_cols, &input_buf);

    //get eigen vectors

    //populate nnu tables

    //clean-up
    free(input_buf);

    return dict;
}

void delete_dict(NNUDictionary *dict)
{
    free(dict->tables);
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
