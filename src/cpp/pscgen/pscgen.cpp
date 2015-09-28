#include "pscgen.h"

uint16_t*** new_dict(const int alpha, const int beta, const int range) {
    uint16_t ***dict = new uint16_t**[alpha];
    for (int i = 0; i < alpha; ++i) {
        dict[i] = new uint16_t*[range];
        for (int j = 0; j < range; ++j) {
            dict[i][j] = new uint16_t[beta];
        }
    }
    return dict;
}

void delete_dict(uint16_t ***dict, const int alpha, const int range) {
    for (int i = 0; i < alpha; ++i) {
        for (int j = 0; j < range; ++j) {
            delete [] dict[i][j];
        }
        delete [] dict[i];
    }
    delete [] dict;
}


void build_dict(double **vtd, uint16_t ***dict, const int alpha,
                const int beta, const int cols, const int range) {
    for (uint32_t dictvalue = 0; dictvalue < range; ++dictvalue) {
        half v;
        v.data_ = dictvalue;
        double dictdouble = v;
        for (int j = 0; j < alpha; ++j) {
            std::vector<double> c(cols, 0.0);
            for (int i = 0; i < cols; ++i) {
                double v = vtd[j][i];
                c[i] = std::abs(v - dictdouble);
            }
            std::vector<size_t> idxs = sort_indexes(c);
            for (int k = 0; k < beta; ++k) {
                dict[j][dictvalue][k] = idxs[k];
            }
        }
    }
}


void atom_lookup(double *x, uint16_t ***dict, int **idx_arr, const int alpha,
                 const int beta) {
    std::set<int> idxs;
    for (int j = 0; j < alpha; ++j) {
        uint16_t *beta_neighbors = dict[alpha][half(x[j]).data_];
        for (int k = 0; k < beta; ++k) {
            idxs.insert(beta_neighbors[k]);
        }
    }
    std::copy(idxs.begin(), idxs.end(), *idx_arr);
}


//read csv from file
double* read_csv(std::string file, int &rows, int &cols) {
    std::ifstream data(file.c_str());
    std::string line;
    rows = cols = 0;

    while(std::getline(data, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        cols = 0;
        while(std::getline(lineStream, cell, ',')) {
            cols++;
        }
        rows++;
    }

    double *ret = new double[rows*cols];

    //rewind file to beginning
    data.clear();
    data.seekg(0, data.beg);
    int i = 0;
    int j = 0;
    while(std::getline(data, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        j = 0;
        while(std::getline(lineStream, cell, ',')) {
            ret[i*rows + j] = std::stod(cell);
            j++;
        }
        i++;
    }

  return ret;
}