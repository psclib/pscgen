#ifndef NNU_GENERATOR_H
#define NNU_GENERATOR_H

#include <stdlib.h>
#include <string.h>
#include "linalg/linalg.h"
#include "nnu_dict.h"
#include "nnu_storage.h"

void generate_nnu(const char *D_path, const char *output_path, const int alpha,
                  const int beta, Storage_Scheme storage);
void dict_to_file(NNUDictionary *dict, const char* output_path);
void uint16_buffer_to_str(char *output, const char *name, uint16_t *buf, int N);
void double_buffer_to_str(char *output, const char *name, double *buf, int N);

/* Dynamically allocated NNUDictionary functionality */
NNUDictionary* new_dict(const int alpha, const int beta,
                        Storage_Scheme storage, const char *input_csv_path,
                        const char *delimiters);
NNUDictionary* new_dict_from_buffer(const int alpha, const int beta,
                                    Storage_Scheme storage, double *D,
                                    int rows, int cols);
void delete_dict(NNUDictionary *dict);
void save_dict(char *filepath, NNUDictionary *dict);
NNUDictionary* load_dict(char *filepath);


#endif /* NNU_GENERATOR_H */
