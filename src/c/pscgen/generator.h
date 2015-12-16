#ifndef GENERATOR_H
#define GENERATOR_H

#include <stdlib.h>
#include <string.h>
#include "nnu_dict.h"
#include "nnu_storage.h"
#include "classifier.h"
#include "pipeline.h"
#include "standalone_char.h"
#include "standalone_types_char.h"

void generate_nnu(const char *D_path, const char *output_path, const int alpha,
                  const int beta, Storage_Scheme storage,
                  Compression_Scheme comp_scheme, const int max_atoms);
char* dict_to_str(NNUDictionary *dict, const char *enc_type);
char* svm_to_str(SVM *svm);
char* pipeline_to_str(Pipeline *pipeline, const char *enc_type,
                      const char *float_type);
void dict_to_file(NNUDictionary *dict, const char* output_path);
void uint16_buffer_to_str(char *output, const char *name, uint16_t *buf, int N);
void int_buffer_to_str(char *output, const char *name, int *buf, int N);
void double_buffer_to_str(char *output, const char *name, double *buf, int N);

#endif /* GENERATOR_H */
