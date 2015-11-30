#include "nnu_generator.h"

void generate_nnu(const char *D_path, const char *output_path, const int alpha,
                  const int beta, Storage_Scheme storage)
{
    const char *delimiters = ",";
    NNUDictionary *dict = new_dict(alpha, beta, storage, D_path, delimiters);
    dict_to_file(dict, output_path);
    delete_dict(dict);
}

void generate_empty_nnu(const char *output_path, const int alpha,
                        const int beta, const int D_rows, 
                        const int max_D_cols, Storage_Scheme storage)
{
    int gamma, s_stride;
    NNUDictionary *dict;

    gamma = ipow(2, storage_gamma_pow(storage));
    s_stride = storage_stride(storage);
    dict = (NNUDictionary *)malloc(sizeof(NNUDictionary));
    dict->alpha = alpha;
    dict->beta = beta;
    dict->gamma = gamma;
    dict->D_rows = D_rows;
    dict->D_cols = 0;
    dict->D = (double *)malloc(D_rows*max_D_cols*sizeof(double));
    dict->Vt = (double *)malloc(alpha*s_stride*D_rows*sizeof(double));
    dict->VD = (double *)malloc(alpha*s_stride*max_D_cols*sizeof(double));
    dict->tables = (uint16_t *)malloc(alpha*beta*gamma*sizeof(uint16_t));

    dict_to_file(dict, output_path);
    delete_dict(dict);
}


void dict_to_file(NNUDictionary *dict, const char* output_path)
{
    int i, abg, int_width, float_width, s_stride, num_src;
    int D_str_sz, table_str_sz, max_str_sz, str_sz;
    char *str, *dict_str, *output_str;
    FILE *output_fp, *tmp_fp;

    /* Source files to include in generated .h file */
    const char *src_dir = "../../../src/c/pscgen/";
    const char * const src_files[] = {"nnu_storage.h", "nnu_storage.c",
                                      "nnu_standalone.h"};
    num_src = 3;
    output_fp = fopen(output_path, "w+");  

    s_stride = storage_stride(dict->storage);

    /* alpha * beta * gamma */
    abg = dict->alpha * dict->beta * dict->gamma;

    /* str width representations */
    float_width = 10;
    int_width = 6;
    str_sz = 10000;

    /* compue is max buffer size */
    D_str_sz = dict->D_rows * dict->D_cols * float_width + 100;
    table_str_sz = abg * int_width + 100;
    max_str_sz = D_str_sz > table_str_sz ? D_str_sz : table_str_sz;

    /* allocate str buffers */
    output_str = (char *)malloc(sizeof(char) * max_str_sz);
    dict_str = (char *)malloc(sizeof(char) * 1000);
    str = (char *)malloc(sizeof(char) * str_sz);

    /* create #defines */
    fprintf(output_fp, "#define STANDALONE\n");
    fprintf(output_fp, "#define ALPHA %d\n", dict->alpha);
    fprintf(output_fp, "#define BETA %d\n", dict->beta);
    fprintf(output_fp, "#define ATOMS %d\n", dict->D_cols);
    fprintf(output_fp, "#define S_STRIDE %d\n", s_stride);

    /* Add all source files to generated .h file*/
    for(i = 0; i < num_src; i++) {
        strcpy(str, src_dir);
        strcat(str, src_files[i]);
        tmp_fp = fopen(str, "r");

        while(fgets(str, str_sz, tmp_fp) != NULL) {
            fprintf(output_fp, "%s", str);
        }
        fprintf(output_fp, "\n");
    }

    /* tables */
    uint16_buffer_to_str(output_str, "nnu_table", dict->tables, abg);
    fprintf(output_fp, "%s", output_str);  
    fprintf(output_fp, "\n");

    /* D */
    double_buffer_to_str(output_str, "D", dict->D, dict->D_rows*dict->D_cols);
    fprintf(output_fp, "%s", output_str);  
    fprintf(output_fp, "\n");

    /* Vt */
    double_buffer_to_str(output_str, "Vt", dict->Vt,
                         dict->alpha*s_stride*dict->D_rows);
    fprintf(output_fp, "%s", output_str);  
    fprintf(output_fp, "\n");

    /* VD */
    double_buffer_to_str(output_str, "VD", dict->VD,
                         dict->alpha*s_stride*dict->D_cols);
    fprintf(output_fp, "%s", output_str);  
    fprintf(output_fp, "\n");


    /* header */
    strcpy(dict_str, "NNUDictionary dict = {");

    /* alpha */
    snprintf(str, str_sz, "%d", dict->alpha);
    strcat(dict_str, str);
    strcat(dict_str, ",");

    /* beta */
    snprintf(str, str_sz, "%d", dict->beta);
    strcat(dict_str, str);
    strcat(dict_str, ",");

    /* gamma */
    snprintf(str, str_sz, "%d", dict->gamma);
    strcat(dict_str, str);
    strcat(dict_str, ",");

    /* storage */
    strcat(dict_str, print_storage(dict->storage));
    strcat(dict_str, ",");

    strcat(dict_str, "nnu_table,D,");

    /* D_rows */
    snprintf(str, str_sz, "%d", dict->D_rows);
    strcat(dict_str, str);
    strcat(dict_str, ",");

    /* D_cols */
    snprintf(str, str_sz, "%d", dict->D_cols);
    strcat(dict_str, str);
    strcat(dict_str, ",");
  
    strcat(dict_str, "Vt,VD};");

    fprintf(output_fp, "%s", dict_str);  
    fclose(output_fp);

    /* clean-up */
    free(output_str);
    free(dict_str);
    free(str);
}


void double_buffer_to_str(char *output, const char *name, double *buf, int N)
{
    int i; 
    char *str = (char *)malloc(sizeof(char)*16);

    sprintf(output, "double %s[%d] = {", name, N);
    for(i = 0; i < N-1; i++) {
        snprintf(str, 16, "%2.6f", buf[i]);
        strcat(output, str);
        strcat(output, ",");
    }
    snprintf(str, 16, "%2.6f", buf[i]);
    strcat(output, str);
    strcat(output, "};");

    /* clean-up */
    free(str);
}

void uint16_buffer_to_str(char *output, const char *name, uint16_t *buf, int N)
{
    int i; 
    char *str = (char *)malloc(sizeof(char)*16);

    sprintf(output, "uint16_t %s[%d] = {", name, N);
    for(i = 0; i < N-1; i++) {
        snprintf(str, 16, "%d", buf[i]);
        strcat(output, str);
        strcat(output, ",");
    }
    snprintf(str, 16, "%d", buf[i]);
    strcat(output, str);
    strcat(output, "};");

    /* clean-up */
    free(str);
}
