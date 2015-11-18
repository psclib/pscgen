#include "nnu_generator.h"

void generate_nnu(const char *D_path, const char *output_path, const int alpha,
                  const int beta, Storage_Scheme storage)
{
    const char *delimiters = ",";
    NNUDictionary *dict = new_dict(alpha, beta, storage, D_path, delimiters);
    dict_to_file(dict, output_path);
}

void dict_to_file(NNUDictionary *dict, const char* output_path)
{
    int i, abg, int_width, float_width, s_stride, num_src;
    int D_str_sz, table_str_sz, max_str_sz, str_sz;
    char *str, *tmp_str, *dict_str, *output_str;
    const char *include_str;
    FILE *output_fp, *tmp_fp;

    /* Source files to include in generated .h file */
    const char *src_dir = "../../../src/c/pscgen/";
    const char * const src_files[] = {"util.h", "util.c", "linalg/linalg.h",
                                      "linalg/linalg.c", "nnu_storage.h", 
                                      "nnu_storage.c",  "nnu_dict.h",
                                      "nnu_dict.c"};
    include_str = "#include \"";
    num_src = 8;
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
    tmp_str = (char *)malloc(sizeof(char) * str_sz);
    strcpy(tmp_str, "#include \"");


    /* Add all source files to generated .h file*/
    for(i = 0; i < num_src; i++) {
        strcpy(str, src_dir);
        strcat(str, src_files[i]);
        tmp_fp = fopen(str, "r");

        while(fgets(str, str_sz, tmp_fp) != NULL) {
            strncpy(tmp_str, str, 10);
            if(strcmp(tmp_str, include_str) != 0) {
                fprintf(output_fp, "%s", str);
            }
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
    snprintf(str, str_sz, "%d", dict->storage);
    strcat(dict_str, str);
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

    free(str);
}

NNUDictionary* new_dict(const int alpha, const int beta,
                        Storage_Scheme storage, const char *input_csv_path,
                        const char *delimiters)
{
    double *D;
    int rows, cols;

    /* read in input dictionary file */
    read_csv(input_csv_path, delimiters, &D, &rows, &cols);

    /* create NNUDictionary using read in file */
    return new_dict_from_buffer(alpha, beta, storage, D, rows, cols);
}

NNUDictionary* new_dict_from_buffer(const int alpha, const int beta,
                                    Storage_Scheme storage, double *D,
                                    int rows, int cols)
{
    int i, j, k, l, idx, gamma, table_idx, s_stride;
    int *idxs;
    double *Dt, *Vt, *Vt_full, *VD, *U, *S, *c;
    float *dv;
    uint16_t *tables;
    NNUDictionary *dict;

    int gamma_pow = storage_gamma_pow(storage);
    gamma = ipow(2, gamma_pow);
    s_stride = storage_stride(storage);
    tables = (uint16_t *)malloc(alpha * beta * gamma * sizeof(uint16_t));
    dv = (float *)malloc(sizeof(float) * s_stride);

    /* transpose D */
    Dt = d_transpose(D, rows, cols);

    /* get eigen vectors of input dictionary file */
    d_SVD(Dt, cols, rows, &U, &S, &Vt_full);

    /* trim to top alpha vectors */
    Vt = d_trim(Vt_full, rows, alpha*s_stride, rows);

    /* compute prod(V, D) */
    VD = dmm_prod(Vt, D, alpha*s_stride, rows, rows, cols);
    
    /* populate nnu tables */
    c = (double *)calloc(cols, sizeof(double));
    idxs = (int *)malloc(sizeof(int) * cols);


    for(i = 0; i < gamma; i++) {
        storage_to_float(dv, i, storage);
        for(j = 0; j < alpha; j++) {
            for(k = 0; k < cols; k++) {
                for(l = 0; l < s_stride; l++) {
                    idx = idx2d(j*s_stride + l, k, alpha*s_stride);
                    c[k] += fabs(VD[idx] - dv[l]);
                }
            }

            d_argsort(c, idxs, cols);
            for(k = 0; k < beta; k++) {
                table_idx = idx3d(j, i, k, beta, gamma);
                tables[table_idx] = idxs[k];
            }

            zero_dvec(c, cols);
        }
    }


    /* Initialze NNUDictionary */
    dict = (NNUDictionary *)malloc(sizeof(NNUDictionary));
    dict->tables = tables;
    dict->alpha = alpha;
    dict->beta = beta;
    dict->gamma = gamma;
    dict->storage = storage;
    dict->D = D;
    dict->D_rows = rows;
    dict->D_cols = cols;
    dict->VD = VD;
    dict->Vt = Vt;

    /* clean-up */
    free(Dt);
    free(U);
    free(S);
    free(Vt_full);
    free(idxs);
    free(dv);
    free(c);

    return dict;
}

void save_dict(char *filepath, NNUDictionary *dict)
{
    int s_stride = storage_stride(dict->storage);

    FILE *fp = fopen(filepath, "w+");
    fwrite(&dict->alpha, sizeof(int), 1, fp);
    fwrite(&dict->beta, sizeof(int), 1, fp);
    fwrite(&dict->gamma, sizeof(int), 1, fp);
    fwrite(&dict->storage, sizeof(Storage_Scheme), 1, fp);
    fwrite(dict->tables, sizeof(uint16_t),
           dict->alpha * dict->beta * dict->gamma, fp);
    fwrite(&dict->D_rows, sizeof(int), 1, fp);
    fwrite(&dict->D_cols, sizeof(int), 1, fp);
    fwrite(dict->D, sizeof(double), dict->D_rows * dict->D_cols, fp);
    fwrite(dict->Vt, sizeof(double), dict->alpha*s_stride * dict->D_rows, fp);
    fwrite(dict->VD, sizeof(double), dict->alpha*s_stride * dict->D_cols, fp);
    fclose(fp);
}

NNUDictionary* load_dict(char *filepath)
{

    int s_stride;
    NNUDictionary *dict = (NNUDictionary *)malloc(sizeof(NNUDictionary));
    FILE *fp = fopen(filepath, "r");
    fread(&dict->alpha, sizeof(int), 1, fp);
    fread(&dict->beta, sizeof(int), 1, fp);
    fread(&dict->gamma, sizeof(int), 1, fp);
    fread(&dict->storage, sizeof(Storage_Scheme), 1, fp);
    s_stride = storage_stride(dict->storage);
    dict->tables = (uint16_t *)malloc(sizeof(uint16_t) * dict->alpha *
                                      dict->beta * dict->gamma);
    fread(dict->tables, sizeof(uint16_t),
          dict->alpha * dict->beta * dict->gamma, fp);
    fread(&dict->D_rows, sizeof(int), 1, fp);
    fread(&dict->D_cols, sizeof(int), 1, fp);
    dict->D = (double *)malloc(sizeof(double) * dict->D_rows * dict->D_cols);
    dict->Vt = (double *)malloc(sizeof(double) * dict->alpha*s_stride *
                                dict->D_rows);
    dict->VD = (double *)malloc(sizeof(double) * dict->alpha*s_stride *
                                dict->D_cols);
    fread(dict->D, sizeof(double), dict->D_rows * dict->D_cols, fp);
    fread(dict->Vt, sizeof(double), dict->alpha * dict->D_rows, fp);
    fread(dict->VD, sizeof(double), dict->alpha * dict->D_cols, fp);
    fclose(fp);

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
