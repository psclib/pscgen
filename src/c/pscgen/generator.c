#include "generator.h"

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
    dict->D = (double *)calloc(D_rows*max_D_cols, sizeof(double));
    dict->Vt = (double *)calloc(alpha*s_stride*D_rows, sizeof(double));
    dict->VD = (double *)calloc(alpha*s_stride*max_D_cols, sizeof(double));
    dict->tables = (uint16_t *)calloc(alpha*beta*gamma, sizeof(uint16_t));

    dict_to_file(dict, output_path);
    delete_dict(dict);
}

char* pipeline_to_str(Pipeline *pipeline)
{
    int len, s_stride;
    int max_str_sz, str_sz;
    char *str, *output_str, *final_str, *nnu_str, *svm_str;
    
    len = 0;

    /* str width representations */
    str_sz = 100000;

    /* compue is max buffer size */
    max_str_sz = str_sz;

    s_stride = storage_stride(pipeline->nnu->storage);

    /* allocate str buffers */
    output_str = (char *)malloc(sizeof(char) * max_str_sz);
    str = (char *)malloc(sizeof(char) * str_sz);
    final_str = (char *)malloc(sizeof(char) * 2000000);

    /* create #defines */
    len += sprintf(final_str + len, "#define WS %d\n",
                   pipeline->ws);
    len += sprintf(final_str + len, "#define SS %d\n",
                   pipeline->ss);
    len += sprintf(final_str + len, "#define NUM_FEATURES %d\n",
                   pipeline->svm->num_features);
    len += sprintf(final_str + len, "#define NUM_CLASSES %d\n",
                   pipeline->svm->num_classes);
    len += sprintf(final_str + len, "#define NUM_CLFS %d\n", 
                   pipeline->svm->num_clfs);
    len += sprintf(final_str + len, "#define ALPHA %d\n", 
                   pipeline->nnu->alpha);
    len += sprintf(final_str + len, "#define BETA %d\n", pipeline->nnu->beta);
    len += sprintf(final_str + len, "#define ATOMS %d\n",
                   pipeline->nnu->D_cols);
    len += sprintf(final_str + len, "#define S_STRIDE %d\n", s_stride);

    /* add STANDALONE_H */
    len += sprintf(final_str + len, "%s\n", standalone_h);

    /* get nnu str */
    nnu_str = dict_to_str(pipeline->nnu);
    len += sprintf(final_str + len, "%s\n", nnu_str);

    /* get svm str */
    svm_str = svm_to_str(pipeline->svm);
    len += sprintf(final_str + len, "%s\n", svm_str);

    /* window_X */
    double_buffer_to_str(output_str, "window_X", pipeline->window_X,
                         pipeline->ws);
    len += sprintf(final_str + len, "%s", output_str);  
    len += sprintf(final_str + len, "\n");

    /* bag_X */
    double_buffer_to_str(output_str, "bag_X", pipeline->bag_X,
                         pipeline->nnu->D_cols);
    len += sprintf(final_str + len, "%s", output_str);  
    len += sprintf(final_str + len, "\n");

    /* header */
    strcpy(output_str, "Pipeline pipeline = {");

    /* ws */
    snprintf(str, str_sz, "%d", pipeline->ws);
    strcat(output_str, str);
    strcat(output_str, ",");

    /* ss */
    snprintf(str, str_sz, "%d", pipeline->ss);
    strcat(output_str, str);
    strcat(output_str, ",");

    strcat(output_str, "window_X,bag_X,&dict,&svm};");

    len += sprintf(final_str + len, "%s", output_str);  

    /* clean-up */
    free(output_str);
    free(str);

    return final_str;
}

char* svm_to_str(SVM *svm)
{
    int len;
    int max_str_sz, str_sz;
    char *str, *output_str, *final_str;
    
    len = 0;

    /* str width representations */
    str_sz = 1000000;

    /* compue is max buffer size */
    max_str_sz = str_sz;

    /* allocate str buffers */
    output_str = (char *)malloc(sizeof(char) * max_str_sz);
    str = (char *)malloc(sizeof(char) * str_sz);
    final_str = (char *)malloc(sizeof(char) * 1000000);


    /* coefs */
    double_buffer_to_str(output_str, "coefs", svm->coefs,
                         svm->num_clfs*svm->num_features);
    len += sprintf(final_str + len, "%s", output_str);  
    len += sprintf(final_str + len, "\n");

    /* intercepts */
    double_buffer_to_str(output_str, "intercepts", svm->intercepts,
                         svm->num_clfs);
    len += sprintf(final_str + len, "%s", output_str);  
    len += sprintf(final_str + len, "\n");

    /* header */
    strcpy(output_str, "SVM svm = {");

    /* num_features */
    snprintf(str, str_sz, "%d", svm->num_features);
    strcat(output_str, str);
    strcat(output_str, ",");

    /* num_classes */
    snprintf(str, str_sz, "%d", svm->num_classes);
    strcat(output_str, str);
    strcat(output_str, ",");

    /* num_clfs */
    snprintf(str, str_sz, "%d", svm->num_clfs);
    strcat(output_str, str);
    strcat(output_str, ",");

    strcat(output_str, "coefs,intercepts};");

    len += sprintf(final_str + len, "%s", output_str);  

    /* clean-up */
    free(output_str);
    free(str);

    return final_str;
}


char* dict_to_str(NNUDictionary *dict)
{
    int abg, int_width, float_width, s_stride, len;
    int D_str_sz, table_str_sz, max_str_sz, str_sz;
    char *str, *dict_str, *output_str, *final_str;

    len = 0;
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
    final_str = (char *)malloc(sizeof(char) * 1000000);

    /* tables */
    uint16_buffer_to_str(output_str, "nnu_table", dict->tables, abg);
    len += sprintf(final_str + len, "%s", output_str);  
    len += sprintf(final_str + len, "\n");

    /* D */
    double_buffer_to_str(output_str, "D", dict->D, dict->D_rows*dict->D_cols);
    len += sprintf(final_str + len, "%s", output_str);  
    len += sprintf(final_str + len, "\n");

    /* D_mean */
    double_buffer_to_str(output_str, "D_mean", dict->D_mean, dict->D_cols);
    len += sprintf(final_str + len, "%s", output_str);  
    len += sprintf(final_str + len, "\n");


    /* Vt */
    double_buffer_to_str(output_str, "Vt", dict->Vt,
                         dict->alpha*s_stride*dict->D_rows);
    len += sprintf(final_str + len, "%s", output_str);  
    len += sprintf(final_str + len, "\n");

    /* VD */
    double_buffer_to_str(output_str, "VD", dict->VD,
                         dict->alpha*s_stride*dict->D_cols);
    len += sprintf(final_str + len, "%s", output_str);  
    len += sprintf(final_str + len, "\n");


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

    strcat(dict_str, "nnu_table,D,D_mean,");

    /* D_rows */
    snprintf(str, str_sz, "%d", dict->D_rows);
    strcat(dict_str, str);
    strcat(dict_str, ",");

    /* D_cols */
    snprintf(str, str_sz, "%d", dict->D_cols);
    strcat(dict_str, str);
    strcat(dict_str, ",");
  
    strcat(dict_str, "Vt,VD};");

    len += sprintf(final_str + len, "%s", dict_str);  

    /* clean-up */
    free(output_str);
    free(dict_str);
    free(str);

    return final_str;
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

void int_buffer_to_str(char *output, const char *name, int *buf, int N)
{
    int i; 
    char *str = (char *)malloc(sizeof(char)*16);

    sprintf(output, "int %s[%d] = {", name, N);
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
