#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include "generator.h"
#include "nnu_dict.h"
#include "classifier.h"
#include "pipeline.h"

PyObject* uint16_to_pobj(uint16_t *buf, int N)
{
    int i;
    PyObject *lst = PyList_New(N);

    if(!lst)
        return NULL;

    for(i = 0; i < N; i++) {
        PyObject *num = Py_BuildValue("H", buf[i]);
        if(!num) {
            Py_DECREF(lst);
            return NULL;
        }
        PyList_SET_ITEM(lst, i, num);
    }

    return lst;
}


PyObject* i_to_pobj(int *buf, int N)
{
    int i;
    PyObject *lst = PyList_New(N);

    if(!lst)
        return NULL;

    for(i = 0; i < N; i++) {
        PyObject *num = PyInt_FromLong(buf[i]);
        if(!num) {
            Py_DECREF(lst);
            return NULL;
        }
        PyList_SET_ITEM(lst, i, num);
    }

    return lst;
}


PyObject* d_to_pobj(double *buf, int N)
{
    int i;
    PyObject *lst = PyList_New(N);

    if(!lst)
        return NULL;

    for(i = 0; i < N; i++) {
        PyObject *num = PyFloat_FromDouble(buf[i]);
        if(!num) {
            Py_DECREF(lst);
            return NULL;
        }
        PyList_SET_ITEM(lst, i, num);
    }

    return lst;
}


static PyObject* p_new_dict(PyObject *self, PyObject *args)
{
    int alpha, beta, max_atoms, gamma_pow, storage;
    const char *input_csv_path, *delimiters;

    if(!PyArg_ParseTuple(args, "iiiiiss", &alpha, &beta, &gamma_pow, &storage,
                         &max_atoms, &input_csv_path, &delimiters))
        return NULL;

    int s_stride = storage_stride(storage);

    //Internal call
    NNUDictionary *dict = new_dict(alpha, beta, max_atoms, storage,
                                   input_csv_path, delimiters);

    //Create return tuple
    PyObject *result = PyTuple_New(7);
    if(!result)
        return NULL;

    //Convert buffers to pyobjects
    PyObject *D = d_to_pobj(dict->D, dict->D_rows*dict->D_cols);
    PyObject *D_mean = d_to_pobj(dict->D_mean, dict->D_cols);
    PyObject *Vt = d_to_pobj(dict->Vt, dict->alpha*s_stride * dict->D_rows);
    PyObject *VD = d_to_pobj(dict->VD, dict->alpha*s_stride * dict->D_cols);
    PyObject *tables = uint16_to_pobj(dict->tables, alpha*beta*USHRT_MAX);
    PyObject *D_rows = PyInt_FromLong(dict->D_rows);
    PyObject *D_cols = PyInt_FromLong(dict->D_cols);

    //Error handling
    if(!D || !D_mean || !Vt || !VD || !tables || !D_rows || !D_cols) {
        Py_DECREF(D);
        Py_DECREF(D_mean);
        Py_DECREF(Vt);
        Py_DECREF(VD);
        Py_DECREF(tables);
        Py_DECREF(D_rows);
        Py_DECREF(D_cols);
        Py_DECREF(result);
        return NULL;
    }

    //Pack return tuple
    PyTuple_SetItem(result, 0, D);
    PyTuple_SetItem(result, 1, D_mean);
    PyTuple_SetItem(result, 2, D_rows);
    PyTuple_SetItem(result, 3, D_cols);
    PyTuple_SetItem(result, 4, tables);
    PyTuple_SetItem(result, 5, Vt);
    PyTuple_SetItem(result, 6, VD);

    //clean-up
    delete_dict(dict);

    return result;
}

static PyObject* p_new_dict_from_buffer(PyObject *self, PyObject *args)
{
    int alpha, beta, max_atoms, gamma_pow, storage, D_rows_c, D_cols_c;
    PyObject *D_obj;
   
    if(!PyArg_ParseTuple(args, "iiiiiiiO", &alpha, &beta, &gamma_pow, &storage,
                         &D_rows_c, &D_cols_c, &max_atoms, &D_obj))
        return NULL;

    int s_stride = storage_stride(storage);
    PyObject *D_array = PyArray_FROM_OTF(D_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    
    if(!D_array) {
        Py_DECREF(D_array);
        return NULL;
    }


    double *D_input_buf = (double*)PyArray_DATA(D_array);

    //Internal call
    NNUDictionary *dict = new_dict_from_buffer(alpha, beta, storage,
                                               D_input_buf, D_rows_c,
                                               D_cols_c, max_atoms);

    //clean-up input 
    Py_DECREF(D_input_buf);

    //Create return tuple
    PyObject *result = PyTuple_New(7);
    if(!result)
        return NULL;

    //Convert buffers to pyobjects
    PyObject *D = d_to_pobj(dict->D, dict->D_rows*dict->D_cols);
    PyObject *D_mean = d_to_pobj(dict->D, dict->D_rows*dict->D_cols);
    PyObject *Vt = d_to_pobj(dict->Vt, dict->alpha*s_stride * dict->D_rows);
    PyObject *VD = d_to_pobj(dict->VD, dict->alpha*s_stride * dict->D_cols);
    PyObject *tables = uint16_to_pobj(dict->tables, alpha*beta*dict->gamma);
    PyObject *D_rows = PyInt_FromLong(dict->D_rows);
    PyObject *D_cols = PyInt_FromLong(dict->D_cols);

    //Error handling
    if(!D || !D_mean || !Vt || !VD || !tables || !D_rows || !D_cols) {
        Py_DECREF(D);
        Py_DECREF(D_mean);
        Py_DECREF(Vt);
        Py_DECREF(VD);
        Py_DECREF(tables);
        Py_DECREF(D_rows);
        Py_DECREF(D_cols);
        Py_DECREF(result);
        return NULL;
    }

    //Pack return tuple
    PyTuple_SetItem(result, 0, D);
    PyTuple_SetItem(result, 1, D_mean);
    PyTuple_SetItem(result, 2, D_rows);
    PyTuple_SetItem(result, 3, D_cols);
    PyTuple_SetItem(result, 4, tables);
    PyTuple_SetItem(result, 5, Vt);
    PyTuple_SetItem(result, 6, VD);

    //clean-up
    delete_dict(dict);

    return result;
}


static PyObject* p_nnu(PyObject *self, PyObject *args)
{
   int alpha, beta, max_alpha, max_beta, gamma, storage, X_rows, X_cols,
        D_rows, D_cols, max_atoms;
    PyObject *X_obj, *D_obj, *D_mean_obj, *tables_obj, *Vt_obj, *VD_obj;

    if(!PyArg_ParseTuple(args, "iiiiiiiiiOOOOOOii", &alpha, &beta, &max_alpha,
                         &max_beta, &gamma, &storage, &D_rows, &D_cols,
                         &max_atoms, &D_obj, 
                         &D_mean_obj, &tables_obj, &Vt_obj, &VD_obj, &X_obj,
                         &X_rows, &X_cols))
        return NULL;

    PyObject *D_array = PyArray_FROM_OTF(D_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *D_mean_array = PyArray_FROM_OTF(D_mean_obj, NPY_DOUBLE,
                                              NPY_IN_ARRAY);
    PyObject *tables_array = PyArray_FROM_OTF(tables_obj, NPY_UINT16,
                                              NPY_IN_ARRAY);
    PyObject *Vt_array = PyArray_FROM_OTF(Vt_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *VD_array = PyArray_FROM_OTF(VD_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *X_array = PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_IN_ARRAY);


    //Error handling
    if(!D_array || !D_mean_array || !tables_array || !Vt_array || !VD_array ||
       !X_array) {
        Py_XDECREF(D_array);
        Py_XDECREF(D_mean_array);
        Py_XDECREF(tables_array);
        Py_XDECREF(Vt_array);
        Py_XDECREF(VD_array);
        Py_XDECREF(X_array);
        return NULL;
    }

    //Get pointers to the data
    double *D = (double*)PyArray_DATA(D_array);
    double *D_mean = (double*)PyArray_DATA(D_mean_array);
    uint16_t *tables = (uint16_t*)PyArray_DATA(tables_array);
    double *Vt = (double*)PyArray_DATA(Vt_array);
    double *VD = (double*)PyArray_DATA(VD_array);
    double *X = (double*)PyArray_DATA(X_array);

    //create NNUDictionary
    NNUDictionary dict = {alpha, beta, max_alpha, max_beta, gamma, storage, 
                          D_rows, D_cols, max_atoms, tables, D, D_mean, Vt,
                          VD};


    //Start timer
    struct timespec start, end, diff;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
 
    double avg_ab = 0;

    //Internal call
    int *nbrs = nnu(&dict, alpha, beta, X, X_rows, X_cols, &avg_ab);

    //End timer
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    diff = t_diff(start, end);

    PyObject *result = PyTuple_New(4);
    if(!result)
        return NULL;

    //create return object
    PyObject *nbrs_obj = i_to_pobj(nbrs, X_cols);
    PyObject *time_secs = PyInt_FromLong(diff.tv_sec);
    PyObject *time_nano = PyInt_FromLong(diff.tv_nsec);
    PyObject *avg_ab_obj = PyFloat_FromDouble(avg_ab);

    PyTuple_SetItem(result, 0, nbrs_obj);
    PyTuple_SetItem(result, 1, time_secs);
    PyTuple_SetItem(result, 2, time_nano);
    PyTuple_SetItem(result, 3, avg_ab_obj);

    //clean-up
    free(nbrs);
    Py_DECREF(D_array);
    Py_DECREF(D_mean_array);
    Py_DECREF(tables_array);
    Py_DECREF(Vt_array);
    Py_DECREF(VD_array);
    Py_DECREF(X_array);

    return result;
}

static PyObject* p_nnu_single(PyObject *self, PyObject *args)
{
   int alpha, beta, max_alpha, max_beta, gamma, storage, X_rows, X_cols,
        D_rows, D_cols, max_atoms;
    PyObject *X_obj, *D_obj, *D_mean_obj, *tables_obj, *Vt_obj, *VD_obj;
    const char *enc_type;

    if(!PyArg_ParseTuple(args, "siiiiiiiiiOOOOOOii", &enc_type, &alpha, &beta,
                         &max_alpha,
                         &max_beta, &gamma, &storage, &D_rows, &D_cols,
                         &max_atoms, &D_obj, 
                         &D_mean_obj, &tables_obj, &Vt_obj, &VD_obj, &X_obj,
                         &X_rows, &X_cols))
        return NULL;

    PyObject *D_array = PyArray_FROM_OTF(D_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *D_mean_array = PyArray_FROM_OTF(D_mean_obj, NPY_DOUBLE,
                                              NPY_IN_ARRAY);
    PyObject *tables_array = PyArray_FROM_OTF(tables_obj, NPY_UINT16,
                                              NPY_IN_ARRAY);
    PyObject *Vt_array = PyArray_FROM_OTF(Vt_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *VD_array = PyArray_FROM_OTF(VD_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *X_array = PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_IN_ARRAY);


    //Error handling
    if(!D_array || !D_mean_array || !tables_array || !Vt_array || !VD_array ||
       !X_array) {
        Py_XDECREF(D_array);
        Py_XDECREF(D_mean_array);
        Py_XDECREF(tables_array);
        Py_XDECREF(Vt_array);
        Py_XDECREF(VD_array);
        Py_XDECREF(X_array);
        return NULL;
    }

    //Get pointers to the data
    double *D = (double*)PyArray_DATA(D_array);
    double *D_mean = (double*)PyArray_DATA(D_mean_array);
    uint16_t *tables = (uint16_t*)PyArray_DATA(tables_array);
    double *Vt = (double*)PyArray_DATA(Vt_array);
    double *VD = (double*)PyArray_DATA(VD_array);
    double *X = (double*)PyArray_DATA(X_array);

    //create NNUDictionary
    NNUDictionary dict = {alpha, beta, max_alpha, max_beta, gamma, storage, 
                          D_rows, D_cols, max_atoms, tables, D, D_mean, Vt,
                          VD};

    int *ret = malloc(max_alpha*max_beta*sizeof(int));
    int idxs;

    idxs = 0;

    if(strcmp(enc_type, "nnu") == 0) {
        ret[0] = nnu_single(&dict, X, X_rows);
        idxs = 1;
    }
    else if(strcmp(enc_type, "nnu_nodot") == 0) {
        idxs = nnu_single_nodot(&dict, X, X_rows, ret);
    } 
    else if(strcmp(enc_type, "nns") == 0) {
        ret[0] = nns_single(&dict, X, X_rows);
        idxs = 1;
    } 
    else if(strcmp(enc_type, "nnu_pca") == 0) {
        ret[0] = nnu_pca_single(&dict, X, X_rows);
        idxs = 1;
    } 


    PyObject *result = PyTuple_New(1);
    if(!result)
        return NULL;

    PyObject *ret_obj = i_to_pobj(ret, idxs);

    PyTuple_SetItem(result, 0, ret_obj);

    //clean-up
    Py_DECREF(D_array);
    Py_DECREF(D_mean_array);
    Py_DECREF(tables_array);
    Py_DECREF(Vt_array);
    Py_DECREF(VD_array);
    Py_DECREF(X_array);
    free(ret);

    return result;
}


static PyObject* p_generate(PyObject *self, PyObject *args)
{
    int ws, ss, num_features, num_classes, max_classes,
        alpha, beta, gamma, storage, D_rows, D_cols, max_atoms;
    const char *output_path, *enc_type, *float_type;
    PyObject *coef_obj, *intercept_obj, *D_obj, *D_mean_obj, *tables_obj,
             *Vt_obj, *VD_obj;

    if(!PyArg_ParseTuple(args, "sssiiiiiOOiiiiiiiOOOOO", &output_path,
                         &enc_type, &float_type, &ws,
                         &ss, &num_features, &num_classes, &max_classes,
                         &coef_obj, &intercept_obj, &alpha, &beta, &gamma,
                         &storage, &D_rows, &D_cols, &max_atoms, &D_obj, 
                         &D_mean_obj, &tables_obj, &Vt_obj, &VD_obj))
        return NULL;



    PyObject *coef_array = PyArray_FROM_OTF(coef_obj, NPY_DOUBLE,
                                            NPY_IN_ARRAY);
    PyObject *intercept_array = PyArray_FROM_OTF(intercept_obj, NPY_DOUBLE,
                                                 NPY_IN_ARRAY);
    PyObject *D_array = PyArray_FROM_OTF(D_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *D_mean_array = PyArray_FROM_OTF(D_mean_obj, NPY_DOUBLE,
                                              NPY_IN_ARRAY);
    PyObject *tables_array = PyArray_FROM_OTF(tables_obj, NPY_UINT16,
                                              NPY_IN_ARRAY);
    PyObject *Vt_array = PyArray_FROM_OTF(Vt_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *VD_array = PyArray_FROM_OTF(VD_obj, NPY_DOUBLE, NPY_IN_ARRAY);


    //Error handling
    if(!coef_array || !intercept_array || !D_array || !D_mean_array ||
       !tables_array || !Vt_array || !VD_array) {
        Py_XDECREF(coef_array);
        Py_XDECREF(intercept_array);
        Py_XDECREF(D_array);
        Py_XDECREF(D_mean_array);
        Py_XDECREF(tables_array);
        Py_XDECREF(Vt_array);
        Py_XDECREF(VD_array);
        return NULL;
    }

    //Get pointers to the data
    double *coef = (double*)PyArray_DATA(coef_array);
    double *intercept = (double*)PyArray_DATA(intercept_array);
    double *D = (double*)PyArray_DATA(D_array);
    double *D_mean = (double*)PyArray_DATA(D_mean_array);
    uint16_t *tables = (uint16_t*)PyArray_DATA(tables_array);
    double *Vt = (double*)PyArray_DATA(Vt_array);
    double *VD = (double*)PyArray_DATA(VD_array);

    /* create NNUDictionary */
    NNUDictionary dict = {alpha, beta, alpha, beta, gamma,
                          storage, D_rows, D_cols, max_atoms, tables, D,
                          D_mean, Vt, VD};

    /* create SVM */
    SVM *svm = new_svm(num_features, num_classes, max_classes, coef,
                       intercept);

    /* create pipeline */
    Pipeline *pipeline = new_pipeline(&dict, svm, ws, ss);
    
    char *output_str = pipeline_to_str(pipeline, enc_type, float_type);
    FILE *output_fp = fopen(output_path, "w+");  
    fprintf(output_fp, "%s", output_str);
    fclose(output_fp);

    delete_pipeline(pipeline);    
    delete_svm(svm);
    free(output_str);

    Py_DECREF(coef_array);
    Py_DECREF(intercept_array);
    Py_DECREF(D_array);
    Py_DECREF(D_mean_array);
    Py_DECREF(tables_array);
    Py_DECREF(Vt_array);
    Py_DECREF(VD_array);


    return Py_BuildValue("");
}

static PyObject* p_classify(PyObject *self, PyObject *args)
{
    int X_rows, ws, ss, num_features, num_classes, max_classes,
        alpha, beta, gamma, storage, D_rows, D_cols, ret, max_atoms;
    PyObject *X_obj, *coef_obj, *intercept_obj, *D_obj, *D_mean_obj,
             *tables_obj, *Vt_obj, *VD_obj;
    const char *enc_type;

    ret = 0;

    if(!PyArg_ParseTuple(args, "sOiiiiiiOOiiiiiiiOOOOO", &enc_type, &X_obj,
                         &X_rows, &ws, &ss, &num_features, &num_classes,
                         &max_classes, &coef_obj, &intercept_obj, &alpha,
                         &beta, &gamma, &storage, &D_rows, &D_cols,
                         &max_atoms, &D_obj, &D_mean_obj, &tables_obj, 
                         &Vt_obj, &VD_obj))
        return NULL;


    PyObject *X_array = PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *coef_array = PyArray_FROM_OTF(coef_obj, NPY_DOUBLE,
                                            NPY_IN_ARRAY);
    PyObject *intercept_array = PyArray_FROM_OTF(intercept_obj, NPY_DOUBLE,
                                                 NPY_IN_ARRAY);
    PyObject *D_array = PyArray_FROM_OTF(D_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *D_mean_array = PyArray_FROM_OTF(D_mean_obj, NPY_DOUBLE,
                                              NPY_IN_ARRAY);
    PyObject *tables_array = PyArray_FROM_OTF(tables_obj, NPY_UINT16,
                                              NPY_IN_ARRAY);
    PyObject *Vt_array = PyArray_FROM_OTF(Vt_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *VD_array = PyArray_FROM_OTF(VD_obj, NPY_DOUBLE, NPY_IN_ARRAY);


    //Error handling
    if(!X_array || !coef_array || !intercept_array || !D_array ||
       !D_mean_array || !tables_array || !Vt_array || !VD_array) {
        Py_XDECREF(X_array);
        Py_XDECREF(coef_array);
        Py_XDECREF(intercept_array);
        Py_XDECREF(D_array);
        Py_XDECREF(D_mean_array);
        Py_XDECREF(tables_array);
        Py_XDECREF(Vt_array);
        Py_XDECREF(VD_array);
        return NULL;
    }

    //Get pointers to the data
    double *X = (double*)PyArray_DATA(X_array);
    double *coef = (double*)PyArray_DATA(coef_array);
    double *intercept = (double*)PyArray_DATA(intercept_array);
    double *D = (double*)PyArray_DATA(D_array);
    double *D_mean = (double*)PyArray_DATA(D_mean_array);
    uint16_t *tables = (uint16_t*)PyArray_DATA(tables_array);
    double *Vt = (double*)PyArray_DATA(Vt_array);
    double *VD = (double*)PyArray_DATA(VD_array);

    /* create NNUDictionary */
    NNUDictionary dict = {alpha, beta, alpha, beta, gamma, storage, D_rows,
                          D_cols, max_atoms, tables, D, D_mean, Vt, VD};


    /* create SVM */
    SVM *svm = new_svm(num_features, num_classes, max_classes, coef,
                       intercept);

    /* create pipeline */
    Pipeline *pipeline = new_pipeline(&dict, svm, ws, ss);
    
    if(strcmp(enc_type, "nnu") == 0) {
        ret = classification_pipeline(X, X_rows, pipeline);
    }
    else if(strcmp(enc_type, "nnu_nodot") == 0) {
        ret = classification_pipeline_nodot(X, X_rows, pipeline);
    } 
    else if(strcmp(enc_type, "nns") == 0) {
        ret = classification_pipeline_nns(X, X_rows, pipeline);
    }
    else if(strcmp(enc_type, "nnu_pca") == 0) {
        ret = classification_pipeline_pca(X, X_rows, pipeline);
    }
    else {
        ret = -1; /* invalid option -- should be caught earlier */
    }

    delete_pipeline(pipeline);    
    delete_svm(svm);

    PyObject *result = PyTuple_New(1);
    if(!result)
        return NULL;

    //create return object
    PyObject *ret_obj = PyInt_FromLong(ret);
    PyTuple_SetItem(result, 0, ret_obj);

    Py_DECREF(coef_array);
    Py_DECREF(intercept_array);
    Py_DECREF(D_array);
    Py_DECREF(D_mean_array);
    Py_DECREF(tables_array);
    Py_DECREF(Vt_array);
    Py_DECREF(VD_array);
    Py_DECREF(X_array);

    return result;
}


static PyMethodDef module_methods[] = {
    {"build_index_from_file", p_new_dict, METH_VARARGS,
     "Create new NNUDictionary from a filepath."},
    {"build_index", p_new_dict_from_buffer, METH_VARARGS,
     "Create new NNUDictionary from a numpy array."},
    {"index", p_nnu, METH_VARARGS, "Index into NNUDictionary."},
    {"index_single", p_nnu_single, METH_VARARGS, "Index into NNUDictionary."},
    {"generate", p_generate, METH_VARARGS, "Generate standalone Pipeline."},
    {"classify", p_classify, METH_VARARGS, "Run classification pipeline."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initpscgen_c()
{
    PyObject *m = Py_InitModule("pscgen_c", module_methods);
    if (m == NULL)
        return;

    import_array();
}
