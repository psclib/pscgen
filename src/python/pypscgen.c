#include <stdio.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include "nnu.h"

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
    int alpha, beta;
    const char *input_csv_path, *delimiters;

    if(!PyArg_ParseTuple(args, "iiss", &alpha, &beta, &input_csv_path,
                         &delimiters))
        return NULL;

    //Internal call
    NNUDictionary *dict = new_dict(alpha, beta, input_csv_path, delimiters);

    //Create return tuple
    PyObject *result = PyTuple_New(7);
    if(!result)
        return NULL;

    //Convert buffers to pyobjects
    PyObject *D = d_to_pobj(dict->D, dict->D_rows*dict->D_cols);
    PyObject *Vt = d_to_pobj(dict->Vt, dict->D_rows*dict->alpha);
    PyObject *VD = d_to_pobj(dict->VD, dict->D_rows*dict->D_cols);
    PyObject *beta_scale = i_to_pobj(dict->beta_scale, dict->alpha);
    PyObject *tables = uint16_to_pobj(dict->tables, alpha*beta*USHRT_MAX);
    PyObject *D_rows = PyInt_FromLong(dict->D_rows);
    PyObject *D_cols = PyInt_FromLong(dict->D_cols);

    //Error handling
    if(!D || !Vt || !VD || !beta_scale || !tables || !D_rows || !D_cols) {
        Py_DECREF(D);
        Py_DECREF(Vt);
        Py_DECREF(VD);
        Py_DECREF(beta_scale);
        Py_DECREF(tables);
        Py_DECREF(D_rows);
        Py_DECREF(D_cols);
        Py_DECREF(result);
        return NULL;
    }

    //Pack return tuple
    PyTuple_SetItem(result, 0, D);
    PyTuple_SetItem(result, 1, D_rows);
    PyTuple_SetItem(result, 2, D_cols);
    PyTuple_SetItem(result, 3, tables);
    PyTuple_SetItem(result, 4, Vt);
    PyTuple_SetItem(result, 5, VD);
    PyTuple_SetItem(result, 6, beta_scale);

    //clean-up
    delete_dict(dict);

    return result;
}

static PyObject* p_nnu(PyObject *self, PyObject *args)
{
    int alpha, beta, X_rows, X_cols, D_rows, D_cols;
    PyObject *X_obj, *D_obj, *tables_obj, *Vt_obj, *VD_obj, *beta_scale_obj; 

    if(!PyArg_ParseTuple(args, "iiOiiOOOOOii", &alpha, &beta, &D_obj, &D_rows,
                         &D_cols, &tables_obj, &Vt_obj, &VD_obj,
                         &beta_scale_obj, &X_obj, &X_rows, &X_cols))
        return NULL;

    PyObject *D_array = PyArray_FROM_OTF(D_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *tables_array = PyArray_FROM_OTF(tables_obj, NPY_INT16,
                                              NPY_IN_ARRAY);
    PyObject *Vt_array = PyArray_FROM_OTF(Vt_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *VD_array = PyArray_FROM_OTF(VD_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *beta_scale_array = PyArray_FROM_OTF(beta_scale_obj, NPY_INT,
                                                  NPY_IN_ARRAY);
    PyObject *X_array = PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_IN_ARRAY);


    //Error handling
    if(!D_array || !tables_array || !Vt_array || !VD_array ||
       !beta_scale_array || !X_array) {
        Py_XDECREF(D_array);
        Py_XDECREF(tables_array);
        Py_XDECREF(Vt_array);
        Py_XDECREF(VD_array);
        Py_XDECREF(beta_scale_array);
        Py_XDECREF(X_array);
        return NULL;
    }

    //Get pointers to the data
    double *D = (double*)PyArray_DATA(D_array);
    uint16_t *tables = (uint16_t*)PyArray_DATA(tables_array);
    double *Vt = (double*)PyArray_DATA(Vt_array);
    double *VD = (double*)PyArray_DATA(VD_array);
    double *X = (double*)PyArray_DATA(X_array);
    int *beta_scale = (int*)PyArray_DATA(beta_scale_array);

    //create NNUDictionary
    NNUDictionary dict = {alpha, beta, tables, D, D_rows, D_cols, Vt, VD,
                          beta_scale};

    //Start timer
    struct timespec start, end, diff;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
 
    double avg_ab = 0;

    //Internal call
    double *nbrs = nnu(&dict, X, X_rows, X_cols, &avg_ab);

    //End timer
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    diff = t_diff(start, end);

    PyObject *result = PyTuple_New(4);
    if(!result)
        return NULL;

    //create return object
    PyObject *nbrs_obj = d_to_pobj(nbrs, X_cols);
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
    Py_DECREF(tables_array);
    Py_DECREF(Vt_array);
    Py_DECREF(VD_array);
    Py_DECREF(X_array);
    Py_DECREF(beta_scale_array);

    return result;
}

static PyMethodDef module_methods[] = {
    {"new_dict", p_new_dict, METH_VARARGS, "Create new dict."},
    {"encode", p_nnu, METH_VARARGS, "Encode with dictionary."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initlibpypscgen()
{
    PyObject *m = Py_InitModule("libpypscgen", module_methods);
    if (m == NULL)
        return;

    import_array();
}
