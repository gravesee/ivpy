#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>


double calcSD(double data[], int size)
{
    int i;
    double sum = 0.0, mean, StandardDeviation = 0.0;

    for (i=0; i < size; ++i) {
        sum += data[i];
    }

    mean = sum/size;
    for(i=0; i < size; ++i) {
        StandardDeviation += pow(data[i] - mean, 2);
    }
    return sqrt(StandardDeviation/(size - 1));
}

static PyObject * std_standard_dev(PyObject *self, PyObject* args)
{
    PyObject *float_list;
    int pr_length;
    double *pr;

    if (!PyArg_ParseTuple(args, "O", &float_list))
        return NULL;
    pr_length = PyObject_Length(float_list);
    if (pr_length < 0)
        return NULL;
    pr = (double *) malloc(sizeof(double *) * pr_length);
    if (pr == NULL)
        return NULL;
    for (int index = 0; index < pr_length; index++) {
        PyObject *item;
        item = PyList_GetItem(float_list, index);
        if (!PyFloat_Check(item))
            pr[index] = 0.0;
        pr[index] = PyFloat_AsDouble(item);
    }
    return PyFloat_FromDouble(calcSD(pr, pr_length));
}

// Help from this article: http://folk.uio.no/inf3330/scripting/doc/python/NumPy/Numeric/numpy-13.html
static PyObject * std_standard_dev_numpy(PyObject *self, PyObject* args)
{

    PyArrayObject *array;
    int n;
    
    if (!PyArg_ParseTuple(args, "O", &array))
        return NULL;
    
    if (array->nd != 1 && array->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "array must be single dimension and of type float");
    }

    n = array->dimensions[0];
       
    return PyFloat_FromDouble(calcSD((double *) array->data, n));
    
}


static PyMethodDef std_methods[] = {
	{"standard_dev", std_standard_dev,	METH_VARARGS,    
	 "Return the standard deviation of a list."},
    {"standard_dev_numpy", std_standard_dev_numpy, METH_VARARGS,
     "Return the standard deviation of a numpy array."},
	{NULL,		NULL}		/* sentinel */
};

PyModuleDef stdmodule = {
    PyModuleDef_HEAD_INIT,
    "std",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,
    std_methods
};

PyMODINIT_FUNC PyInit_std(void)
{
    return PyModule_Create(&stdmodule);
}
