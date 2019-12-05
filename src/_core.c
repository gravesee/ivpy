#include <Python.h>
#include <numpy/arrayobject.h>

#include "stdio.h"
#include "variable.h"
#include "queue.h"
#include "xtab.h"
#include "bin.h"

static PyObject * c_discretize(PyObject *self, PyObject *args) {

  PyArrayObject *array_x;
  PyArrayObject *array_y;
  PyArrayObject *array_w;

  double miniv;
  int mincnt;
  int minres;
  int maxbin;
  int mono;

  PyArrayObject *array_e;

  if (!PyArg_ParseTuple(args, "OOOdiiiiO", &array_x, &array_y, &array_w, &miniv, &mincnt, &minres, &maxbin, &mono, &array_e)) {
    return NULL;
  }

  // check all 1-d
  if (array_x->nd != 1 && array_y->nd != 1 && array_w->nd != 1 && array_e->nd != 1) {
    PyErr_SetString(PyExc_ValueError, "x, y, w, and exceptions must all have a single dimension");
    return NULL;
  }

  // check all same length
  int len = array_x->dimensions[0];
  if (array_y->dimensions[0] != len && array_w->dimensions[0] != len) {
    PyErr_SetString(PyExc_ValueError, "x, y, and w must all be the same length");
  }


  // check all dtype float
  if (array_x->descr->type_num != PyArray_DOUBLE ||
      array_y->descr->type_num != PyArray_DOUBLE ||
      array_y->descr->type_num != PyArray_DOUBLE ||
      array_e->descr->type_num != PyArray_DOUBLE)  {
    PyErr_SetString(PyExc_ValueError, "x, y, w, and exceptions must all be of type np.float");
    return NULL;
  }
 
  
  struct variable* v = variable_factory((double *) array_x->data, len);
  
  struct xtab* xtab = xtab_factory(v, (double *) array_y->data, (double *) array_w->data); // create the xtab
  double* grand_tots = get_xtab_totals(xtab, 0, xtab->size);

  struct queue* q = queue_factory(); // create the queue
  struct work w = {0, xtab->size - 1}; // last index is one less than the size
  enqueue(q, w);

  // create a vector to store the split rows and init to zero
  size_t* breaks = calloc(xtab->size, sizeof(size_t));
  int num_bins = 1;

  // populate options structure
  struct opts o = {miniv, mincnt, minres, maxbin, mono, array_e};

  // bin the variable until it's done
  while(!is_empty(q)) {
    struct work w = dequeue(q); // take work from queue
    size_t split = find_best_split(w.start, w.stop, xtab, grand_tots, &o);

    if ((split != -1) & (num_bins < o.max_bin)) { // split found!
      num_bins++;
      breaks[split] = 1; // update breaks array
      struct work w1 = {w.start, split};
      struct work w2 = {split + 1, w.stop};
      enqueue(q, w1); // add work to queue
      enqueue(q, w2);
    }
  }

  // return breaks in an python list
  PyObject * py_breaks = PyList_New(0); // create an empty list
  if (!py_breaks) { // check that allocation was successfull
    PyErr_SetString(PyExc_ValueError, "error allocating return list.");
    return NULL;
  }

  
  for(size_t i = 0; i < xtab->size; i++) {
    if (breaks[i] == 1) {
      double val = (xtab->values[i] + xtab->values[i+1]) / 2;
      int err = PyList_Append(py_breaks, PyFloat_FromDouble(val));
      if (err) {
        PyErr_SetString(PyExc_ValueError, "error appending values to return list.");
        return NULL;
      }
    }
  }

  // Release resources
  release_variable(v);
  release_xtab(xtab);
  release_queue(q);
  free(breaks);
  free(grand_tots);

  // Py_DECREF(py_breaks);
  return py_breaks;

}

size_t find_best_split(int start, int stop, struct xtab* xtab, double* grand_tot, struct opts* o) {

  //double* tot = get_xtab_totals(xtab, start, stop + 1);
  double tot[2] = {0};
  double asc[2] = {0}, dsc[2] = {0};
  double best_iv = -1;
  int valid = 0;
  size_t best_split_idx = -1;
  
  int n_exceptions = o->except->dimensions[0];

  // need totals without exceptions
  for (size_t i = start; i <= stop; i++) {
    int skip = 0;
    if (n_exceptions > 0) {
      for (size_t j = 0; j < n_exceptions; j++){
        if (xtab->values[i] == ((double *) o->except->data)[j * o->except->strides[0]]) skip = 1;
      }
    }

    if (!skip) {
      tot[0] += xtab->zero_ct[i];
      tot[1] += xtab->ones_ct[i];
    }
  }


  // now get cumulative counts
  for (size_t i = start; i <= stop; i++) {
    valid = 0;

    int skip = 0;
    if (n_exceptions > 0) {
      for (size_t j = 0; j < n_exceptions; j++){
        // Rprintf("exception: %f\n", REAL(o.except)[j]);
        if (xtab->values[i] == ((double *) o->except->data)[j * o->except->strides[0]]) skip = 1;
      }
    }

    if (!skip) {
      asc[0] += xtab->zero_ct[i];
      asc[1] += xtab->ones_ct[i];
      dsc[0] = tot[0] - asc[0];
      dsc[1] = tot[1] - asc[1];
    }

    struct iv iv = calc_iv(asc, dsc, grand_tot);
    int woe_sign = (iv.asc_woe > iv.dsc_woe) ? 1 : -1;

    if ((asc[0] + asc[1]) < o->min_cnt) { // minsplit
      valid = -1;
    } else if ((dsc[0] + dsc[1]) < o->min_cnt) { // minsplit
      valid = -1;
    } else if (isinf(iv.iv) | isnan(iv.iv)) { // infinite or nan iv
      valid = -1;
    } else if (iv.iv < o->min_iv) { // min iv
      valid = -1;
    } else if ((asc[1] < o->min_res) | (dsc[1] < o->min_res))  {
      valid = -1;
    } else if ((o->mono == 1) | (o->mono == -1)) {
      if (woe_sign != o->mono) {
        valid = -1;
      }
    }

    if ((valid != -1) & (iv.iv > best_iv)) {
      best_iv = iv.iv;
      best_split_idx = i;
      if (o->mono == 2) o->mono = woe_sign;
    }
  }

  //free(tot);
  return best_split_idx;
}

struct iv calc_iv(double* asc_cnts, double* dsc_cnts, double* tots) {
  struct iv iv = {0};
  iv.asc_woe = log((asc_cnts[0]/tots[0])/(asc_cnts[1]/tots[1]));
  iv.dsc_woe = log((dsc_cnts[0]/tots[0])/(dsc_cnts[1]/tots[1]));

  iv.asc_iv  = iv.asc_woe * (asc_cnts[0]/tots[0] - asc_cnts[1]/tots[1]);
  iv.dsc_iv  = iv.dsc_woe * (dsc_cnts[0]/tots[0] - dsc_cnts[1]/tots[1]);
  iv.iv = iv.asc_iv + iv.dsc_iv;

  return iv;
}




static PyMethodDef module_methods[] = {
	{"c_discretize", c_discretize,	METH_VARARGS,
	 "discretize continuous array using information value"}, 
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef _coremodule = {
    PyModuleDef_HEAD_INIT,
    "_core", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__core(void)
{
    return PyModule_Create(&_coremodule);
}