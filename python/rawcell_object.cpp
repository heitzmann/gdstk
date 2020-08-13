/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* rawcell_object_str(RawCellObject* self) {
    char buffer[256];
    snprintf(buffer, COUNT(buffer),
             "RawCell '%s' with %" PRId64 " bytes and %" PRId64 " dependencies",
             self->rawcell->name, self->rawcell->size, self->rawcell->dependencies.size);
    return PyUnicode_FromString(buffer);
}

static void rawcell_object_dealloc(RawCellObject* self) {
    RawCell* rawcell = self->rawcell;
    if (rawcell) {
        rawcell->clear();
        free(rawcell);
    }
    PyObject_Del(self);
}

static int rawcell_object_init(RawCellObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"name", NULL};
    char* name = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:RawCell", (char**)keywords, &name)) return -1;
    int64_t len = strlen(name) + 1;
    RawCell* rawcell = self->rawcell;
    if (rawcell) {
        rawcell->clear();
    } else {
        self->rawcell = (RawCell*)calloc(1, sizeof(RawCell));
        rawcell = self->rawcell;
    }
    rawcell->name = (char*)malloc(sizeof(char) * len);
    memcpy(rawcell->name, name, len);
    rawcell->owner = self;
    return 0;
}

static PyObject* rawcell_object_dependencies(RawCellObject* self, PyObject* args) {
    int recursive;
    if (!PyArg_ParseTuple(args, "p:dependencies", &recursive)) return NULL;
    Array<RawCell*> rawcell_array = {0};
    self->rawcell->get_dependencies(recursive > 0, rawcell_array);
    PyObject* result = PyList_New(rawcell_array.size);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create list.");
        rawcell_array.clear();
        return NULL;
    }
    RawCell** rawcell = rawcell_array.items;
    for (int64_t i = 0; i < rawcell_array.size; i++, rawcell++) {
        PyObject* rawcell_obj = (PyObject*)(*rawcell)->owner;
        Py_INCREF(rawcell_obj);
        PyList_SET_ITEM(result, i, rawcell_obj);
    }
    rawcell_array.clear();
    return result;
}

static PyMethodDef rawcell_object_methods[] = {
    {"dependencies", (PyCFunction)rawcell_object_dependencies, METH_VARARGS,
     rawcell_object_dependencies_doc},
    {NULL}};

PyObject* rawcell_object_get_name(RawCellObject* self, void*) {
    PyObject* result = PyUnicode_FromString(self->rawcell->name);
    if (!result) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert value to string.");
        return NULL;
    }
    return result;
}

PyObject* rawcell_object_get_size(RawCellObject* self, void*) {
    PyObject* result = PyLong_FromLong(self->rawcell->size);
    if (!result) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert value to long.");
        return NULL;
    }
    return result;
}

static PyGetSetDef rawcell_object_getset[] = {
    {"name", (getter)rawcell_object_get_name, NULL, rawcell_object_name_doc, NULL},
    {"size", (getter)rawcell_object_get_size, NULL, rawcell_object_size_doc, NULL},
    {NULL}};
