/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* repetition_object_str(RepetitionObject* self) {
    char buffer[64];
    int64_t size = self->repetition.get_size();
    switch (self->repetition.type) {
        case RepetitionType::None:
            snprintf(buffer, COUNT(buffer), "No repetition");
            break;
        case RepetitionType::Rectangular:
            snprintf(buffer, COUNT(buffer), "Repetition (rectangular) of size %" PRId64, size);
            break;
        case RepetitionType::Regular:
            snprintf(buffer, COUNT(buffer), "Repetition (regular) of size %" PRId64, size);
            break;
        case RepetitionType::Explicit:
            snprintf(buffer, COUNT(buffer), "Repetition (explicit) of size %" PRId64, size);
            break;
        case RepetitionType::ExplicitX:
            snprintf(buffer, COUNT(buffer), "Repetition (x-explicit) of size %" PRId64, size);
            break;
        case RepetitionType::ExplicitY:
            snprintf(buffer, COUNT(buffer), "Repetition (y-explicit) of size %" PRId64, size);
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError, "Uknown repetition type.");
            return NULL;
    }
    return PyUnicode_FromString(buffer);
}

static void repetition_object_dealloc(RepetitionObject* self) { PyObject_Del(self); }

static int repetition_object_init(RepetitionObject* self, PyObject* args, PyObject* kwds) {
    PyObject* spacing_obj = Py_None;
    PyObject* v1_obj = Py_None;
    PyObject* v2_obj = Py_None;
    PyObject* offsets_obj = Py_None;
    PyObject* xoff_obj = Py_None;
    PyObject* yoff_obj = Py_None;
    int64_t columns = 0;
    int64_t rows = 0;
    const char* keywords[] = {"columns", "rows",      "spacing",   "v1", "v2",
                              "offsets", "x_offsets", "y_offsets", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|LLOOOOOO:Repetition", (char**)keywords, &columns,
                                     &rows, &spacing_obj, &v1_obj, &v2_obj, &offsets_obj, &xoff_obj,
                                     &yoff_obj))
        return -1;

    Repetition* repetition = &self->repetition;
    repetition->clear();

    if (columns > 0 && rows > 0 && spacing_obj != Py_None) {
        repetition->type = RepetitionType::Rectangular;
        repetition->columns = columns;
        repetition->rows = rows;
        if (parse_point(spacing_obj, repetition->spacing, "spacing") < 0) return -1;
    } else if (columns > 0 && rows > 0 && v1_obj != Py_None && v2_obj != Py_None) {
        repetition->type = RepetitionType::Regular;
        repetition->columns = columns;
        repetition->rows = rows;
        if (parse_point(v1_obj, repetition->v1, "v1") < 0) return -1;
        if (parse_point(v2_obj, repetition->v2, "v2") < 0) return -1;
    } else if (offsets_obj != Py_None) {
        repetition->type = RepetitionType::Explicit;
        if (parse_point_sequence(offsets_obj, repetition->offsets, "offsets") < 0) return -1;
    } else if (xoff_obj != Py_None) {
        repetition->type = RepetitionType::ExplicitX;
        if (parse_double_sequence(xoff_obj, repetition->coords, "x_offsets") < 0) return -1;
    } else if (yoff_obj != Py_None) {
        repetition->type = RepetitionType::ExplicitY;
        if (parse_double_sequence(yoff_obj, repetition->coords, "y_offsets") < 0) return -1;
    } else {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Repetition type undefined. Please define either columns + rows + spacing, columns + rows + v1 + v2, offsets, x_offsets, or y_offsets.");
        return -1;
    }
    return 0;
}

static PyObject* repetition_object_getoffsets(RepetitionObject* self, PyObject* args) {
    Array<Vec2> offsets = {0};
    self->repetition.get_offsets(offsets);
    npy_intp dims[] = {(npy_intp)offsets.size, 2};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_MemoryError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    memcpy(data, offsets.items, sizeof(double) * offsets.size * 2);
    offsets.clear();
    return (PyObject*)result;
}

static PyMethodDef repetition_object_methods[] = {
    // {"copy", (PyCFunction)repetition_object_copy, METH_NOARGS, repetition_object_copy_doc},
    {"get_offsets", (PyCFunction)repetition_object_getoffsets, METH_NOARGS,
     repetition_object_getoffsets_doc},
    {NULL}};

static PyObject* repetition_object_get_size(RepetitionObject* self, void*) {
    return PyLong_FromLong((long)self->repetition.get_size());
}

static PyObject* repetition_object_get_columns(RepetitionObject* self, void*) {
    Repetition* repetition = &self->repetition;
    if (repetition->type == RepetitionType::Rectangular ||
        repetition->type == RepetitionType::Regular) {
        return PyLong_FromLong((long)self->repetition.columns);
    }
    Py_INCREF(Py_None);
    return Py_None;
}

// static int repetition_object_set_columns(RepetitionObject* self, PyObject* arg, void*) {
//     Repetition* repetition = &self->repetition;
//     if (repetition->type != RepetitionType::Rectangular &&
//         repetition->type != RepetitionType::Regular) {
//         repetition->type = RepetitionType::Rectangular;
//     }
//     repetition->columns = PyLong_AsLong(arg);
//     if (PyErr_Occurred()) {
//         PyErr_SetString(PyExc_TypeError, "Unable to convert columns to int.");
//         return -1;
//     }
//     return 0;
// }

static PyObject* repetition_object_get_rows(RepetitionObject* self, void*) {
    Repetition* repetition = &self->repetition;
    if (repetition->type == RepetitionType::Rectangular ||
        repetition->type == RepetitionType::Regular) {
        return PyLong_FromLong((long)repetition->rows);
    }
    Py_INCREF(Py_None);
    return Py_None;
}

// static int repetition_object_set_rows(RepetitionObject* self, PyObject* arg, void*) {
//     Repetition* repetition = &self->repetition;
//     if (repetition->type != RepetitionType::Rectangular &&
//         repetition->type != RepetitionType::Regular) {
//         repetition->type = RepetitionType::Rectangular;
//     }
//     repetition->rows = PyLong_AsLong(arg);
//     if (PyErr_Occurred()) {
//         PyErr_SetString(PyExc_TypeError, "Unable to convert rows to int.");
//         return -1;
//     }
//     return 0;
// }

static PyObject* repetition_object_get_spacing(RepetitionObject* self, void*) {
    Repetition* repetition = &self->repetition;
    if (repetition->type == RepetitionType::Rectangular) {
        PyObject* x = PyFloat_FromDouble(repetition->spacing.x);
        PyObject* y = PyFloat_FromDouble(repetition->spacing.y);
        PyObject* result = PyTuple_New(2);
        if (!x || !y || !result) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create return tuple.");
            Py_XDECREF(x);
            Py_XDECREF(y);
            Py_XDECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, 0, x);
        PyTuple_SET_ITEM(result, 1, y);
        return result;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

// static int repetition_object_set_spacing(RepetitionObject* self, PyObject* arg, void*) {
//     Repetition* repetition = &self->repetition;
//     repetition->type = RepetitionType::Rectangular;
//     if (parse_point(arg, repetition->spacing, "spacing") < 0) return -1;
//     return 0;
// }

static PyObject* repetition_object_get_v1(RepetitionObject* self, void*) {
    Repetition* repetition = &self->repetition;
    if (repetition->type == RepetitionType::Regular) {
        PyObject* x = PyFloat_FromDouble(repetition->v1.x);
        PyObject* y = PyFloat_FromDouble(repetition->v1.y);
        PyObject* result = PyTuple_New(2);
        if (!x || !y || !result) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create return tuple.");
            Py_XDECREF(x);
            Py_XDECREF(y);
            Py_XDECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, 0, x);
        PyTuple_SET_ITEM(result, 1, y);
        return result;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

// static int repetition_object_set_v1(RepetitionObject* self, PyObject* arg, void*) {
//     Repetition* repetition = &self->repetition;
//     repetition->type = RepetitionType::Regular;
//     if (parse_point(arg, repetition->v1, "v1") < 0) return -1;
//     return 0;
// }

static PyObject* repetition_object_get_v2(RepetitionObject* self, void*) {
    Repetition* repetition = &self->repetition;
    if (repetition->type == RepetitionType::Regular) {
        PyObject* x = PyFloat_FromDouble(repetition->v2.x);
        PyObject* y = PyFloat_FromDouble(repetition->v2.y);
        PyObject* result = PyTuple_New(2);
        if (!x || !y || !result) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create return tuple.");
            Py_XDECREF(x);
            Py_XDECREF(y);
            Py_XDECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, 0, x);
        PyTuple_SET_ITEM(result, 1, y);
        return result;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

// static int repetition_object_set_v2(RepetitionObject* self, PyObject* arg, void*) {
//     Repetition* repetition = &self->repetition;
//     repetition->type = RepetitionType::Regular;
//     if (parse_point(arg, repetition->v2, "v2") < 0) return -1;
//     return 0;
// }

static PyObject* repetition_object_get_offsets(RepetitionObject* self, void*) {
    Repetition* repetition = &self->repetition;
    if (repetition->type == RepetitionType::Explicit) {
        npy_intp dims[] = {(npy_intp)repetition->offsets.size, 2};
        PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (!result) {
            PyErr_SetString(PyExc_MemoryError, "Unable to create return array.");
            return NULL;
        }
        double* data = (double*)PyArray_DATA((PyArrayObject*)result);
        memcpy(data, repetition->offsets.items, sizeof(double) * repetition->offsets.size * 2);
        return (PyObject*)result;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

// static int repetition_object_set_offsets(RepetitionObject* self, PyObject* arg, void*) {
//     Repetition* repetition = &self->repetition;
//     repetition->type = RepetitionType::Explicit;
//     if (parse_point_sequence(arg, repetition->offsets, "offsets") < 0) return -1;
//     return 0;
// }

static PyObject* repetition_object_get_x_offsets(RepetitionObject* self, void*) {
    Repetition* repetition = &self->repetition;
    if (repetition->type == RepetitionType::ExplicitX) {
        npy_intp dims[] = {(npy_intp)repetition->coords.size};
        PyObject* result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (!result) {
            PyErr_SetString(PyExc_MemoryError, "Unable to create return array.");
            return NULL;
        }
        double* data = (double*)PyArray_DATA((PyArrayObject*)result);
        memcpy(data, repetition->coords.items, sizeof(double) * repetition->coords.size);
        return (PyObject*)result;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

// static int repetition_object_set_x_offsets(RepetitionObject* self, PyObject* arg, void*) {
//     Repetition* repetition = &self->repetition;
//     repetition->type = RepetitionType::ExplicitX;
//     if (parse_double_sequence(arg, repetition->coords, "x_offsets") < 0) return -1;
//     return 0;
// }

static PyObject* repetition_object_get_y_offsets(RepetitionObject* self, void*) {
    Repetition* repetition = &self->repetition;
    if (repetition->type == RepetitionType::ExplicitY) {
        npy_intp dims[] = {(npy_intp)repetition->coords.size};
        PyObject* result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (!result) {
            PyErr_SetString(PyExc_MemoryError, "Unable to create return array.");
            return NULL;
        }
        double* data = (double*)PyArray_DATA((PyArrayObject*)result);
        memcpy(data, repetition->coords.items, sizeof(double) * repetition->coords.size);
        return (PyObject*)result;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

// static int repetition_object_set_y_offsets(RepetitionObject* self, PyObject* arg, void*) {
//     Repetition* repetition = &self->repetition;
//     repetition->type = RepetitionType::ExplicitY;
//     if (parse_double_sequence(arg, repetition->coords, "y_offsets") < 0) return -1;
//     return 0;
// }

static PyGetSetDef repetition_object_getset[] = {
    {"size", (getter)repetition_object_get_size, NULL, repetition_object_size_doc, NULL},
    {"columns", (getter)repetition_object_get_columns, NULL, repetition_object_columns_doc, NULL},
    {"rows", (getter)repetition_object_get_rows, NULL, repetition_object_rows_doc, NULL},
    {"spacing", (getter)repetition_object_get_spacing, NULL, repetition_object_spacing_doc, NULL},
    {"v1", (getter)repetition_object_get_v1, NULL, repetition_object_v1_doc, NULL},
    {"v2", (getter)repetition_object_get_v2, NULL, repetition_object_v2_doc, NULL},
    {"offsets", (getter)repetition_object_get_offsets, NULL, repetition_object_offsets_doc, NULL},
    {"x_offsets", (getter)repetition_object_get_x_offsets, NULL, repetition_object_x_offsets_doc,
     NULL},
    {"y_offsets", (getter)repetition_object_get_y_offsets, NULL, repetition_object_y_offsets_doc,
     NULL},
    {NULL}};
