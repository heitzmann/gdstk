/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* reference_object_str(ReferenceObject* self) {
    char buffer[128];
    Reference* reference = self->reference;
    snprintf(buffer, COUNT(buffer), "Reference to %s'%s' at (%lg, %lg)",
             reference->type == ReferenceType::Cell
                 ? "Cell "
                 : (reference->type == ReferenceType::RawCell ? "RawCell " : ""),
             reference->type == ReferenceType::Cell
                 ? reference->cell->name
                 : (reference->type == ReferenceType::RawCell ? reference->rawcell->name
                                                              : reference->name),
             reference->origin.x, reference->origin.y);
    return PyUnicode_FromString(buffer);
}

static void reference_object_dealloc(ReferenceObject* self) {
    Reference* reference = self->reference;
    if (reference) {
        if (reference->type == ReferenceType::Cell)
            Py_DECREF(reference->cell->owner);
        else if (reference->type == ReferenceType::RawCell)
            Py_DECREF(reference->rawcell->owner);
        reference->clear();
        free(reference);
    }
    PyObject_Del(self);
}

static int reference_object_init(ReferenceObject* self, PyObject* args, PyObject* kwds) {
    PyObject* cell_obj = NULL;
    PyObject* origin_obj = NULL;
    PyObject* spacing_obj = NULL;
    double rotation = 0;
    double magnification = 1;
    int x_reflection = 0;
    uint16_t columns = 1;
    uint16_t rows = 1;
    Vec2 origin = {0, 0};
    Vec2 spacing = {0, 0};
    const char* keywords[] = {"cell",          "origin",       "rotation",
                              "magnification", "x_reflection", "columns",
                              "rows",          "spacing",      NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OddpHHO:Reference", (char**)keywords, &cell_obj,
                                     &origin_obj, &rotation, &magnification, &x_reflection,
                                     &columns, &rows, &spacing_obj))
        return -1;
    if (parse_point(origin_obj, origin, "origin") < 0) return -1;
    if (parse_point(spacing_obj, spacing, "spacing") < 0) return -1;

    if (self->reference)
        self->reference->clear();
    else
        self->reference = (Reference*)calloc(1, sizeof(Reference));
    Reference* reference = self->reference;

    if (CellObject_Check(cell_obj)) {
        reference->type = ReferenceType::Cell;
        reference->cell = ((CellObject*)cell_obj)->cell;
        Py_INCREF(cell_obj);
    } else if (RawCellObject_Check(cell_obj)) {
        reference->type = ReferenceType::RawCell;
        reference->rawcell = ((RawCellObject*)cell_obj)->rawcell;
        Py_INCREF(cell_obj);
    } else if (PyUnicode_Check(cell_obj)) {
        reference->type = ReferenceType::Name;
        Py_ssize_t len = 0;
        const char* name = PyUnicode_AsUTF8AndSize(cell_obj, &len);
        if (!name) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert cell argument to string.");
            return -1;
        }
        reference->name = (char*)malloc(sizeof(char) * (++len));
        memcpy(reference->name, name, len);
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument cell must be a Cell, RawCell, or string.");
        return -1;
    }

    reference->origin = origin;
    reference->rotation = rotation;
    reference->magnification = magnification;
    reference->x_reflection = x_reflection > 0;
    reference->columns = columns;
    reference->rows = rows;
    reference->spacing = spacing;
    properties_clear(reference->properties);
    reference->properties = NULL;
    reference->owner = self;
    return 0;
}

static PyObject* reference_object_copy(ReferenceObject* self, PyObject* args) {
    ReferenceObject* result = PyObject_New(ReferenceObject, &reference_object_type);
    result = (ReferenceObject*)PyObject_Init((PyObject*)result, &reference_object_type);
    result->reference = (Reference*)calloc(1, sizeof(Reference));
    result->reference->copy_from(*self->reference);
    result->reference->owner = result;
    return (PyObject*)result;
}

static PyObject* reference_object_bounding_box(ReferenceObject* self, PyObject* args) {
    Vec2 min, max;
    self->reference->bounding_box(min, max);
    if (min.x > max.x) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return Py_BuildValue("((dd)(dd))", min.x, min.y, max.x, max.y);
}

static PyObject* reference_object_set_property(ReferenceObject* self, PyObject* args) {
    int16_t attr;
    char* value;
    if (!PyArg_ParseTuple(args, "hs:set_property", &attr, &value)) return NULL;
    set_property(self->reference->properties, attr, value);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* reference_object_get_property(ReferenceObject* self, PyObject* args) {
    Property* property = self->reference->properties;

    if (PyTuple_Size(args) == 0 || PyTuple_GetItem(args, 0) == Py_None) {
        PyObject* result = PyDict_New();
        for (; property; property = property->next) {
            PyObject* key = PyLong_FromLong(property->key);
            if (!key) {
                PyErr_SetString(PyExc_TypeError, "Unable to convert key to int.");
                Py_DECREF(result);
                return NULL;
            }
            PyObject* val = PyUnicode_FromString(property->value);
            if (!val) {
                PyErr_SetString(PyExc_TypeError, "Unable to convert value to string.");
                Py_DECREF(key);
                Py_DECREF(result);
                return NULL;
            }
            PyDict_SetItem(result, key, val);
            Py_DECREF(key);
            Py_DECREF(val);
        }
        return result;
    }

    int16_t attr;
    if (!PyArg_ParseTuple(args, "h:get_property", &attr)) return NULL;
    const char* value = get_property(property, attr);
    if (!value) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return PyUnicode_FromString(value);
}

static PyObject* reference_object_delete_property(ReferenceObject* self, PyObject* args) {
    int16_t attr;
    if (!PyArg_ParseTuple(args, "h:delete_property", &attr)) return NULL;
    delete_property(self->reference->properties, attr);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyMethodDef reference_object_methods[] = {
    {"copy", (PyCFunction)reference_object_copy, METH_NOARGS, reference_object_copy_doc},
    {"bounding_box", (PyCFunction)reference_object_bounding_box, METH_NOARGS,
     reference_object_bounding_box_doc},
    {"set_property", (PyCFunction)reference_object_set_property, METH_VARARGS,
     reference_object_set_property_doc},
    {"get_property", (PyCFunction)reference_object_get_property, METH_VARARGS,
     reference_object_get_property_doc},
    {"delete_property", (PyCFunction)reference_object_delete_property, METH_VARARGS,
     reference_object_delete_property_doc},
    {NULL}};

static PyObject* reference_object_get_cell(ReferenceObject* self, void*) {
    PyObject* result = Py_None;
    switch (self->reference->type) {
        case ReferenceType::Cell:
            result = (PyObject*)self->reference->cell->owner;
            break;
        case ReferenceType::RawCell:
            result = (PyObject*)self->reference->rawcell->owner;
            break;
        case ReferenceType::Name:
            result = PyUnicode_FromString(self->reference->name);
            if (!result) {
                PyErr_SetString(PyExc_TypeError, "Unable to convert cell name to string.");
                return NULL;
            }
            break;
    }
    Py_INCREF(result);
    return result;
}

static PyObject* reference_object_get_origin(ReferenceObject* self, void*) {
    return Py_BuildValue("(dd)", self->reference->origin.x, self->reference->origin.y);
}

int reference_object_set_origin(ReferenceObject* self, PyObject* arg, void*) {
    if (parse_point(arg, self->reference->origin, "origin") != 0) return -1;
    return 0;
}

static PyObject* reference_object_get_spacing(ReferenceObject* self, void*) {
    return Py_BuildValue("(dd)", self->reference->spacing.x, self->reference->spacing.y);
}

int reference_object_set_spacing(ReferenceObject* self, PyObject* arg, void*) {
    return parse_point(arg, self->reference->spacing, "spacing");
}

static PyObject* reference_object_get_rotation(ReferenceObject* self, void*) {
    PyObject* result = PyFloat_FromDouble(self->reference->rotation);
    if (!result) PyErr_SetString(PyExc_RuntimeError, "Unable to create float.");
    return result;
}

int reference_object_set_rotation(ReferenceObject* self, PyObject* arg, void*) {
    self->reference->rotation = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to convert value to float.");
        return -1;
    }
    return 0;
}

static PyObject* reference_object_get_magnification(ReferenceObject* self, void*) {
    PyObject* result = PyFloat_FromDouble(self->reference->magnification);
    if (!result) PyErr_SetString(PyExc_RuntimeError, "Unable to create float.");
    return result;
}

int reference_object_set_magnification(ReferenceObject* self, PyObject* arg, void*) {
    self->reference->magnification = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to convert value to float.");
        return -1;
    }
    return 0;
}

static PyObject* reference_object_get_x_reflection(ReferenceObject* self, void*) {
    if (self->reference->x_reflection) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

int reference_object_set_x_reflection(ReferenceObject* self, PyObject* arg, void*) {
    int test = PyObject_IsTrue(arg);
    if (test < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to determine truth value.");
        return -1;
    } else if (test > 0)
        self->reference->x_reflection = test > 0;
    return 0;
}

static PyObject* reference_object_get_columns(ReferenceObject* self, void*) {
    PyObject* result = PyLong_FromUnsignedLong(self->reference->columns);
    if (!result) PyErr_SetString(PyExc_RuntimeError, "Unable to create long.");
    return result;
}

int reference_object_set_columns(ReferenceObject* self, PyObject* arg, void*) {
    self->reference->columns = PyLong_AsUnsignedLong(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to convert value to unsigned long.");
        return -1;
    }
    return 0;
}

static PyObject* reference_object_get_rows(ReferenceObject* self, void*) {
    PyObject* result = PyLong_FromUnsignedLong(self->reference->rows);
    if (!result) PyErr_SetString(PyExc_RuntimeError, "Unable to create long.");
    return result;
}

int reference_object_set_rows(ReferenceObject* self, PyObject* arg, void*) {
    self->reference->rows = PyLong_AsUnsignedLong(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to convert value to unsigned long.");
        return -1;
    }
    return 0;
}

static PyGetSetDef reference_object_getset[] = {
    {"cell", (getter)reference_object_get_cell, NULL, reference_object_cell_doc, NULL},
    {"origin", (getter)reference_object_get_origin, (setter)reference_object_set_origin,
     reference_object_origin_doc, NULL},
    {"rotation", (getter)reference_object_get_rotation, (setter)reference_object_set_rotation,
     reference_object_rotation_doc, NULL},
    {"magnification", (getter)reference_object_get_magnification,
     (setter)reference_object_set_magnification, reference_object_magnification_doc, NULL},
    {"x_reflection", (getter)reference_object_get_x_reflection,
     (setter)reference_object_set_x_reflection, reference_object_x_reflection_doc, NULL},
    {"columns", (getter)reference_object_get_columns, (setter)reference_object_set_columns,
     reference_object_columns_doc, NULL},
    {"rows", (getter)reference_object_get_rows, (setter)reference_object_set_rows,
     reference_object_rows_doc, NULL},
    {"spacing", (getter)reference_object_get_spacing, (setter)reference_object_set_spacing,
     reference_object_spacing_doc, NULL},
    {NULL}};
