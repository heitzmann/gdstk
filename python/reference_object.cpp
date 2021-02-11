/*
Copyright 2020 Lucas Heitzmann Gabrielli.
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
        free_allocation(reference);
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
    Vec2 origin = {0, 0};
    uint64_t columns = 1;
    uint64_t rows = 1;
    const char* keywords[] = {"cell",          "origin",       "rotation",
                              "magnification", "x_reflection", "columns",
                              "rows",          "spacing",      NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OddpKKO:Reference", (char**)keywords, &cell_obj,
                                     &origin_obj, &rotation, &magnification, &x_reflection,
                                     &columns, &rows, &spacing_obj))
        return -1;
    if (parse_point(origin_obj, origin, "origin") < 0) return -1;

    if (self->reference)
        self->reference->clear();
    else
        self->reference = (Reference*)allocate_clear(sizeof(Reference));
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
        reference->name = (char*)allocate(++len);
        memcpy(reference->name, name, len);
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument cell must be a Cell, RawCell, or string.");
        return -1;
    }

    if (spacing_obj != NULL && spacing_obj != Py_None && columns > 0 && rows > 0) {
        Repetition* repetition = &reference->repetition;
        Vec2 spacing;
        if (parse_point(spacing_obj, spacing, "spacing") < 0) return -1;
        // If any of these are zero, we won't be able to detect the AREF construction in to_gds().
        if (columns == 1 && spacing.x == 0) spacing.x = 1;
        if (rows == 1 && spacing.y == 0) spacing.y = 1;
        repetition->type = RepetitionType::Rectangular;
        repetition->columns = columns;
        repetition->rows = rows;
        repetition->spacing = spacing;
        if (rotation != 0 || x_reflection) {
            repetition->transform(1, x_reflection > 0, rotation);
        }
    }

    reference->origin = origin;
    reference->rotation = rotation;
    reference->magnification = magnification;
    reference->x_reflection = x_reflection > 0;
    reference->owner = self;
    return 0;
}

static PyObject* reference_object_copy(ReferenceObject* self, PyObject* args) {
    ReferenceObject* result = PyObject_New(ReferenceObject, &reference_object_type);
    result = (ReferenceObject*)PyObject_Init((PyObject*)result, &reference_object_type);
    result->reference = (Reference*)allocate_clear(sizeof(Reference));
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

static PyObject* reference_object_apply_repetition(ReferenceObject* self, PyObject* args) {
    Array<Reference*> array = {0};
    self->reference->apply_repetition(array);
    PyObject* result = PyList_New(array.count);
    for (uint64_t i = 0; i < array.count; i++) {
        ReferenceObject* obj = PyObject_New(ReferenceObject, &reference_object_type);
        obj = (ReferenceObject*)PyObject_Init((PyObject*)obj, &reference_object_type);
        obj->reference = array[i];
        array[i]->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }
    array.clear();
    return result;
}

static PyObject* reference_object_set_property(ReferenceObject* self, PyObject* args) {
    if (!parse_property(self->reference->properties, args)) return NULL;
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* reference_object_get_property(ReferenceObject* self, PyObject* args) {
    return build_property(self->reference->properties, args);
}

static PyObject* reference_object_delete_property(ReferenceObject* self, PyObject* args) {
    char* name;
    if (!PyArg_ParseTuple(args, "s:delete_property", &name)) return NULL;
    remove_property(self->reference->properties, name, false);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* reference_object_set_gds_property(ReferenceObject* self, PyObject* args) {
    uint16_t attribute;
    char* value;
    if (!PyArg_ParseTuple(args, "Hs:set_gds_property", &attribute, &value)) return NULL;
    set_gds_property(self->reference->properties, attribute, value);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* reference_object_get_gds_property(ReferenceObject* self, PyObject* args) {
    uint16_t attribute;
    if (!PyArg_ParseTuple(args, "H:get_gds_property", &attribute)) return NULL;
    const PropertyValue* value = get_gds_property(self->reference->properties, attribute);
    if (!value) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return PyUnicode_FromString((char*)value->bytes);
}

static PyObject* reference_object_delete_gds_property(ReferenceObject* self, PyObject* args) {
    uint16_t attribute;
    if (!PyArg_ParseTuple(args, "H:delete_gds_property", &attribute)) return NULL;
    remove_gds_property(self->reference->properties, attribute);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyMethodDef reference_object_methods[] = {
    {"copy", (PyCFunction)reference_object_copy, METH_NOARGS, reference_object_copy_doc},
    {"bounding_box", (PyCFunction)reference_object_bounding_box, METH_NOARGS,
     reference_object_bounding_box_doc},
    {"apply_repetition", (PyCFunction)reference_object_apply_repetition, METH_NOARGS,
     reference_object_apply_repetition_doc},
    {"set_property", (PyCFunction)reference_object_set_property, METH_VARARGS,
     object_set_property_doc},
    {"get_property", (PyCFunction)reference_object_get_property, METH_VARARGS,
     object_get_property_doc},
    {"delete_property", (PyCFunction)reference_object_delete_property, METH_VARARGS,
     object_delete_property_doc},
    {"set_gds_property", (PyCFunction)reference_object_set_gds_property, METH_VARARGS,
     object_set_gds_property_doc},
    {"get_gds_property", (PyCFunction)reference_object_get_gds_property, METH_VARARGS,
     object_get_gds_property_doc},
    {"delete_gds_property", (PyCFunction)reference_object_delete_gds_property, METH_VARARGS,
     object_delete_gds_property_doc},
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
    } else if (test > 0) {
        self->reference->x_reflection = test > 0;
    }
    return 0;
}

static PyObject* reference_object_get_properties(ReferenceObject* self, void*) {
    return build_properties(self->reference->properties);
}

int reference_object_set_properties(ReferenceObject* self, PyObject* arg, void*) {
    return parse_properties(self->reference->properties, arg);
}

static PyObject* reference_object_get_repetition(ReferenceObject* self, void*) {
    RepetitionObject* obj = PyObject_New(RepetitionObject, &repetition_object_type);
    obj = (RepetitionObject*)PyObject_Init((PyObject*)obj, &repetition_object_type);
    obj->repetition.copy_from(self->reference->repetition);
    return (PyObject*)obj;
}

int reference_object_set_repetition(ReferenceObject* self, PyObject* arg, void*) {
    if (arg == Py_None) {
        self->reference->repetition.clear();
        return 0;
    } else if (!RepetitionObject_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Value must be a Repetition object.");
        return -1;
    }
    RepetitionObject* repetition_obj = (RepetitionObject*)arg;
    self->reference->repetition.clear();
    self->reference->repetition.copy_from(repetition_obj->repetition);
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
    {"properties", (getter)reference_object_get_properties, (setter)reference_object_set_properties,
     object_properties_doc, NULL},
    {"repetition", (getter)reference_object_get_repetition, (setter)reference_object_set_repetition,
     object_repetition_doc, NULL},
    {NULL}};
