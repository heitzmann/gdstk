/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* reference_object_str(ReferenceObject* self) {
    char buffer[GDSTK_PRINT_BUFFER_COUNT];
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
        if (reference->type == ReferenceType::Cell) {
            Py_XDECREF(reference->cell->owner);
        } else if (reference->type == ReferenceType::RawCell) {
            Py_XDECREF(reference->rawcell->owner);
        }
        reference->clear();
        free_allocation(reference);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int reference_object_init(ReferenceObject* self, PyObject* args, PyObject* kwds) {
    PyObject* cell_obj = NULL;
    PyObject* origin_obj = NULL;
    PyObject* spacing_obj = Py_None;
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

    if (self->reference) {
        self->reference->clear();
    } else {
        self->reference = (Reference*)allocate_clear(sizeof(Reference));
    }
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
        free_allocation(reference);
        self->reference = NULL;
        PyErr_SetString(PyExc_TypeError, "Argument cell must be a Cell, RawCell, or string.");
        return -1;
    }

    if (spacing_obj != Py_None && columns > 0 && rows > 0) {
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

static PyObject* reference_object_copy(ReferenceObject* self, PyObject*) {
    ReferenceObject* result = PyObject_New(ReferenceObject, &reference_object_type);
    result = (ReferenceObject*)PyObject_Init((PyObject*)result, &reference_object_type);
    result->reference = (Reference*)allocate_clear(sizeof(Reference));
    result->reference->copy_from(*self->reference);

    if (result->reference->type == ReferenceType::Cell)
        Py_INCREF(result->reference->cell->owner);
    else if (result->reference->type == ReferenceType::RawCell)
        Py_INCREF(result->reference->rawcell->owner);

    result->reference->owner = result;
    return (PyObject*)result;
}

static PyObject* reference_object_bounding_box(ReferenceObject* self, PyObject*) {
    Vec2 min, max;
    self->reference->bounding_box(min, max);
    if (min.x > max.x) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return Py_BuildValue("((dd)(dd))", min.x, min.y, max.x, max.y);
}

static PyObject* reference_object_convex_hull(ReferenceObject* self, PyObject*) {
    Array<Vec2> points = {};
    self->reference->convex_hull(points);
    npy_intp dims[] = {(npy_intp)points.count, 2};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_MemoryError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    memcpy(data, points.items, sizeof(double) * points.count * 2);
    points.clear();
    return (PyObject*)result;
}

static PyObject* reference_object_get_polygons(ReferenceObject* self, PyObject* args,
                                               PyObject* kwds) {
    int apply_repetitions = 1;
    int include_paths = 1;
    PyObject* py_depth = Py_None;
    PyObject* py_layer = Py_None;
    PyObject* py_datatype = Py_None;
    const char* keywords[] = {
        "apply_repetitions", "include_paths", "depth", "layer", "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ppOOO:get_polygons", (char**)keywords,
                                     &apply_repetitions, &include_paths, &py_depth, &py_layer,
                                     &py_datatype))
        return NULL;

    int64_t depth = -1;
    if (py_depth != Py_None) {
        depth = PyLong_AsLongLong(py_depth);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert depth to integer.");
            return NULL;
        }
    }

    if ((py_layer == Py_None) != (py_datatype == Py_None)) {
        PyErr_SetString(PyExc_ValueError,
                        "Filtering is only enabled if both layer and datatype are set.");
        return NULL;
    }

    uint32_t layer = 0;
    uint32_t datatype = 0;
    bool filter = (py_layer != Py_None) && (py_datatype != Py_None);
    if (filter) {
        layer = PyLong_AsUnsignedLong(py_layer);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert layer to unsigned integer.");
            return NULL;
        }
        datatype = PyLong_AsUnsignedLong(py_datatype);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert datatype to unsigned integer.");
            return NULL;
        }
    }

    Array<Polygon*> array = {};
    self->reference->get_polygons(apply_repetitions > 0, include_paths > 0, depth, filter,
                                  make_tag(layer, datatype), array);

    PyObject* result = PyList_New(array.count);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        for (uint64_t i = 0; i < array.count; i++) {
            array[i]->clear();
            free_allocation(array[i]);
        }
        array.clear();
        return NULL;
    }

    for (uint64_t i = 0; i < array.count; i++) {
        Polygon* poly = array[i];
        PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
        obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
        obj->polygon = poly;
        poly->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }

    array.clear();
    return result;
}

static PyObject* reference_object_get_paths(ReferenceObject* self, PyObject* args, PyObject* kwds) {
    int apply_repetitions = 1;
    PyObject* py_depth = Py_None;
    PyObject* py_layer = Py_None;
    PyObject* py_datatype = Py_None;
    const char* keywords[] = {"apply_repetitions", "depth", "layer", "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|pOOO:get_polygons", (char**)keywords,
                                     &apply_repetitions, &py_depth, &py_layer, &py_datatype))
        return NULL;

    int64_t depth = -1;
    if (py_depth != Py_None) {
        depth = PyLong_AsLongLong(py_depth);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert depth to integer.");
            return NULL;
        }
    }

    uint32_t layer = 0;
    uint32_t datatype = 0;
    bool filter = (py_layer != Py_None) && (py_datatype != Py_None);
    if (filter) {
        layer = PyLong_AsUnsignedLong(py_layer);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert layer to unsigned integer.");
            return NULL;
        }
        datatype = PyLong_AsUnsignedLong(py_datatype);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert datatype to unsigned integer.");
            return NULL;
        }
    }

    Array<FlexPath*> fp_array = {};
    self->reference->get_flexpaths(apply_repetitions > 0, depth, filter, make_tag(layer, datatype),
                                   fp_array);

    Array<RobustPath*> rp_array = {};
    self->reference->get_robustpaths(apply_repetitions > 0, depth, filter,
                                     make_tag(layer, datatype), rp_array);

    PyObject* result = PyList_New(fp_array.count + rp_array.count);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        for (uint64_t i = 0; i < fp_array.count; i++) {
            fp_array[i]->clear();
            free_allocation(fp_array[i]);
        }
        fp_array.clear();
        for (uint64_t i = 0; i < rp_array.count; i++) {
            rp_array[i]->clear();
            free_allocation(rp_array[i]);
        }
        rp_array.clear();
        return NULL;
    }

    for (uint64_t i = 0; i < fp_array.count; i++) {
        FlexPath* path = fp_array[i];
        FlexPathObject* obj = PyObject_New(FlexPathObject, &flexpath_object_type);
        obj = (FlexPathObject*)PyObject_Init((PyObject*)obj, &flexpath_object_type);
        obj->flexpath = path;
        path->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }
    for (uint64_t i = 0; i < rp_array.count; i++) {
        RobustPath* path = rp_array[i];
        RobustPathObject* obj = PyObject_New(RobustPathObject, &robustpath_object_type);
        obj = (RobustPathObject*)PyObject_Init((PyObject*)obj, &robustpath_object_type);
        obj->robustpath = path;
        path->owner = obj;
        PyList_SET_ITEM(result, i + fp_array.count, (PyObject*)obj);
    }

    fp_array.clear();
    rp_array.clear();
    return result;
}

static PyObject* reference_object_get_labels(ReferenceObject* self, PyObject* args,
                                             PyObject* kwds) {
    int apply_repetitions = 1;
    PyObject* py_depth = Py_None;
    PyObject* py_layer = Py_None;
    PyObject* py_texttype = Py_None;
    const char* keywords[] = {"apply_repetitions", "depth", "layer", "texttype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|pOOO:get_polygons", (char**)keywords,
                                     &apply_repetitions, &py_depth, &py_layer, &py_texttype))
        return NULL;

    int64_t depth = -1;
    if (py_depth != Py_None) {
        depth = PyLong_AsLongLong(py_depth);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert depth to integer.");
            return NULL;
        }
    }

    uint32_t layer = 0;
    uint32_t texttype = 0;
    bool filter = (py_layer != Py_None) && (py_texttype != Py_None);
    if (filter) {
        layer = PyLong_AsUnsignedLong(py_layer);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert layer to unsigned integer.");
            return NULL;
        }
        texttype = PyLong_AsUnsignedLong(py_texttype);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert texttype to unsigned integer.");
            return NULL;
        }
    }

    Array<Label*> array = {};
    self->reference->get_labels(apply_repetitions > 0, depth, filter, make_tag(layer, texttype),
                                array);

    PyObject* result = PyList_New(array.count);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        for (uint64_t i = 0; i < array.count; i++) {
            array[i]->clear();
            free_allocation(array[i]);
        }
        array.clear();
        return NULL;
    }

    for (uint64_t i = 0; i < array.count; i++) {
        Label* label = array[i];
        LabelObject* obj = PyObject_New(LabelObject, &label_object_type);
        obj = (LabelObject*)PyObject_Init((PyObject*)obj, &label_object_type);
        obj->label = label;
        label->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }

    array.clear();
    return result;
}

static PyObject* reference_object_apply_repetition(ReferenceObject* self, PyObject*) {
    Array<Reference*> array = {};
    self->reference->apply_repetition(array);
    PyObject* result = PyList_New(array.count);
    for (uint64_t i = 0; i < array.count; i++) {
        ReferenceObject* obj = PyObject_New(ReferenceObject, &reference_object_type);
        obj = (ReferenceObject*)PyObject_Init((PyObject*)obj, &reference_object_type);
        obj->reference = array[i];
        array[i]->owner = obj;
        if (array[i]->type == ReferenceType::Cell) {
            Py_INCREF(array[i]->cell->owner);
        } else if (array[i]->type == ReferenceType::RawCell) {
            Py_INCREF(array[i]->rawcell->owner);
        }
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
    Py_ssize_t count;
    if (!PyArg_ParseTuple(args, "Hs#:set_gds_property", &attribute, &value, &count)) return NULL;
    if (count >= 0)
        set_gds_property(self->reference->properties, attribute, value, (uint64_t)count);
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
    PyObject* result = PyUnicode_FromStringAndSize((char*)value->bytes, (Py_ssize_t)value->count);
    if (PyErr_Occurred()) {
        Py_XDECREF(result);
        PyErr_Clear();
        result = PyBytes_FromStringAndSize((char*)value->bytes, (Py_ssize_t)value->count);
    }
    return result;
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
    {"convex_hull", (PyCFunction)reference_object_convex_hull, METH_NOARGS,
     reference_object_convex_hull_doc},
    {"get_polygons", (PyCFunction)reference_object_get_polygons, METH_VARARGS | METH_KEYWORDS,
     reference_object_get_polygons_doc},
    {"get_paths", (PyCFunction)reference_object_get_paths, METH_VARARGS | METH_KEYWORDS,
     reference_object_get_paths_doc},
    {"get_labels", (PyCFunction)reference_object_get_labels, METH_VARARGS | METH_KEYWORDS,
     reference_object_get_labels_doc},
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

int reference_object_set_cell(ReferenceObject* self, PyObject* arg, void*) {
    Reference* reference = self->reference;
    ReferenceType new_type;
    char* new_name = NULL;

    if (CellObject_Check(arg)) {
        new_type = ReferenceType::Cell;
    } else if (RawCellObject_Check(arg)) {
        new_type = ReferenceType::RawCell;
    } else if (PyUnicode_Check(arg)) {
        new_type = ReferenceType::Name;
        Py_ssize_t len = 0;
        const char* name = PyUnicode_AsUTF8AndSize(arg, &len);
        if (!name) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert cell argument to string.");
            return -1;
        }
        new_name = (char*)allocate(++len);
        memcpy(new_name, name, len);
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument cell must be a Cell, RawCell, or string.");
        return -1;
    }

    switch (reference->type) {
        case ReferenceType::Cell:
            Py_DECREF(reference->cell->owner);
            break;
        case ReferenceType::RawCell:
            Py_DECREF(reference->rawcell->owner);
            break;
        case ReferenceType::Name:
            free_allocation(reference->name);
    }

    reference->type = new_type;
    switch (new_type) {
        case ReferenceType::Cell:
            reference->cell = ((CellObject*)arg)->cell;
            Py_INCREF(arg);
            break;
        case ReferenceType::RawCell:
            reference->rawcell = ((RawCellObject*)arg)->rawcell;
            Py_INCREF(arg);
            break;
        case ReferenceType::Name:
            reference->name = new_name;
    }
    return 0;
}

static PyObject* reference_object_get_cell_name(ReferenceObject* self, void*) {
    char const* name = NULL;
    switch (self->reference->type) {
        case ReferenceType::Cell:
            name = self->reference->cell->name;
            break;
        case ReferenceType::RawCell:
            name = self->reference->rawcell->name;
            break;
        case ReferenceType::Name:
            name = self->reference->name;
            break;
    }
    PyObject* result = PyUnicode_FromString(name);
    if (!result) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert cell name to string.");
        return NULL;
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
    } else
        self->reference->x_reflection = test > 0;
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
    {"cell", (getter)reference_object_get_cell, (setter)reference_object_set_cell,
     reference_object_cell_doc, NULL},
    {"cell_name", (getter)reference_object_get_cell_name, NULL, reference_object_cell_name_doc,
     NULL},
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
