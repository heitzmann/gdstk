/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* polygon_object_str(PolygonObject* self) {
    char buffer[128];
    snprintf(buffer, COUNT(buffer),
             "Polygon at layer %" PRIu32 ", datatype %" PRIu32 ", with %" PRIu64 " points",
             self->polygon->layer, self->polygon->datatype, self->polygon->point_array.count);
    return PyUnicode_FromString(buffer);
}

static void polygon_object_dealloc(PolygonObject* self) {
    if (self->polygon) {
        self->polygon->clear();
        free_allocation(self->polygon);
    }
    PyObject_Del(self);
}

static int polygon_object_init(PolygonObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_points = NULL;
    unsigned long layer = 0;
    unsigned long datatype = 0;
    const char* keywords[] = {"points", "layer", "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|kk:Polygon", (char**)keywords, &py_points,
                                     &layer, &datatype))
        return -1;

    if (self->polygon)
        self->polygon->clear();
    else
        self->polygon = (Polygon*)allocate_clear(sizeof(Polygon));
    Polygon* polygon = self->polygon;
    polygon->layer = layer;
    polygon->datatype = datatype;
    if (parse_point_sequence(py_points, polygon->point_array, "points") < 0) return -1;
    polygon->owner = self;
    return 0;
}

static PyObject* polygon_object_copy(PolygonObject* self, PyObject* args) {
    PolygonObject* result = PyObject_New(PolygonObject, &polygon_object_type);
    result = (PolygonObject*)PyObject_Init((PyObject*)result, &polygon_object_type);
    result->polygon = (Polygon*)allocate_clear(sizeof(Polygon));
    result->polygon->copy_from(*self->polygon);
    result->polygon->owner = result;
    return (PyObject*)result;
}

static PyObject* polygon_object_area(PolygonObject* self, PyObject* args) {
    const double area = self->polygon->area();
    return PyFloat_FromDouble(area);
}

static PyObject* polygon_object_bounding_box(PolygonObject* self, PyObject* args) {
    Vec2 min, max;
    self->polygon->bounding_box(min, max);
    if (min.x > max.x) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return Py_BuildValue("((dd)(dd))", min.x, min.y, max.x, max.y);
}

static PyObject* polygon_object_translate(PolygonObject* self, PyObject* args) {
    Vec2 v = {0, 0};
    PyObject* dx;
    PyObject* dy = NULL;
    if (!PyArg_ParseTuple(args, "O|O:translate", &dx, &dy)) return NULL;
    if (parse_point(dx, v, "") < 0) {
        PyErr_Clear();
        v.x = PyFloat_AsDouble(dx);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert first argument to float.");
            return NULL;
        }
        v.y = PyFloat_AsDouble(dy);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert second argument to float.");
            return NULL;
        }
    }
    self->polygon->translate(v);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* polygon_object_scale(PolygonObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"sx", "sy", "center", NULL};
    Vec2 scale = {0, 0};
    Vec2 center = {0, 0};
    PyObject* center_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|dO:scale", (char**)keywords, &scale.x, &scale.y,
                                     &center_obj))
        return NULL;
    if (scale.y == 0) scale.y = scale.x;
    if (parse_point(center_obj, center, "center") < 0) return NULL;
    self->polygon->scale(scale, center);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* polygon_object_mirror(PolygonObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"p1", "p2", NULL};
    Vec2 p1;
    Vec2 p2 = {0, 0};
    PyObject* p1_obj = NULL;
    PyObject* p2_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O:mirror", (char**)keywords, &p1_obj, &p2_obj))
        return NULL;
    if (parse_point(p1_obj, p1, "p1") < 0) return NULL;
    if (parse_point(p2_obj, p2, "p2") < 0) return NULL;
    self->polygon->mirror(p1, p2);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* polygon_object_rotate(PolygonObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"angle", "center", NULL};
    double angle;
    Vec2 center = {0, 0};
    PyObject* center_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|O:rotate", (char**)keywords, &angle,
                                     &center_obj))
        return NULL;
    if (parse_point(center_obj, center, "center") < 0) return NULL;
    self->polygon->rotate(angle, center);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* polygon_object_fillet(PolygonObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"radius", "tolerance", NULL};
    bool free_items = false;
    double radius = 0;
    double tol = 0.01;
    PyObject* radius_obj = NULL;
    Array<double> radius_array = {0};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|d:fillet", (char**)keywords, &radius_obj, &tol))
        return NULL;
    if (PySequence_Check(radius_obj)) {
        if (parse_double_sequence(radius_obj, radius_array, "radius") < 0) return NULL;
        free_items = true;
    } else {
        radius = PyFloat_AsDouble(radius_obj);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Unable to convert radius to float.");
            return NULL;
        }
        radius_array.count = 1;
        radius_array.items = &radius;
    }
    self->polygon->fillet(radius_array, tol);
    if (free_items) free_allocation(radius_array.items);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* polygon_object_fracture(PolygonObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"max_points", "precision", NULL};
    uint64_t max_points = 199;
    double precision = 0.001;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Kd:fracture", (char**)keywords, &max_points,
                                     &precision))
        return NULL;

    Array<Polygon*> array = {0};
    self->polygon->fracture(max_points, precision, array);
    PyObject* result = PyList_New(array.count);
    for (uint64_t i = 0; i < array.count; i++) {
        PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
        obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
        obj->polygon = array[i];
        array[i]->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }
    array.clear();
    return result;
}

static PyObject* polygon_object_apply_repetition(PolygonObject* self, PyObject* args) {
    Array<Polygon*> array = {0};
    self->polygon->apply_repetition(array);
    PyObject* result = PyList_New(array.count);
    for (uint64_t i = 0; i < array.count; i++) {
        PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
        obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
        obj->polygon = array[i];
        array[i]->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }
    array.clear();
    return result;
}

static PyObject* polygon_object_set_property(PolygonObject* self, PyObject* args) {
    if (!parse_property(self->polygon->properties, args)) return NULL;
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* polygon_object_get_property(PolygonObject* self, PyObject* args) {
    return build_property(self->polygon->properties, args);
}

static PyObject* polygon_object_delete_property(PolygonObject* self, PyObject* args) {
    char* name;
    if (!PyArg_ParseTuple(args, "s:delete_property", &name)) return NULL;
    remove_property(self->polygon->properties, name, false);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* polygon_object_set_gds_property(PolygonObject* self, PyObject* args) {
    uint16_t attribute;
    char* value;
    if (!PyArg_ParseTuple(args, "Hs:set_gds_property", &attribute, &value)) return NULL;
    set_gds_property(self->polygon->properties, attribute, value);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* polygon_object_get_gds_property(PolygonObject* self, PyObject* args) {
    uint16_t attribute;
    if (!PyArg_ParseTuple(args, "H:get_gds_property", &attribute)) return NULL;
    const PropertyValue* value = get_gds_property(self->polygon->properties, attribute);
    if (!value) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return PyUnicode_FromString((char*)value->bytes);
}

static PyObject* polygon_object_delete_gds_property(PolygonObject* self, PyObject* args) {
    uint16_t attribute;
    if (!PyArg_ParseTuple(args, "H:delete_gds_property", &attribute)) return NULL;
    remove_gds_property(self->polygon->properties, attribute);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyMethodDef polygon_object_methods[] = {
    {"copy", (PyCFunction)polygon_object_copy, METH_NOARGS, polygon_object_copy_doc},
    {"area", (PyCFunction)polygon_object_area, METH_NOARGS, polygon_object_area_doc},
    {"bounding_box", (PyCFunction)polygon_object_bounding_box, METH_NOARGS,
     polygon_object_bounding_box_doc},
    {"translate", (PyCFunction)polygon_object_translate, METH_VARARGS,
     polygon_object_translate_doc},
    {"scale", (PyCFunction)polygon_object_scale, METH_VARARGS | METH_KEYWORDS,
     polygon_object_scale_doc},
    {"mirror", (PyCFunction)polygon_object_mirror, METH_VARARGS | METH_KEYWORDS,
     polygon_object_mirror_doc},
    {"rotate", (PyCFunction)polygon_object_rotate, METH_VARARGS | METH_KEYWORDS,
     polygon_object_rotate_doc},
    {"fillet", (PyCFunction)polygon_object_fillet, METH_VARARGS | METH_KEYWORDS,
     polygon_object_fillet_doc},
    {"fracture", (PyCFunction)polygon_object_fracture, METH_VARARGS | METH_KEYWORDS,
     polygon_object_fracture_doc},
    {"apply_repetition", (PyCFunction)polygon_object_apply_repetition, METH_NOARGS,
     polygon_object_apply_repetition_doc},
    {"set_property", (PyCFunction)polygon_object_set_property, METH_VARARGS,
     object_set_property_doc},
    {"get_property", (PyCFunction)polygon_object_get_property, METH_VARARGS,
     object_get_property_doc},
    {"delete_property", (PyCFunction)polygon_object_delete_property, METH_VARARGS,
     object_delete_property_doc},
    {"set_gds_property", (PyCFunction)polygon_object_set_gds_property, METH_VARARGS,
     object_set_gds_property_doc},
    {"get_gds_property", (PyCFunction)polygon_object_get_gds_property, METH_VARARGS,
     object_get_gds_property_doc},
    {"delete_gds_property", (PyCFunction)polygon_object_delete_gds_property, METH_VARARGS,
     object_delete_gds_property_doc},
    {NULL}};

static PyObject* polygon_object_get_points(PolygonObject* self, void*) {
    const Array<Vec2>* point_array = &self->polygon->point_array;
    npy_intp dims[] = {(npy_intp)point_array->count, 2};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_MemoryError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    memcpy(data, point_array->items, sizeof(double) * point_array->count * 2);
    return (PyObject*)result;
}

static PyObject* polygon_object_get_layer(PolygonObject* self, void*) {
    return PyLong_FromUnsignedLongLong(self->polygon->layer);
}

static int polygon_object_set_layer(PolygonObject* self, PyObject* arg, void*) {
    self->polygon->layer = (uint32_t)PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert layer to int.");
        return -1;
    }
    return 0;
}

static PyObject* polygon_object_get_datatype(PolygonObject* self, void*) {
    return PyLong_FromUnsignedLongLong(self->polygon->datatype);
}

static int polygon_object_set_datatype(PolygonObject* self, PyObject* arg, void*) {
    self->polygon->datatype = (uint32_t)PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert datatype to int.");
        return -1;
    }
    return 0;
}

static PyObject* polygon_object_get_size(PolygonObject* self, void*) {
    return PyLong_FromUnsignedLongLong(self->polygon->point_array.count);
}

static PyObject* polygon_object_get_properties(PolygonObject* self, void*) {
    return build_properties(self->polygon->properties);
}

int polygon_object_set_properties(PolygonObject* self, PyObject* arg, void*) {
    return parse_properties(self->polygon->properties, arg);
}

static PyObject* polygon_object_get_repetition(PolygonObject* self, void*) {
    RepetitionObject* obj = PyObject_New(RepetitionObject, &repetition_object_type);
    obj = (RepetitionObject*)PyObject_Init((PyObject*)obj, &repetition_object_type);
    obj->repetition.copy_from(self->polygon->repetition);
    return (PyObject*)obj;
}

int polygon_object_set_repetition(PolygonObject* self, PyObject* arg, void*) {
    if (arg == Py_None) {
        self->polygon->repetition.clear();
        return 0;
    } else if (!RepetitionObject_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Value must be a Repetition object.");
        return -1;
    }
    RepetitionObject* repetition_obj = (RepetitionObject*)arg;
    self->polygon->repetition.clear();
    self->polygon->repetition.copy_from(repetition_obj->repetition);
    return 0;
}

static PyGetSetDef polygon_object_getset[] = {
    {"points", (getter)polygon_object_get_points, NULL, polygon_object_points_doc},
    {"layer", (getter)polygon_object_get_layer, (setter)polygon_object_set_layer,
     polygon_object_layer_doc, NULL},
    {"datatype", (getter)polygon_object_get_datatype, (setter)polygon_object_set_datatype,
     polygon_object_datatype_doc, NULL},
    {"size", (getter)polygon_object_get_size, NULL, polygon_object_size_doc, NULL},
    {"properties", (getter)polygon_object_get_properties, (setter)polygon_object_set_properties,
     object_properties_doc, NULL},
    {"repetition", (getter)polygon_object_get_repetition, (setter)polygon_object_set_repetition,
     object_repetition_doc, NULL},
    {NULL}};
