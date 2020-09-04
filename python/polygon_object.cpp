/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* polygon_object_str(PolygonObject* self) {
    char buffer[128];
    snprintf(buffer, COUNT(buffer), "Polygon at layer %hd, datatype %hd, with %" PRId64 " points",
             self->polygon->layer, self->polygon->datatype, self->polygon->point_array.size);
    return PyUnicode_FromString(buffer);
}

static void polygon_object_dealloc(PolygonObject* self) {
    if (self->polygon) {
        self->polygon->clear();
        free(self->polygon);
    }
    PyObject_Del(self);
}

static int polygon_object_init(PolygonObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_points = NULL;
    short layer = 0;
    short datatype = 0;
    const char* keywords[] = {"points", "layer", "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|hh:Polygon", (char**)keywords, &py_points,
                                     &layer, &datatype))
        return -1;

    if (self->polygon)
        self->polygon->clear();
    else
        self->polygon = (Polygon*)calloc(1, sizeof(Polygon));
    Polygon* polygon = self->polygon;
    polygon->layer = layer;
    polygon->datatype = datatype;
    if (parse_point_sequence(py_points, polygon->point_array, "points") < 0) return -1;
    properties_clear(polygon->properties);
    polygon->properties = NULL;
    polygon->owner = self;
    return 0;
}

static PyObject* polygon_object_copy(PolygonObject* self, PyObject* args) {
    PolygonObject* result = PyObject_New(PolygonObject, &polygon_object_type);
    result = (PolygonObject*)PyObject_Init((PyObject*)result, &polygon_object_type);
    result->polygon = (Polygon*)calloc(1, sizeof(Polygon));
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
    Vec2 v;
    if (!PyArg_ParseTuple(args, "dd:translate", &v.x, &v.y)) return NULL;
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
    double* radii;
    double tol = 0.01;
    PyObject* radius = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|d:fillet", (char**)keywords, &radius, &tol))
        return NULL;
    if (PySequence_Check(radius)) {
        int64_t num = 0;
        radii = parse_sequence_double(radius, num, "radius");
        if (num < self->polygon->point_array.size) {
            PyErr_Format(PyExc_TypeError, "Not enough items in sequence (expecting %" PRId64 ").",
                         self->polygon->point_array.size);
            if (radii) free(radii);
            return NULL;
        }
    } else {
        double r = PyFloat_AsDouble(radius);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Unable to convert radius to float.");
            return NULL;
        }
        int64_t num = self->polygon->point_array.size;
        radii = (double*)malloc(sizeof(double) * num);
        double* p = radii;
        for (int64_t j = 0; j < num; j++) *p++ = r;
    }
    self->polygon->fillet(radii, tol);
    free(radii);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* polygon_object_fracture(PolygonObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"max_points", "precision", NULL};
    int64_t max_points = 199;
    double precision = 0.001;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ld:fracture", (char**)keywords, &max_points,
                                     &precision))
        return NULL;
    Array<Polygon*> array = self->polygon->fracture(max_points, precision);

    PyObject* result = PyList_New(array.size);
    for (Py_ssize_t i = 0; i < array.size; i++) {
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
    int16_t attr;
    char* value;
    if (!PyArg_ParseTuple(args, "hs:set_property", &attr, &value)) return NULL;
    set_property(self->polygon->properties, attr, value);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* polygon_object_get_property(PolygonObject* self, PyObject* args) {
    Property* property = self->polygon->properties;

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

static PyObject* polygon_object_delete_property(PolygonObject* self, PyObject* args) {
    int16_t attr;
    if (!PyArg_ParseTuple(args, "h:delete_property", &attr)) return NULL;
    delete_property(self->polygon->properties, attr);
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
    {"set_property", (PyCFunction)polygon_object_set_property, METH_VARARGS,
     polygon_object_set_property_doc},
    {"get_property", (PyCFunction)polygon_object_get_property, METH_VARARGS,
     polygon_object_get_property_doc},
    {"delete_property", (PyCFunction)polygon_object_delete_property, METH_VARARGS,
     polygon_object_delete_property_doc},
    {NULL}};

static PyObject* polygon_object_get_points(PolygonObject* self, void*) {
    const Array<Vec2>* point_array = &self->polygon->point_array;
    npy_intp dims[] = {(npy_intp)point_array->size, 2};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_MemoryError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    memcpy(data, point_array->items, sizeof(double) * point_array->size * 2);
    return (PyObject*)result;
}

static PyObject* polygon_object_get_layer(PolygonObject* self, void*) {
    return PyLong_FromLong(self->polygon->layer);
}

static int polygon_object_set_layer(PolygonObject* self, PyObject* arg, void*) {
    self->polygon->layer = PyLong_AsLong(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert layer to int.");
        return -1;
    }
    return 0;
}

static PyObject* polygon_object_get_datatype(PolygonObject* self, void*) {
    return PyLong_FromLong(self->polygon->datatype);
}

static int polygon_object_set_datatype(PolygonObject* self, PyObject* arg, void*) {
    self->polygon->datatype = PyLong_AsLong(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert datatype to int.");
        return -1;
    }
    return 0;
}

static PyObject* polygon_object_get_size(PolygonObject* self, void*) {
    return PyLong_FromLong(self->polygon->point_array.size);
}

static PyGetSetDef polygon_object_getset[] = {
    {"points", (getter)polygon_object_get_points, NULL, polygon_object_points_doc},
    {"layer", (getter)polygon_object_get_layer, (setter)polygon_object_set_layer,
     polygon_object_layer_doc, NULL},
    {"datatype", (getter)polygon_object_get_datatype, (setter)polygon_object_set_datatype,
     polygon_object_datatype_doc, NULL},
    {"size", (getter)polygon_object_get_size, NULL, polygon_object_size_doc, NULL},
    {NULL}};
