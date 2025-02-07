/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* polygon_object_str(PolygonObject* self) {
    char buffer[GDSTK_PRINT_BUFFER_COUNT];
    snprintf(buffer, COUNT(buffer),
             "Polygon at layer %" PRIu32 ", datatype %" PRIu32 ", with %" PRIu64 " points",
             get_layer(self->polygon->tag), get_type(self->polygon->tag),
             self->polygon->point_array.count);
    return PyUnicode_FromString(buffer);
}

static void polygon_object_dealloc(PolygonObject* self) {
    if (self->polygon) {
        self->polygon->clear();
        free_allocation(self->polygon);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
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
    polygon->tag = make_tag(layer, datatype);
    polygon->owner = self;
    if (parse_point_sequence(py_points, polygon->point_array, "points") < 0) {
        return -1;
    }
    if (polygon->point_array.count == 0) {
        PyErr_SetString(PyExc_ValueError, "Cannot create a polygon without vertices.");
        return -1;
    }
    return 0;
}

static PyObject* polygon_object_copy(PolygonObject* self, PyObject*) {
    PolygonObject* result = PyObject_New(PolygonObject, &polygon_object_type);
    result = (PolygonObject*)PyObject_Init((PyObject*)result, &polygon_object_type);
    result->polygon = (Polygon*)allocate_clear(sizeof(Polygon));
    result->polygon->copy_from(*self->polygon);
    result->polygon->owner = result;
    return (PyObject*)result;
}

static PyObject* polygon_object_deepcopy(PolygonObject* self, PyObject* arg) {
    return polygon_object_copy(self, NULL);
}

static PyObject* polygon_object_area(PolygonObject* self, PyObject*) {
    const double area = self->polygon->area();
    return PyFloat_FromDouble(area);
}

static PyObject* polygon_object_perimeter(PolygonObject* self, PyObject*) {
    const double perimeter = self->polygon->perimeter();
    return PyFloat_FromDouble(perimeter);
}

static PyObject* polygon_object_bounding_box(PolygonObject* self, PyObject*) {
    Vec2 min, max;
    self->polygon->bounding_box(min, max);
    if (min.x > max.x) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return Py_BuildValue("((dd)(dd))", min.x, min.y, max.x, max.y);
}

static PyObject* polygon_object_contain(PolygonObject* self, PyObject* args) {
    PyObject* result;
    Polygon* polygon = self->polygon;

    if (PyTuple_GET_SIZE(args) == 2) {
        PyObject* x = PyTuple_GET_ITEM(args, 0);
        PyObject* y = PyTuple_GET_ITEM(args, 1);
        if (PyNumber_Check(x) && PyNumber_Check(y) && !PyComplex_Check(x) && !PyComplex_Check(y)) {
            Vec2 point;
            point.x = PyFloat_AsDouble(x);
            point.y = PyFloat_AsDouble(y);
            result = polygon->contain(point) ? Py_True : Py_False;
            Py_INCREF(result);
            return result;
        }
    }

    Array<Vec2> points = {};
    if (parse_point_sequence(args, points, "points") < 0) {
        points.clear();
        return NULL;
    }

    if (points.count == 1) {
        result = polygon->contain(points[0]) ? Py_True : Py_False;
        Py_INCREF(result);
    } else {
        result = PyTuple_New(points.count);
        for (uint64_t i = 0; i < points.count; i++) {
            PyObject* res = polygon->contain(points[i]) ? Py_True : Py_False;
            Py_INCREF(res);
            PyTuple_SET_ITEM(result, i, res);
        }
    }
    points.clear();
    return result;
}

static PyObject* polygon_object_contain_all(PolygonObject* self, PyObject* args) {
    Polygon* polygon = self->polygon;
    Array<Vec2> points = {};
    if (parse_point_sequence(args, points, "points") < 0) {
        points.clear();
        return NULL;
    }
    PyObject* result = polygon->contain_all(points) ? Py_True : Py_False;
    points.clear();
    Py_INCREF(result);
    return result;
}

static PyObject* polygon_object_contain_any(PolygonObject* self, PyObject* args) {
    Polygon* polygon = self->polygon;
    Array<Vec2> points = {};
    if (parse_point_sequence(args, points, "points") < 0) {
        points.clear();
        return NULL;
    }
    PyObject* result = polygon->contain_any(points) ? Py_True : Py_False;
    points.clear();
    Py_INCREF(result);
    return result;
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
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert dx to vector or float.");
            return NULL;
        }
        v.y = PyFloat_AsDouble(dy);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Unable to convert dy to float and dx is not a vector.");
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

static PyObject* polygon_object_transform(PolygonObject* self, PyObject* args, PyObject* kwds) {
    const char matrix_error[] = "Matrix must be a 2×2, 2×3, 3×2, or 3×3 array-like object.";
    double m[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const char* keywords[] = {"magnification", "x_reflection", "rotation",
                              "translation",   "matrix",       NULL};
    PyObject* matrix_obj = Py_None;
    PyObject* translation_obj = Py_None;
    double magnification = 1;
    double rotation = 0;
    Vec2 origin = {0, 0};
    int x_reflection = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|dpdOO:transform", (char**)keywords,
                                     &magnification, &x_reflection, &rotation, &translation_obj,
                                     &matrix_obj))
        return NULL;

    if (translation_obj != Py_None && parse_point(translation_obj, origin, "translation") < 0)
        return NULL;

    if (origin.x != 0 || origin.y != 0 || rotation != 0 || magnification != 1 || x_reflection > 0) {
        self->polygon->transform(magnification, x_reflection > 0, rotation, origin);
    }

    if (matrix_obj != Py_None) {
        if (!PySequence_Check(matrix_obj)) {
            PyErr_SetString(PyExc_TypeError, matrix_error);
            return NULL;
        }
        Py_ssize_t rows = PySequence_Size(matrix_obj);
        if (rows != 2 && rows != 3) {
            PyErr_SetString(PyExc_TypeError, matrix_error);
            return NULL;
        }
        const bool homogeneous = rows == 3;
        for (rows--; rows >= 0; rows--) {
            PyObject* row_obj = PySequence_ITEM(matrix_obj, rows);
            if (!row_obj) {
                PyErr_SetString(PyExc_RuntimeError, "Unable to get element from matrix.");
                return NULL;
            }
            if (!PySequence_Check(row_obj)) {
                Py_DECREF(row_obj);
                PyErr_SetString(PyExc_TypeError, matrix_error);
                return NULL;
            }
            Py_ssize_t cols = PySequence_Size(row_obj);
            if (cols != 2 && cols != 3) {
                Py_DECREF(row_obj);
                PyErr_SetString(PyExc_TypeError, matrix_error);
                return NULL;
            }
            for (cols--; cols >= 0; cols--) {
                PyObject* elem = PySequence_ITEM(row_obj, cols);
                if (!elem) {
                    Py_DECREF(row_obj);
                    PyErr_SetString(PyExc_RuntimeError, "Unable to get element from matrix.");
                    return NULL;
                }
                m[rows * 3 + cols] = PyFloat_AsDouble(elem);
                Py_DECREF(elem);
                if (PyErr_Occurred()) {
                    Py_DECREF(row_obj);
                    PyErr_SetString(PyExc_TypeError, "Unable to convert matrix element to float.");
                    return NULL;
                }
            }
            Py_DECREF(row_obj);
        }

        Array<Vec2>* point_array = &self->polygon->point_array;
        Vec2* p = point_array->items;
        if (homogeneous) {
            for (uint64_t num = point_array->count; num > 0; num--, p++) {
                double x = p->x;
                double y = p->y;
                double w = 1.0 / (m[6] * x + m[7] * y + m[8]);
                p->x = (m[0] * x + m[1] * y + m[2]) * w;
                p->y = (m[3] * x + m[4] * y + m[5]) * w;
            }
        } else {
            for (uint64_t num = point_array->count; num > 0; num--, p++) {
                double x = p->x;
                double y = p->y;
                p->x = m[0] * x + m[1] * y + m[2];
                p->y = m[3] * x + m[4] * y + m[5];
            }
        }
    }

    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* polygon_object_fillet(PolygonObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"radius", "tolerance", NULL};
    bool free_items = false;
    double radius = 0;
    double tolerance = 0.01;
    PyObject* radius_obj = NULL;
    Array<double> radius_array = {};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|d:fillet", (char**)keywords, &radius_obj,
                                     &tolerance))
        return NULL;
    if (tolerance <= 0) {
        PyErr_SetString(PyExc_ValueError, "Tolerance must be positive.");
        return NULL;
    }
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
    self->polygon->fillet(radius_array, tolerance);
    if (free_items) free_allocation(radius_array.items);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* polygon_object_fracture(PolygonObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"max_points", "precision", NULL};
    uint64_t max_points = 199;
    double precision = 1e-3;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Kd:fracture", (char**)keywords, &max_points,
                                     &precision))
        return NULL;

    if (precision <= 0) {
        PyErr_SetString(PyExc_ValueError, "Precision must be positive.");
        return NULL;
    }

    Array<Polygon*> array = {};
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

static PyObject* polygon_object_apply_repetition(PolygonObject* self, PyObject*) {
    Array<Polygon*> array = {};
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
    Py_ssize_t count;
    if (!PyArg_ParseTuple(args, "Hs#:set_gds_property", &attribute, &value, &count)) return NULL;
    if (count >= 0) set_gds_property(self->polygon->properties, attribute, value, (uint64_t)count);
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
    PyObject* result = PyUnicode_FromStringAndSize((char*)value->bytes, (Py_ssize_t)value->count);
    if (PyErr_Occurred()) {
        Py_XDECREF(result);
        PyErr_Clear();
        result = PyBytes_FromStringAndSize((char*)value->bytes, (Py_ssize_t)value->count);
    }
    return result;
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
    {"__deepcopy__", (PyCFunction)polygon_object_deepcopy, METH_VARARGS | METH_KEYWORDS,
     polygon_object_deepcopy_doc},
    {"area", (PyCFunction)polygon_object_area, METH_NOARGS, polygon_object_area_doc},
    {"perimeter", (PyCFunction)polygon_object_perimeter, METH_NOARGS, polygon_object_perimeter_doc},
    {"bounding_box", (PyCFunction)polygon_object_bounding_box, METH_NOARGS,
     polygon_object_bounding_box_doc},
    {"contain", (PyCFunction)polygon_object_contain, METH_VARARGS, polygon_object_contain_doc},
    {"contain_all", (PyCFunction)polygon_object_contain_all, METH_VARARGS,
     polygon_object_contain_all_doc},
    {"contain_any", (PyCFunction)polygon_object_contain_any, METH_VARARGS,
     polygon_object_contain_any_doc},
    {"translate", (PyCFunction)polygon_object_translate, METH_VARARGS,
     polygon_object_translate_doc},
    {"scale", (PyCFunction)polygon_object_scale, METH_VARARGS | METH_KEYWORDS,
     polygon_object_scale_doc},
    {"mirror", (PyCFunction)polygon_object_mirror, METH_VARARGS | METH_KEYWORDS,
     polygon_object_mirror_doc},
    {"rotate", (PyCFunction)polygon_object_rotate, METH_VARARGS | METH_KEYWORDS,
     polygon_object_rotate_doc},
    {"transform", (PyCFunction)polygon_object_transform, METH_VARARGS | METH_KEYWORDS,
     polygon_object_transform_doc},
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
    return PyLong_FromUnsignedLongLong(get_layer(self->polygon->tag));
}

static int polygon_object_set_layer(PolygonObject* self, PyObject* arg, void*) {
    set_layer(self->polygon->tag, (uint32_t)PyLong_AsUnsignedLongLong(arg));
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert layer to int.");
        return -1;
    }
    return 0;
}

static PyObject* polygon_object_get_datatype(PolygonObject* self, void*) {
    return PyLong_FromUnsignedLongLong(get_type(self->polygon->tag));
}

static int polygon_object_set_datatype(PolygonObject* self, PyObject* arg, void*) {
    set_type(self->polygon->tag, (uint32_t)PyLong_AsUnsignedLongLong(arg));
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
