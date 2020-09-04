/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* flexpath_object_str(FlexPathObject* self) {
    char buffer[64];
    snprintf(buffer, COUNT(buffer), "FlexPath with %" PRId64 " paths and %" PRId64 " points",
             self->flexpath->num_elements, self->flexpath->spine.point_array.size);
    return PyUnicode_FromString(buffer);
}

static void flexpath_cleanup(FlexPathObject* self) {
    FlexPathElement* el = self->flexpath->elements;
    for (int64_t j = self->flexpath->num_elements - 1; j >= 0; j--, el++) {
        Py_XDECREF(el->join_function_data);
        Py_XDECREF(el->end_function_data);
        Py_XDECREF(el->bend_function_data);
    }
    self->flexpath->clear();
    free(self->flexpath);
    self->flexpath = NULL;
}

static void flexpath_object_dealloc(FlexPathObject* self) {
    if (self->flexpath) flexpath_cleanup(self);
    PyObject_Del(self);
}

static int flexpath_object_init(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_points = NULL;
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    PyObject* py_joins = NULL;
    PyObject* py_ends = NULL;
    PyObject* py_bend_radius = NULL;
    PyObject* py_bend_function = NULL;
    PyObject* py_layer = NULL;
    PyObject* py_datatype = NULL;
    double tolerance = 1e-2;
    int gdsii_path = 0;
    int scale_width = 1;
    const char* keywords[] = {"points",     "width",       "offset",        "joins",
                              "ends",       "bend_radius", "bend_function", "tolerance",
                              "gdsii_path", "scale_width", "layer",         "datatype",
                              NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OOOOOdppOO:FlexPath", (char**)keywords,
                                     &py_points, &py_width, &py_offset, &py_joins, &py_ends,
                                     &py_bend_radius, &py_bend_function, &tolerance, &gdsii_path,
                                     &scale_width, &py_layer, &py_datatype))
        return -1;

    if (self->flexpath) {
        FlexPath* flexpath = self->flexpath;
        FlexPathElement* el = flexpath->elements;
        for (int64_t i = 0; i < flexpath->num_elements; i++, el++) {
            Py_XDECREF(el->join_function_data);
            Py_XDECREF(el->end_function_data);
            Py_XDECREF(el->bend_function_data);
        }
        flexpath->clear();
    } else {
        self->flexpath = (FlexPath*)calloc(1, sizeof(FlexPath));
    }
    FlexPath* flexpath = self->flexpath;

    Vec2 single_point;
    if (parse_point(py_points, single_point, "points") == 0) {
        flexpath->spine.point_array.append(single_point);
    } else {
        PyErr_Clear();
        if (parse_point_sequence(py_points, flexpath->spine.point_array, "points") < 0) {
            flexpath_cleanup(self);
            return -1;
        }
    }

    int64_t num_elements = 1;
    const int64_t size = flexpath->spine.point_array.size;
    if (size > 1)
        flexpath->spine.last_ctrl = flexpath->spine.point_array[size - 2];

    if (PySequence_Check(py_width)) {
        num_elements = PySequence_Length(py_width);
        flexpath->num_elements = num_elements;
        flexpath->elements = (FlexPathElement*)calloc(num_elements, sizeof(FlexPathElement));
        if (py_offset && PySequence_Check(py_offset)) {
            if (PySequence_Length(py_offset) != num_elements) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError,
                                "Sequences width and offset must have the same length.");
                return -1;
            }

            // Case 1: width and offset are sequences with the same length
            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PySequence_ITEM(py_width, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRId64 " from width sequence.", i);
                    return -1;
                }
                const double half_width = 0.5 * PyFloat_AsDouble(item);
                Py_DECREF(item);
                if (PyErr_Occurred()) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert width[%" PRId64 "] to float.", i);
                    return -1;
                }

                item = PySequence_ITEM(py_offset, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRId64 " from offset sequence.", i);
                    return -1;
                }
                const double offset = PyFloat_AsDouble(item);
                Py_DECREF(item);
                if (PyErr_Occurred()) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert offset[%" PRId64 "] to float.", i);
                    return -1;
                }

                const Vec2 half_width_and_offset = {half_width, offset};
                el->half_width_and_offset.ensure_slots(size);
                Vec2* wo = el->half_width_and_offset.items;
                for (int64_t j = 0; j < size; j++) *wo++ = half_width_and_offset;
                el->half_width_and_offset.size = size;
            }
        } else {
            // Case 2: width is a sequence, offset a number
            const double offset = py_offset == NULL ? 0 : PyFloat_AsDouble(py_offset);
            if (PyErr_Occurred()) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert offset to float.");
                return -1;
            }

            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PySequence_ITEM(py_width, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRId64 " from width sequence.", i);
                    return -1;
                }
                const double half_width = 0.5 * PyFloat_AsDouble(item);
                Py_DECREF(item);
                if (PyErr_Occurred()) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert width[%" PRId64 "] to float.", i);
                    return -1;
                }

                const Vec2 half_width_and_offset = {half_width,
                                                    (i - 0.5 * (num_elements - 1)) * offset};
                el->half_width_and_offset.ensure_slots(size);
                Vec2* wo = el->half_width_and_offset.items;
                for (int64_t j = 0; j < size; j++) *wo++ = half_width_and_offset;
                el->half_width_and_offset.size = size;
            }
        }
    } else if (py_offset && PySequence_Check(py_offset)) {
        // Case 3: offset is a sequence, width a number
        num_elements = PySequence_Length(py_offset);
        flexpath->num_elements = num_elements;
        flexpath->elements = (FlexPathElement*)calloc(num_elements, sizeof(FlexPathElement));
        const double half_width = 0.5 * PyFloat_AsDouble(py_width);
        if (PyErr_Occurred()) {
            flexpath_cleanup(self);
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert width to float.");
            return -1;
        }

        FlexPathElement* el = flexpath->elements;
        for (int64_t i = 0; i < num_elements; i++, el++) {
            PyObject* item = PySequence_ITEM(py_offset, i);
            if (item == NULL) {
                flexpath_cleanup(self);
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to retrieve item %" PRId64 " from offset sequence.", i);
                return -1;
            }
            const double offset = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                flexpath_cleanup(self);
                PyErr_Format(PyExc_RuntimeError, "Unable to convert offset[%" PRId64 "] to float.",
                             i);
                return -1;
            }

            const Vec2 half_width_and_offset = {half_width, offset};
            el->half_width_and_offset.ensure_slots(size);
            Vec2* wo = el->half_width_and_offset.items;
            for (int64_t j = 0; j < size; j++) *wo++ = half_width_and_offset;
            el->half_width_and_offset.size = size;
        }
    } else {
        // Case 4: width and offset are numbers
        flexpath->num_elements = 1;
        flexpath->elements = (FlexPathElement*)calloc(1, sizeof(FlexPathElement));
        FlexPathElement* el = flexpath->elements;
        const double half_width = 0.5 * PyFloat_AsDouble(py_width);
        if (PyErr_Occurred()) {
            flexpath_cleanup(self);
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert width to float.");
            return -1;
        }
        const double offset = py_offset == NULL ? 0 : PyFloat_AsDouble(py_offset);
        if (PyErr_Occurred()) {
            flexpath_cleanup(self);
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert offset to float.");
            return -1;
        }

        const Vec2 half_width_and_offset = {half_width, offset};
        el->half_width_and_offset.ensure_slots(size);
        Vec2* wo = el->half_width_and_offset.items;
        for (int64_t j = 0; j < size; j++) *wo++ = half_width_and_offset;
        el->half_width_and_offset.size = size;
    }

    if (py_layer) {
        if (PyList_Check(py_layer)) {
            if (PyList_GET_SIZE(py_layer) != num_elements) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError,
                                "List layer must have the same length as the number of paths.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_layer, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to get item %" PRId64 " from layer list.", i);
                    return -1;
                }
                el->layer = PyLong_AsLong(item);
                if (PyErr_Occurred()) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError, "Unable to convert layer[%" PRId64 "] to int.",
                                 i);
                    return -1;
                }
            }
        } else {
            const int16_t layer = PyLong_AsLong(py_layer);
            if (PyErr_Occurred()) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert layer to int.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++) (el++)->layer = layer;
        }
    }

    if (py_datatype) {
        if (PyList_Check(py_datatype)) {
            if (PyList_GET_SIZE(py_datatype) != num_elements) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError,
                                "List datatype must have the same length as the number of paths.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_datatype, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to get item %" PRId64 " from datatype list.", i);
                    return -1;
                }
                el->datatype = PyLong_AsLong(item);
                if (PyErr_Occurred()) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert datatype[%" PRId64 "] to int.", i);
                    return -1;
                }
            }
        } else {
            const int16_t datatype = PyLong_AsLong(py_datatype);
            if (PyErr_Occurred()) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert datatype to int.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++) (el++)->datatype = datatype;
        }
    }

    if (py_joins) {
        if (PyList_Check(py_joins)) {
            if (PyList_GET_SIZE(py_joins) != num_elements) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError,
                                "List joins must have the same length as the number of paths.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_joins, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRId64 " from joins list.", i);
                    return -1;
                }
                if (PyCallable_Check(item)) {
                    el->join_type = JoinType::Function;
                    el->join_function = (JoinFunction)custom_join_function;
                    el->join_function_data = (void*)item;
                    Py_INCREF(item);
                } else {
                    if (!PyUnicode_Check(item)) {
                        flexpath_cleanup(self);
                        PyErr_SetString(
                            PyExc_TypeError,
                            "Argument joins must be one of 'natural', 'miter', 'bevel', 'round', 'smooth', a callable, or a list of those.");
                        return -1;
                    }
                    JoinType jt = JoinType::Natural;
                    if (PyUnicode_CompareWithASCIIString(item, "miter") == 0)
                        jt = JoinType::Miter;
                    else if (PyUnicode_CompareWithASCIIString(item, "bevel") == 0)
                        jt = JoinType::Bevel;
                    else if (PyUnicode_CompareWithASCIIString(item, "round") == 0)
                        jt = JoinType::Round;
                    else if (PyUnicode_CompareWithASCIIString(item, "smooth") == 0)
                        jt = JoinType::Smooth;
                    else if (PyUnicode_CompareWithASCIIString(item, "natural") != 0) {
                        flexpath_cleanup(self);
                        PyErr_SetString(
                            PyExc_RuntimeError,
                            "Argument joins must be one of 'natural', 'miter', 'bevel', 'round', 'smooth', a callable, or a list of those.");
                        return -1;
                    }
                    el->join_type = jt;
                }
            }
        } else if (PyCallable_Check(py_joins)) {
            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++, el++) {
                el->join_type = JoinType::Function;
                el->join_function = (JoinFunction)custom_join_function;
                el->join_function_data = (void*)py_joins;
                Py_INCREF(py_joins);
            }
        } else {
            JoinType jt = JoinType::Natural;
            if (!PyUnicode_Check(py_joins)) {
                flexpath_cleanup(self);
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Argument joins must be one of 'natural', 'miter', 'bevel', 'round', 'smooth', a callable, or a list of those.");
                return -1;
            }
            if (PyUnicode_CompareWithASCIIString(py_joins, "miter") == 0)
                jt = JoinType::Miter;
            else if (PyUnicode_CompareWithASCIIString(py_joins, "bevel") == 0)
                jt = JoinType::Bevel;
            else if (PyUnicode_CompareWithASCIIString(py_joins, "round") == 0)
                jt = JoinType::Round;
            else if (PyUnicode_CompareWithASCIIString(py_joins, "smooth") == 0)
                jt = JoinType::Smooth;
            else if (PyUnicode_CompareWithASCIIString(py_joins, "natural") != 0) {
                flexpath_cleanup(self);
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Argument joins must be one of 'natural', 'miter', 'bevel', 'round', 'smooth', a callable, or a list of those.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++) (el++)->join_type = jt;
        }
    }

    if (py_ends) {
        if (PyList_Check(py_ends)) {
            if (PyList_GET_SIZE(py_ends) != num_elements) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError,
                                "List ends must have the same length as the number of paths.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_ends, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRId64 " from ends list.", i);
                    return -1;
                }
                if (PyCallable_Check(item)) {
                    el->end_type = EndType::Function;
                    el->end_function = (EndFunction)custom_end_function;
                    el->end_function_data = (void*)item;
                    Py_INCREF(item);
                } else {
                    EndType et = EndType::Flush;
                    if (PyUnicode_Check(item)) {
                        if (PyUnicode_CompareWithASCIIString(item, "extended") == 0) {
                            et = EndType::Extended;
                            el->end_extensions = Vec2{-1, -1};
                        } else if (PyUnicode_CompareWithASCIIString(item, "round") == 0)
                            et = EndType::Round;
                        else if (PyUnicode_CompareWithASCIIString(item, "smooth") == 0)
                            et = EndType::Smooth;
                        else if (PyUnicode_CompareWithASCIIString(item, "flush") != 0) {
                            flexpath_cleanup(self);
                            PyErr_SetString(
                                PyExc_RuntimeError,
                                "Argument ends must be one of 'flush', 'extended', 'round', 'smooth', a 2-tuple, a callable, or a list of those.");
                            return -1;
                        }
                    } else {
                        et = EndType::Extended;
                        if (!PyTuple_Check(item) ||
                            PyArg_ParseTuple(item, "dd", &el->end_extensions.u,
                                             &el->end_extensions.v) < 0) {
                            flexpath_cleanup(self);
                            PyErr_SetString(
                                PyExc_RuntimeError,
                                "Argument ends must be one of 'flush', 'extended', 'round', 'smooth', a 2-tuple, a callable, or a list of those.");
                            return -1;
                        }
                    }
                    el->end_type = et;
                }
            }
        } else if (PyCallable_Check(py_ends)) {
            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++, el++) {
                el->end_type = EndType::Function;
                el->end_function = (EndFunction)custom_end_function;
                el->end_function_data = (void*)py_ends;
                Py_INCREF(py_ends);
            }
        } else {
            EndType et = EndType::Flush;
            Vec2 ex = Vec2{-1, -1};
            if (PyUnicode_Check(py_ends)) {
                if (PyUnicode_CompareWithASCIIString(py_ends, "extended") == 0)
                    et = EndType::Extended;
                else if (PyUnicode_CompareWithASCIIString(py_ends, "round") == 0)
                    et = EndType::Round;
                else if (PyUnicode_CompareWithASCIIString(py_ends, "smooth") == 0)
                    et = EndType::Smooth;
                else if (PyUnicode_CompareWithASCIIString(py_ends, "flush") != 0) {
                    flexpath_cleanup(self);
                    PyErr_SetString(
                        PyExc_RuntimeError,
                        "Argument ends must be one of 'flush', 'extended', 'round', 'smooth', a 2-tuple, a callable, or a list of those.");
                    return -1;
                }
            } else {
                et = EndType::Extended;
                if (!PyTuple_Check(py_ends) || PyArg_ParseTuple(py_ends, "dd", &ex.u, &ex.v) < 0) {
                    flexpath_cleanup(self);
                    PyErr_SetString(
                        PyExc_RuntimeError,
                        "Argument ends must be one of 'flush', 'extended', 'round', 'smooth', a 2-tuple, a callable, or a list of those.");
                    return -1;
                }
            }
            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++, el++) {
                el->end_type = et;
                el->end_extensions = ex;
            }
        }
    }

    if (py_bend_radius) {
        if (PyList_Check(py_bend_radius)) {
            if (PyList_GET_SIZE(py_bend_radius) != num_elements) {
                flexpath_cleanup(self);
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Sequence bend_radius must have the same length as the number of paths.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_bend_radius, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRId64 " from bend_radius sequence.",
                                 i);
                    return -1;
                }
                const double bend_radius = PyFloat_AsDouble(item);
                if (PyErr_Occurred()) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert bend_radius[%" PRId64 "] to float.", i);
                    return -1;
                }
                if (bend_radius > 0) {
                    el->bend_type = BendType::Circular;
                    el->bend_radius = bend_radius;
                }
            }
        } else {
            const double bend_radius = PyFloat_AsDouble(py_bend_radius);
            if (PyErr_Occurred()) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert bend_radius to float.");
                return -1;
            }
            if (bend_radius > 0) {
                FlexPathElement* el = flexpath->elements;
                for (int64_t i = 0; i < num_elements; i++, el++) {
                    el->bend_type = BendType::Circular;
                    el->bend_radius = bend_radius;
                }
            }
        }
    }

    if (py_bend_function && py_bend_function != Py_None) {
        if (PyList_Check(py_bend_function)) {
            if (PyList_GET_SIZE(py_bend_function) != num_elements) {
                flexpath_cleanup(self);
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Sequence bend_function must have the same length as the number of paths.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_bend_function, i);
                if (item == NULL || !PyCallable_Check(item)) {
                    flexpath_cleanup(self);
                    PyErr_Format(
                        PyExc_RuntimeError,
                        "Unable to get callable from item %" PRId64 " from bend_function list.", i);
                    return -1;
                }
                el->bend_type = BendType::Function;
                el->bend_function = (BendFunction)custom_bend_function;
                el->bend_function_data = (void*)item;
                Py_INCREF(item);
            }
        } else {
            if (!PyCallable_Check(py_bend_function)) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_TypeError,
                                "Argument bend_function must be a list or a callable.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (int64_t i = 0; i < num_elements; i++, el++) {
                el->bend_type = BendType::Function;
                el->bend_function = (BendFunction)custom_bend_function;
                el->bend_function_data = (void*)py_bend_function;
                Py_INCREF(py_bend_function);
            }
        }
    }

    flexpath->spine.tolerance = tolerance;
    flexpath->gdsii_path = gdsii_path > 0;
    flexpath->scale_width = scale_width > 0;
    properties_clear(flexpath->properties);
    flexpath->properties = NULL;
    flexpath->owner = self;
    return 0;
}

static PyObject* flexpath_object_copy(FlexPathObject* self, PyObject* args) {
    FlexPathObject* result = PyObject_New(FlexPathObject, &flexpath_object_type);
    result = (FlexPathObject*)PyObject_Init((PyObject*)result, &flexpath_object_type);
    result->flexpath = (FlexPath*)calloc(1, sizeof(FlexPath));
    result->flexpath->copy_from(*self->flexpath);
    result->flexpath->owner = result;
    return (PyObject*)result;
}

static PyObject* flexpath_object_spine(FlexPathObject* self, PyObject* args) {
    const Array<Vec2>* point_array = &self->flexpath->spine.point_array;
    npy_intp dims[] = {(npy_intp)point_array->size, 2};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    memcpy(data, point_array->items, sizeof(double) * point_array->size * 2);
    return (PyObject*)result;
}

static PyObject* flexpath_object_widths(FlexPathObject* self, PyObject* args) {
    const FlexPath* flexpath = self->flexpath;
    npy_intp dims[] = {(npy_intp)flexpath->spine.point_array.size, flexpath->num_elements};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    double* d = data;
    for (int64_t j = 0; j < flexpath->spine.point_array.size; j++) {
        const FlexPathElement* el = flexpath->elements;
        for (int64_t i = 0; i < flexpath->num_elements; i++)
            *d++ = 2 * (el++)->half_width_and_offset[j].u;
    }
    return (PyObject*)result;
}

static PyObject* flexpath_object_offsets(FlexPathObject* self, PyObject* args) {
    const FlexPath* flexpath = self->flexpath;
    npy_intp dims[] = {(npy_intp)flexpath->spine.point_array.size, flexpath->num_elements};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    double* d = data;
    for (int64_t j = 0; j < flexpath->spine.point_array.size; j++) {
        const FlexPathElement* el = flexpath->elements;
        for (int64_t i = 0; i < flexpath->num_elements; i++)
            *d++ = (el++)->half_width_and_offset[j].v;
    }
    return (PyObject*)result;
}

static PyObject* flexpath_object_to_polygons(FlexPathObject* self, PyObject* args) {
    Array<Polygon*> array = self->flexpath->to_polygons();
    PyObject* result = PyList_New(array.size);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return array.");
        for (int64_t i = 0; i < array.size; i++) array[i]->clear();
        array.clear();
        return NULL;
    }
    for (int64_t i = 0; i < array.size; i++) {
        PolygonObject* item = PyObject_New(PolygonObject, &polygon_object_type);
        item = (PolygonObject*)PyObject_Init((PyObject*)item, &polygon_object_type);
        item->polygon = array[i];
        item->polygon->owner = item;
        PyList_SET_ITEM(result, i, (PyObject*)item);
    }
    array.clear();
    return (PyObject*)result;
}

static PyObject* flexpath_object_set_layers(FlexPathObject* self, PyObject* arg) {
    if (!PySequence_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Value must be a sequence of layer numbers.");
        return NULL;
    }
    int64_t len = PySequence_Length(arg);
    FlexPath* flexpath = self->flexpath;
    if (len != flexpath->num_elements) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Length of layer sequence must match the number of paths.");
        return NULL;
    }
    for (int64_t i = 0; i < len; i++) {
        PyObject* item = PySequence_ITEM(arg, i);
        if (item == NULL) {
            PyErr_Format(PyExc_RuntimeError, "Unable to get item %" PRId64 " from sequence.", i);
            return NULL;
        }
        flexpath->elements[i].layer = PyLong_AsLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_Format(PyExc_RuntimeError, "Unable to convert sequence item %" PRId64 " to int.",
                         i);
            return NULL;
        }
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_set_datatypes(FlexPathObject* self, PyObject* arg) {
    if (!PySequence_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Value must be a sequence of datatype numbers.");
        return NULL;
    }
    int64_t len = PySequence_Length(arg);
    FlexPath* flexpath = self->flexpath;
    if (len != flexpath->num_elements) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Length of datatype sequence must match the number of paths.");
        return NULL;
    }
    for (int64_t i = 0; i < len; i++) {
        PyObject* item = PySequence_ITEM(arg, i);
        if (item == NULL) {
            PyErr_Format(PyExc_RuntimeError, "Unable to get item %" PRId64 " from sequence.", i);
            return NULL;
        }
        flexpath->elements[i].datatype = PyLong_AsLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_Format(PyExc_TypeError, "Unable to convert sequence item %" PRId64 " to int.", i);
            return NULL;
        }
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static int parse_flexpath_width(const FlexPath flexpath, PyObject* py_width, double* width) {
    if (PySequence_Check(py_width)) {
        if (PySequence_Length(py_width) < flexpath.num_elements) {
            PyErr_SetString(PyExc_RuntimeError, "Sequence width doesn't have enough elements.");
            return -1;
        }
        for (int64_t i = 0; i < flexpath.num_elements; i++) {
            PyObject* item = PySequence_ITEM(py_width, i);
            if (item == NULL) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get item %" PRId64 " from sequence width.", i);
                return -1;
            }
            *width++ = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to convert item %" PRId64 " from sequence width to float.", i);
                return -1;
            }
        }
    } else {
        const double value = PyFloat_AsDouble(py_width);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert width to float.");
            return -1;
        }
        for (int64_t i = 0; i < flexpath.num_elements; i++) *width++ = value;
    }
    return 0;
}

// If offset is a single number, it's the new distance between paths (analogous to what is used in
// init).
static int parse_flexpath_offset(const FlexPath flexpath, PyObject* py_offset, double* offset) {
    if (PySequence_Check(py_offset)) {
        if (PySequence_Length(py_offset) < flexpath.num_elements) {
            PyErr_SetString(PyExc_RuntimeError, "Sequence offset doesn't have enough elements.");
            return -1;
        }
        for (int64_t i = 0; i < flexpath.num_elements; i++) {
            PyObject* item = PySequence_ITEM(py_offset, i);
            if (item == NULL) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get item %" PRId64 " from sequence offset.", i);
                return -1;
            }
            *offset++ = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to convert item %" PRId64 " from sequence offset to float.",
                             i);
                return -1;
            }
        }
    } else {
        const double value = PyFloat_AsDouble(py_offset);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert offset to float.");
            return -1;
        }
        for (int64_t i = 0; i < flexpath.num_elements; i++)
            *offset++ = (i - 0.5 * (flexpath.num_elements - 1)) * value;
    }
    return 0;
}

static PyObject* flexpath_object_horizontal(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_coord;
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    int relative = 0;
    const char* keywords[] = {"x", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:horizontal", (char**)keywords, &py_coord,
                                     &py_width, &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    double* buffer = (double*)malloc(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != NULL && py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            free(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != NULL && py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            free(buffer);
            return NULL;
        }
    }
    if (PySequence_Check(py_coord)) {
        int64_t size;
        double* coord = parse_sequence_double(py_coord, size, "x");
        if (coord == NULL) {
            free(buffer);
            return NULL;
        }
        flexpath->horizontal(coord, size, width, offset, relative > 0);
        free(coord);
    } else {
        double single = PyFloat_AsDouble(py_coord);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert coordinate to float.");
            free(buffer);
            return NULL;
        }
        flexpath->horizontal(&single, 1, width, offset, relative > 0);
    }
    free(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_vertical(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_coord;
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    int relative = 0;
    const char* keywords[] = {"y", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:vertical", (char**)keywords, &py_coord,
                                     &py_width, &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    double* buffer = (double*)malloc(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != NULL && py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            free(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != NULL && py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            free(buffer);
            return NULL;
        }
    }
    if (PySequence_Check(py_coord)) {
        int64_t size;
        double* coord = parse_sequence_double(py_coord, size, "y");
        if (coord == NULL) {
            free(buffer);
            return NULL;
        }
        flexpath->vertical(coord, size, width, offset, relative > 0);
        free(coord);
    } else {
        double single = PyFloat_AsDouble(py_coord);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert coordinate to float.");
            free(buffer);
            return NULL;
        }
        flexpath->vertical(&single, 1, width, offset, relative > 0);
    }
    free(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_segment(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:segment", (char**)keywords, &xy, &py_width,
                                     &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    Array<Vec2> point_array = {0};
    point_array.ensure_slots(1);
    if (parse_point(xy, *point_array.items, "xy") == 0)
        point_array.size = 1;
    else {
        PyErr_Clear();
        if (parse_point_sequence(xy, point_array, "xy") < 0) {
            point_array.clear();
            return NULL;
        }
    }
    double* buffer = (double*)malloc(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != NULL && py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            point_array.clear();
            free(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != NULL && py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            point_array.clear();
            free(buffer);
            return NULL;
        }
    }
    flexpath->segment(point_array, width, offset, relative > 0);
    point_array.clear();
    free(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_cubic(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:cubic", (char**)keywords, &xy, &py_width,
                                     &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    Array<Vec2> point_array = {0};
    if (parse_point_sequence(xy, point_array, "xy") < 0 || point_array.size < 3) {
        point_array.clear();
        PyErr_SetString(PyExc_RuntimeError,
                        "Argument xy must be a sequence of at least 3 coordinates.");
        return NULL;
    }
    double* buffer = (double*)malloc(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != NULL && py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            point_array.clear();
            free(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != NULL && py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            point_array.clear();
            free(buffer);
            return NULL;
        }
    }
    flexpath->cubic(point_array, width, offset, relative > 0);
    point_array.clear();
    free(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_cubic_smooth(FlexPathObject* self, PyObject* args,
                                              PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:cubic_smooth", (char**)keywords, &xy,
                                     &py_width, &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    Array<Vec2> point_array = {0};
    if (parse_point_sequence(xy, point_array, "xy") < 0 || point_array.size < 2) {
        point_array.clear();
        PyErr_SetString(PyExc_RuntimeError,
                        "Argument xy must be a sequence of at least 2 coordinates.");
        return NULL;
    }
    double* buffer = (double*)malloc(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != NULL && py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            point_array.clear();
            free(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != NULL && py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            point_array.clear();
            free(buffer);
            return NULL;
        }
    }
    flexpath->cubic_smooth(point_array, width, offset, relative > 0);
    point_array.clear();
    free(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_quadratic(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:quadratic", (char**)keywords, &xy,
                                     &py_width, &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    Array<Vec2> point_array = {0};
    point_array.ensure_slots(1);
    if (parse_point_sequence(xy, point_array, "xy") < 0 || point_array.size < 2) {
        point_array.clear();
        PyErr_SetString(PyExc_RuntimeError,
                        "Argument xy must be a sequence of at least 2 coordinates.");
        return NULL;
    }
    double* buffer = (double*)malloc(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != NULL && py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            point_array.clear();
            free(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != NULL && py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            point_array.clear();
            free(buffer);
            return NULL;
        }
    }
    flexpath->quadratic(point_array, width, offset, relative > 0);
    point_array.clear();
    free(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_quadratic_smooth(FlexPathObject* self, PyObject* args,
                                                  PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:quadratic_smooth", (char**)keywords, &xy,
                                     &py_width, &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    Array<Vec2> point_array = {0};
    point_array.ensure_slots(1);
    if (parse_point(xy, *point_array.items, "xy") == 0)
        point_array.size = 1;
    else {
        PyErr_Clear();
        if (parse_point_sequence(xy, point_array, "xy") < 0) {
            point_array.clear();
            return NULL;
        }
    }
    double* buffer = (double*)malloc(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != NULL && py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            point_array.clear();
            free(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != NULL && py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            point_array.clear();
            free(buffer);
            return NULL;
        }
    }
    flexpath->quadratic_smooth(point_array, width, offset, relative > 0);
    point_array.clear();
    free(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_bezier(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:bezier", (char**)keywords, &xy, &py_width,
                                     &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    Array<Vec2> point_array = {0};
    point_array.ensure_slots(1);
    if (parse_point(xy, *point_array.items, "xy") == 0)
        point_array.size = 1;
    else {
        PyErr_Clear();
        if (parse_point_sequence(xy, point_array, "xy") < 0) {
            point_array.clear();
            return NULL;
        }
    }
    double* buffer = (double*)malloc(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != NULL && py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            point_array.clear();
            free(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != NULL && py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            point_array.clear();
            free(buffer);
            return NULL;
        }
    }
    flexpath->bezier(point_array, width, offset, relative > 0);
    point_array.clear();
    free(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_intepolation(FlexPathObject* self, PyObject* args,
                                              PyObject* kwds) {
    PyObject* py_points = NULL;
    PyObject* py_angles = NULL;
    PyObject* py_tension_in = NULL;
    PyObject* py_tension_out = NULL;
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    double initial_curl = 1;
    double final_curl = 1;
    int cycle = 0;
    int relative = 0;
    Vec2* tension;
    double* angles;
    bool* angle_constraints;
    const char* keywords[] = {"points",       "angles",     "tension_in", "tension_out",
                              "initial_curl", "final_curl", "cycle",      "width",
                              "offset",       "relative",   NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOOddpOOp:interpolation", (char**)keywords,
                                     &py_points, &py_angles, &py_tension_in, &py_tension_out,
                                     &initial_curl, &final_curl, &cycle, &py_width, &py_offset,
                                     &relative))
        return NULL;

    FlexPath* flexpath = self->flexpath;
    Array<Vec2> point_array = {0};
    if (parse_point_sequence(py_points, point_array, "points") < 0) {
        point_array.clear();
        return NULL;
    }
    const int64_t size = point_array.size;

    tension = (Vec2*)malloc((sizeof(Vec2) + sizeof(double) + sizeof(bool)) * (size + 1));
    angles = (double*)(tension + (size + 1));
    angle_constraints = (bool*)(angles + (size + 1));

    if (!py_angles || py_angles == Py_None) {
        memset(angle_constraints, 0, sizeof(bool) * (size + 1));
    } else {
        if (PySequence_Length(py_angles) != size + 1) {
            free(tension);
            point_array.clear();
            PyErr_SetString(
                PyExc_TypeError,
                "Argument angles must be None or a sequence with size len(points) + 1.");
            return NULL;
        }
        for (int64_t i = 0; i < size + 1; i++) {
            PyObject* item = PySequence_ITEM(py_angles, i);
            if (!item) {
                free(tension);
                point_array.clear();
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get item %" PRId64 " from angles sequence.", i);
                return NULL;
            }
            if (item == Py_None)
                angle_constraints[i] = false;
            else {
                angle_constraints[i] = true;
                angles[i] = PyFloat_AsDouble(item);
                if (PyErr_Occurred()) {
                    free(tension);
                    point_array.clear();
                    Py_DECREF(item);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert angle[%" PRId64 "] to float.", i);
                    return NULL;
                }
            }
            Py_DECREF(item);
        }
    }

    if (!py_tension_in) {
        Vec2* t = tension;
        for (int64_t i = 0; i < size + 1; i++) (t++)->u = 1;
    } else if (!PySequence_Check(py_tension_in)) {
        double t_in = PyFloat_AsDouble(py_tension_in);
        if (PyErr_Occurred()) {
            free(tension);
            point_array.clear();
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert tension_in to float.");
            return NULL;
        }
        Vec2* t = tension;
        for (int64_t i = 0; i < size + 1; i++) (t++)->u = t_in;
    } else {
        if (PySequence_Length(py_tension_in) != size + 1) {
            free(tension);
            point_array.clear();
            PyErr_SetString(
                PyExc_TypeError,
                "Argument tension_in must be a number or a sequence with size len(points) + 1.");
            return NULL;
        }
        for (int64_t i = 0; i < size + 1; i++) {
            PyObject* item = PySequence_ITEM(py_tension_in, i);
            if (!item) {
                free(tension);
                point_array.clear();
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get item %" PRId64 " from tension_in sequence.", i);
                return NULL;
            }
            tension[i].u = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                free(tension);
                point_array.clear();
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to convert tension_in[%" PRId64 "] to float.", i);
                return NULL;
            }
        }
    }

    if (!py_tension_out) {
        Vec2* t = tension;
        for (int64_t i = 0; i < size + 1; i++) (t++)->v = 1;
    } else if (!PySequence_Check(py_tension_out)) {
        double t_out = PyFloat_AsDouble(py_tension_out);
        if (PyErr_Occurred()) {
            free(tension);
            point_array.clear();
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert tension_out to float.");
            return NULL;
        }
        Vec2* t = tension;
        for (int64_t i = 0; i < size + 1; i++) (t++)->v = t_out;
    } else {
        if (PySequence_Length(py_tension_out) != size + 1) {
            free(tension);
            point_array.clear();
            PyErr_SetString(
                PyExc_TypeError,
                "Argument tension_out must be a number or a sequence with size len(points) + 1.");
            return NULL;
        }
        for (int64_t i = 0; i < size + 1; i++) {
            PyObject* item = PySequence_ITEM(py_tension_out, i);
            if (!item) {
                free(tension);
                point_array.clear();
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get item %" PRId64 " from tension_out sequence.", i);
                return NULL;
            }
            tension[i].v = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                free(tension);
                point_array.clear();
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to convert tension_out[%" PRId64 "] to float.", i);
                return NULL;
            }
        }
    }

    double* buffer = (double*)malloc(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != NULL && py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            free(tension);
            point_array.clear();
            free(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != NULL && py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            free(tension);
            point_array.clear();
            free(buffer);
            return NULL;
        }
    }

    flexpath->interpolation(point_array, angles, angle_constraints, tension, initial_curl,
                            final_curl, cycle > 0, width, offset, relative > 0);

    point_array.clear();
    free(tension);
    free(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_arc(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_radius;
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    double radius_x;
    double radius_y;
    double initial_angle;
    double final_angle;
    double rotation = 0;
    const char* keywords[] = {"radius", "initial_angle", "final_angle", "rotation",
                              "width",  "offset",        NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Odd|dOO:arc", (char**)keywords, &py_radius,
                                     &initial_angle, &final_angle, &rotation, &py_width,
                                     &py_offset))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    if (!PySequence_Check(py_radius)) {
        radius_x = radius_y = PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Unable to convert radius to float.");
            return NULL;
        }
    } else if (PySequence_Length(py_radius) != 2) {
        PyErr_SetString(PyExc_TypeError,
                        "Argument radius must be a number of a sequence of 2 numbers.");
        return NULL;
    } else {
        PyObject* item = PySequence_ITEM(py_radius, 0);
        if (!item) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to get first item from radius sequence.");
            return NULL;
        }
        radius_x = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Unable to convert first item from radius to float.");
            return NULL;
        }
        item = PySequence_ITEM(py_radius, 1);
        if (!item) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to get second item from radius sequence.");
            return NULL;
        }
        radius_y = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Unable to convert second item from radius to float.");
            return NULL;
        }
    }
    double* buffer = (double*)malloc(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != NULL && py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            free(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != NULL && py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            free(buffer);
            return NULL;
        }
    }
    flexpath->arc(radius_x, radius_y, initial_angle, final_angle, rotation, width, offset);
    free(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_turn(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    double radius;
    double angle;
    const char* keywords[] = {"radius", "angle", "width", "offset", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dd|OO:turn", (char**)keywords, &radius, &angle,
                                     &py_width, &py_offset))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    double* buffer = (double*)malloc(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != NULL && py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            free(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != NULL && py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            free(buffer);
            return NULL;
        }
    }
    flexpath->turn(radius, angle, width, offset);
    free(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_parametric(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_function;
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    int relative = 1;
    const char* keywords[] = {"path_function", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:parametric", (char**)keywords, &py_function,
                                     &py_width, &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    if (!PyCallable_Check(py_function)) {
        PyErr_SetString(PyExc_TypeError, "Argument path_function must be callable.");
        return NULL;
    }
    double* buffer = (double*)malloc(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != NULL && py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            free(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != NULL && py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            free(buffer);
            return NULL;
        }
    }
    Py_INCREF(py_function);
    flexpath->parametric((ParametricVec2)eval_parametric_vec2, (void*)py_function, width, offset,
                         relative > 0);
    Py_DECREF(py_function);
    free(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_commands(FlexPathObject* self, PyObject* args) {
    Py_ssize_t size = PyTuple_GET_SIZE(args);
    CurveInstruction* instructions = (CurveInstruction*)malloc(sizeof(CurveInstruction) * size * 2);
    CurveInstruction* instr = instructions;

    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PyTuple_GET_ITEM(args, i);
        if (PyUnicode_Check(item)) {
            Py_ssize_t len = 0;
            const char* command = PyUnicode_AsUTF8AndSize(item, &len);
            if (len != 1) {
                PyErr_SetString(PyExc_RuntimeError,
                                "Path instructions must be single characters or numbers.");
                free(instructions);
                return NULL;
            }
            (instr++)->command = command[0];
        } else {
            if (PyComplex_Check(item)) {
                (instr++)->number = PyComplex_RealAsDouble(item);
                (instr++)->number = PyComplex_ImagAsDouble(item);
            } else {
                (instr++)->number = PyFloat_AsDouble(item);
            }
            if (PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError,
                                "Path instructions must be single characters or numbers.");
                free(instructions);
                return NULL;
            }
        }
    }

    int64_t instr_size = instr - instructions;
    int64_t processed = self->flexpath->commands(instructions, instr_size);
    if (processed < instr_size) {
        PyErr_Format(PyExc_RuntimeError,
                     "Error parsing argument %" PRId64 " in curve construction.", processed);
        free(instructions);
        return NULL;
    }

    free(instructions);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_translate(FlexPathObject* self, PyObject* args) {
    Vec2 v;
    if (!PyArg_ParseTuple(args, "dd:translate", &v.x, &v.y)) return NULL;
    self->flexpath->translate(v);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_scale(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"s", "center", NULL};
    double scale = 0;
    Vec2 center = {0, 0};
    PyObject* center_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|O:scale", (char**)keywords, &scale,
                                     &center_obj))
        return NULL;
    if (parse_point(center_obj, center, "center") < 0) return NULL;
    self->flexpath->scale(scale, center);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_mirror(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"p1", "p2", NULL};
    Vec2 p1;
    Vec2 p2 = {0, 0};
    PyObject* p1_obj = NULL;
    PyObject* p2_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O:mirror", (char**)keywords, &p1_obj, &p2_obj))
        return NULL;
    if (parse_point(p1_obj, p1, "p1") < 0) return NULL;
    if (parse_point(p2_obj, p2, "p2") < 0) return NULL;
    self->flexpath->mirror(p1, p2);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_rotate(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"angle", "center", NULL};
    double angle;
    Vec2 center = {0, 0};
    PyObject* center_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|O:rotate", (char**)keywords, &angle,
                                     &center_obj))
        return NULL;
    if (parse_point(center_obj, center, "center") < 0) return NULL;
    self->flexpath->rotate(angle, center);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_set_property(FlexPathObject* self, PyObject* args) {
    int16_t attr;
    char* value;
    if (!PyArg_ParseTuple(args, "hs:set_property", &attr, &value)) return NULL;
    set_property(self->flexpath->properties, attr, value);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_get_property(FlexPathObject* self, PyObject* args) {
    Property* property = self->flexpath->properties;

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

static PyObject* flexpath_object_delete_property(FlexPathObject* self, PyObject* args) {
    int16_t attr;
    if (!PyArg_ParseTuple(args, "h:delete_property", &attr)) return NULL;
    delete_property(self->flexpath->properties, attr);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyMethodDef flexpath_object_methods[] = {
    {"copy", (PyCFunction)flexpath_object_copy, METH_NOARGS, flexpath_object_copy_doc},
    {"spine", (PyCFunction)flexpath_object_spine, METH_NOARGS, flexpath_object_spine_doc},
    {"widths", (PyCFunction)flexpath_object_widths, METH_NOARGS, flexpath_object_widths_doc},
    {"offsets", (PyCFunction)flexpath_object_offsets, METH_NOARGS, flexpath_object_offsets_doc},
    {"to_polygons", (PyCFunction)flexpath_object_to_polygons, METH_NOARGS,
     flexpath_object_to_polygons_doc},
    {"set_layers", (PyCFunction)flexpath_object_set_layers, METH_VARARGS,
     flexpath_object_set_layers_doc},
    {"set_datatypes", (PyCFunction)flexpath_object_set_datatypes, METH_VARARGS,
     flexpath_object_set_datatypes_doc},
    {"horizontal", (PyCFunction)flexpath_object_horizontal, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_horizontal_doc},
    {"vertical", (PyCFunction)flexpath_object_vertical, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_vertical_doc},
    {"segment", (PyCFunction)flexpath_object_segment, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_segment_doc},
    {"cubic", (PyCFunction)flexpath_object_cubic, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_cubic_doc},
    {"cubic_smooth", (PyCFunction)flexpath_object_cubic_smooth, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_cubic_smooth_doc},
    {"quadratic", (PyCFunction)flexpath_object_quadratic, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_quadratic_doc},
    {"quadratic_smooth", (PyCFunction)flexpath_object_quadratic_smooth,
     METH_VARARGS | METH_KEYWORDS, flexpath_object_quadratic_smooth_doc},
    {"bezier", (PyCFunction)flexpath_object_bezier, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_bezier_doc},
    {"interpolation", (PyCFunction)flexpath_object_intepolation, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_intepolation_doc},
    {"arc", (PyCFunction)flexpath_object_arc, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_arc_doc},
    {"turn", (PyCFunction)flexpath_object_turn, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_turn_doc},
    {"parametric", (PyCFunction)flexpath_object_parametric, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_parametric_doc},
    {"commands", (PyCFunction)flexpath_object_commands, METH_VARARGS, flexpath_object_commands_doc},
    {"translate", (PyCFunction)flexpath_object_translate, METH_VARARGS,
     flexpath_object_translate_doc},
    {"scale", (PyCFunction)flexpath_object_scale, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_scale_doc},
    {"mirror", (PyCFunction)flexpath_object_mirror, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_mirror_doc},
    {"rotate", (PyCFunction)flexpath_object_rotate, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_rotate_doc},
    {"set_property", (PyCFunction)flexpath_object_set_property, METH_VARARGS,
     flexpath_object_set_property_doc},
    {"get_property", (PyCFunction)flexpath_object_get_property, METH_VARARGS,
     flexpath_object_get_property_doc},
    {"delete_property", (PyCFunction)flexpath_object_delete_property, METH_VARARGS,
     flexpath_object_delete_property_doc},
    {NULL}};

static PyObject* flexpath_object_get_layers(FlexPathObject* self, void*) {
    FlexPath* flexpath = self->flexpath;
    PyObject* result = PyTuple_New(flexpath->num_elements);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        return NULL;
    }
    for (int64_t i = 0; i < flexpath->num_elements; i++) {
        PyObject* item = PyLong_FromLong(flexpath->elements[i].layer);
        if (item == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create int from layer");
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* flexpath_object_get_datatypes(FlexPathObject* self, void*) {
    FlexPath* flexpath = self->flexpath;
    PyObject* result = PyTuple_New(flexpath->num_elements);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        return NULL;
    }
    for (int64_t i = 0; i < flexpath->num_elements; i++) {
        PyObject* item = PyLong_FromLong(flexpath->elements[i].datatype);
        if (item == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create int from datatype");
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* flexpath_object_get_num_paths(FlexPathObject* self, void*) {
    return PyLong_FromLong(self->flexpath->num_elements);
}

static PyObject* flexpath_object_get_size(FlexPathObject* self, void*) {
    return PyLong_FromLong(self->flexpath->spine.point_array.size);
}

static PyGetSetDef flexpath_object_getset[] = {
    {"layers", (getter)flexpath_object_get_layers, NULL, flexpath_object_layers_doc, NULL},
    {"datatypes", (getter)flexpath_object_get_datatypes, NULL, flexpath_object_datatypes_doc, NULL},
    {"num_paths", (getter)flexpath_object_get_num_paths, NULL, flexpath_object_num_paths_doc, NULL},
    {"size", (getter)flexpath_object_get_size, NULL, flexpath_object_size_doc, NULL},
    {NULL}};
