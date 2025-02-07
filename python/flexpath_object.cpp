/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* flexpath_object_str(FlexPathObject* self) {
    char buffer[GDSTK_PRINT_BUFFER_COUNT];
    snprintf(buffer, COUNT(buffer), "FlexPath with %" PRIu64 " paths and %" PRIu64 " points",
             self->flexpath->num_elements, self->flexpath->spine.point_array.count);
    return PyUnicode_FromString(buffer);
}

static void flexpath_cleanup(FlexPathObject* self) {
    FlexPathElement* el = self->flexpath->elements;
    for (uint64_t j = self->flexpath->num_elements; j > 0; j--, el++) {
        Py_XDECREF(el->join_function_data);
        Py_XDECREF(el->end_function_data);
        Py_XDECREF(el->bend_function_data);
    }
    self->flexpath->clear();
    free_allocation(self->flexpath);
    self->flexpath = NULL;
}

static void flexpath_object_dealloc(FlexPathObject* self) {
    if (self->flexpath) flexpath_cleanup(self);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int flexpath_object_init(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_points = NULL;
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    PyObject* py_joins = NULL;
    PyObject* py_ends = NULL;
    PyObject* py_bend_radius = NULL;
    PyObject* py_bend_function = Py_None;
    PyObject* py_layer = NULL;
    PyObject* py_datatype = NULL;
    double tolerance = 1e-2;
    int simple_path = 0;
    int scale_width = 1;
    const char* keywords[] = {"points",      "width",       "offset",        "joins",
                              "ends",        "bend_radius", "bend_function", "tolerance",
                              "simple_path", "scale_width", "layer",         "datatype",
                              NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OOOOOdppOO:FlexPath", (char**)keywords,
                                     &py_points, &py_width, &py_offset, &py_joins, &py_ends,
                                     &py_bend_radius, &py_bend_function, &tolerance, &simple_path,
                                     &scale_width, &py_layer, &py_datatype))
        return -1;

    if (tolerance <= 0) {
        PyErr_SetString(PyExc_ValueError, "Tolerance must be positive.");
        return -1;
    }

    if (self->flexpath) {
        FlexPath* flexpath = self->flexpath;
        FlexPathElement* el = flexpath->elements;
        for (uint64_t i = 0; i < flexpath->num_elements; i++, el++) {
            Py_XDECREF(el->join_function_data);
            Py_XDECREF(el->end_function_data);
            Py_XDECREF(el->bend_function_data);
        }
        flexpath->clear();
    } else {
        self->flexpath = (FlexPath*)allocate_clear(sizeof(FlexPath));
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

    uint64_t num_elements = 1;
    const uint64_t count = flexpath->spine.point_array.count;
    if (count > 1) flexpath->spine.last_ctrl = flexpath->spine.point_array[count - 2];

    if (PySequence_Check(py_width)) {
        num_elements = PySequence_Length(py_width);
        flexpath->num_elements = num_elements;
        flexpath->elements =
            (FlexPathElement*)allocate_clear(num_elements * sizeof(FlexPathElement));
        if (py_offset && PySequence_Check(py_offset)) {
            if ((uint64_t)PySequence_Length(py_offset) != num_elements) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError,
                                "Sequences width and offset must have the same length.");
                return -1;
            }

            // Case 1: width and offset are sequences with the same length
            FlexPathElement* el = flexpath->elements;
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PySequence_ITEM(py_width, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRIu64 " from width sequence.", i);
                    return -1;
                }
                const double half_width = 0.5 * PyFloat_AsDouble(item);
                Py_DECREF(item);
                if (PyErr_Occurred()) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert width[%" PRIu64 "] to float.", i);
                    return -1;
                }
                if (half_width < 0) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_ValueError,
                                 "Negative width value not allowed: width[%" PRIu64 "].", i);
                    return -1;
                }

                item = PySequence_ITEM(py_offset, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRIu64 " from offset sequence.", i);
                    return -1;
                }
                const double offset = PyFloat_AsDouble(item);
                Py_DECREF(item);
                if (PyErr_Occurred()) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert offset[%" PRIu64 "] to float.", i);
                    return -1;
                }

                const Vec2 half_width_and_offset = {half_width, offset};
                el->half_width_and_offset.ensure_slots(count);
                Vec2* wo = el->half_width_and_offset.items;
                for (uint64_t j = 0; j < count; j++) *wo++ = half_width_and_offset;
                el->half_width_and_offset.count = count;
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
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PySequence_ITEM(py_width, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRIu64 " from width sequence.", i);
                    return -1;
                }
                const double half_width = 0.5 * PyFloat_AsDouble(item);
                Py_DECREF(item);
                if (PyErr_Occurred()) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert width[%" PRIu64 "] to float.", i);
                    return -1;
                }
                if (half_width < 0) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_ValueError,
                                 "Negative width value not allowed: width[%" PRIu64 "].", i);
                    return -1;
                }

                const Vec2 half_width_and_offset = {half_width,
                                                    (i - 0.5 * (num_elements - 1)) * offset};
                el->half_width_and_offset.ensure_slots(count);
                Vec2* wo = el->half_width_and_offset.items;
                for (uint64_t j = 0; j < count; j++) *wo++ = half_width_and_offset;
                el->half_width_and_offset.count = count;
            }
        }
    } else if (py_offset && PySequence_Check(py_offset)) {
        // Case 3: offset is a sequence, width a number
        num_elements = PySequence_Length(py_offset);
        flexpath->num_elements = num_elements;
        flexpath->elements =
            (FlexPathElement*)allocate_clear(num_elements * sizeof(FlexPathElement));
        const double half_width = 0.5 * PyFloat_AsDouble(py_width);
        if (PyErr_Occurred()) {
            flexpath_cleanup(self);
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert width to float.");
            return -1;
        }
        if (half_width < 0) {
            flexpath_cleanup(self);
            PyErr_SetString(PyExc_ValueError, "Negative width value not allowed.");
            return -1;
        }

        FlexPathElement* el = flexpath->elements;
        for (uint64_t i = 0; i < num_elements; i++, el++) {
            PyObject* item = PySequence_ITEM(py_offset, i);
            if (item == NULL) {
                flexpath_cleanup(self);
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to retrieve item %" PRIu64 " from offset sequence.", i);
                return -1;
            }
            const double offset = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                flexpath_cleanup(self);
                PyErr_Format(PyExc_RuntimeError, "Unable to convert offset[%" PRIu64 "] to float.",
                             i);
                return -1;
            }

            const Vec2 half_width_and_offset = {half_width, offset};
            el->half_width_and_offset.ensure_slots(count);
            Vec2* wo = el->half_width_and_offset.items;
            for (uint64_t j = 0; j < count; j++) *wo++ = half_width_and_offset;
            el->half_width_and_offset.count = count;
        }
    } else {
        // Case 4: width and offset are numbers
        flexpath->num_elements = 1;
        flexpath->elements = (FlexPathElement*)allocate_clear(sizeof(FlexPathElement));
        FlexPathElement* el = flexpath->elements;
        const double half_width = 0.5 * PyFloat_AsDouble(py_width);
        if (PyErr_Occurred()) {
            flexpath_cleanup(self);
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert width to float.");
            return -1;
        }
        if (half_width < 0) {
            flexpath_cleanup(self);
            PyErr_SetString(PyExc_ValueError, "Negative width value not allowed.");
            return -1;
        }
        const double offset = py_offset == NULL ? 0 : PyFloat_AsDouble(py_offset);
        if (PyErr_Occurred()) {
            flexpath_cleanup(self);
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert offset to float.");
            return -1;
        }

        const Vec2 half_width_and_offset = {half_width, offset};
        el->half_width_and_offset.ensure_slots(count);
        Vec2* wo = el->half_width_and_offset.items;
        for (uint64_t j = 0; j < count; j++) *wo++ = half_width_and_offset;
        el->half_width_and_offset.count = count;
    }

    if (py_layer) {
        if (PyList_Check(py_layer)) {
            if ((uint64_t)PyList_GET_SIZE(py_layer) != num_elements) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError,
                                "List layer must have the same length as the number of paths.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_layer, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to get item %" PRIu64 " from layer list.", i);
                    return -1;
                }
                set_layer(el->tag, (uint32_t)PyLong_AsUnsignedLongLong(item));
                if (PyErr_Occurred()) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError, "Unable to convert layer[%" PRIu64 "] to int.",
                                 i);
                    return -1;
                }
            }
        } else {
            const uint32_t layer = (uint32_t)PyLong_AsUnsignedLongLong(py_layer);
            if (PyErr_Occurred()) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert layer to int.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (uint64_t i = 0; i < num_elements; i++) set_layer((el++)->tag, layer);
        }
    }

    if (py_datatype) {
        if (PyList_Check(py_datatype)) {
            if ((uint64_t)PyList_GET_SIZE(py_datatype) != num_elements) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError,
                                "List datatype must have the same length as the number of paths.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_datatype, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to get item %" PRIu64 " from datatype list.", i);
                    return -1;
                }
                set_type(el->tag, (uint32_t)PyLong_AsUnsignedLongLong(item));
                if (PyErr_Occurred()) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert datatype[%" PRIu64 "] to int.", i);
                    return -1;
                }
            }
        } else {
            const uint32_t datatype = (uint32_t)PyLong_AsUnsignedLongLong(py_datatype);
            if (PyErr_Occurred()) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert datatype to int.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (uint64_t i = 0; i < num_elements; i++) set_type((el++)->tag, datatype);
        }
    }

    if (py_joins) {
        if (PyList_Check(py_joins)) {
            if ((uint64_t)PyList_GET_SIZE(py_joins) != num_elements) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError,
                                "List joins must have the same length as the number of paths.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_joins, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRIu64 " from joins list.", i);
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
                    if (PyUnicode_CompareWithASCIIString(item, "miter") == 0) {
                        jt = JoinType::Miter;
                    } else if (PyUnicode_CompareWithASCIIString(item, "bevel") == 0) {
                        jt = JoinType::Bevel;
                    } else if (PyUnicode_CompareWithASCIIString(item, "round") == 0) {
                        jt = JoinType::Round;
                    } else if (PyUnicode_CompareWithASCIIString(item, "smooth") == 0) {
                        jt = JoinType::Smooth;
                    } else if (PyUnicode_CompareWithASCIIString(item, "natural") != 0) {
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
            for (uint64_t i = 0; i < num_elements; i++, el++) {
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
            for (uint64_t i = 0; i < num_elements; i++) (el++)->join_type = jt;
        }
    }

    if (py_ends) {
        if (PyList_Check(py_ends)) {
            if ((uint64_t)PyList_GET_SIZE(py_ends) != num_elements) {
                flexpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError,
                                "List ends must have the same length as the number of paths.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_ends, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRIu64 " from ends list.", i);
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
                            et = EndType::HalfWidth;
                        } else if (PyUnicode_CompareWithASCIIString(item, "round") == 0) {
                            et = EndType::Round;
                        } else if (PyUnicode_CompareWithASCIIString(item, "smooth") == 0) {
                            et = EndType::Smooth;
                        } else if (PyUnicode_CompareWithASCIIString(item, "flush") != 0) {
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
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                el->end_type = EndType::Function;
                el->end_function = (EndFunction)custom_end_function;
                el->end_function_data = (void*)py_ends;
                Py_INCREF(py_ends);
            }
        } else {
            EndType et = EndType::Flush;
            Vec2 ex = Vec2{0, 0};
            if (PyUnicode_Check(py_ends)) {
                if (PyUnicode_CompareWithASCIIString(py_ends, "extended") == 0)
                    et = EndType::HalfWidth;
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
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                el->end_type = et;
                el->end_extensions = ex;
            }
        }
    }

    if (py_bend_radius) {
        if (PyList_Check(py_bend_radius)) {
            if ((uint64_t)PyList_GET_SIZE(py_bend_radius) != num_elements) {
                flexpath_cleanup(self);
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Sequence bend_radius must have the same length as the number of paths.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_bend_radius, i);
                if (item == NULL) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRIu64 " from bend_radius sequence.",
                                 i);
                    return -1;
                }
                const double bend_radius = PyFloat_AsDouble(item);
                if (PyErr_Occurred()) {
                    flexpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert bend_radius[%" PRIu64 "] to float.", i);
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
                for (uint64_t i = 0; i < num_elements; i++, el++) {
                    el->bend_type = BendType::Circular;
                    el->bend_radius = bend_radius;
                }
            }
        }
    }

    if (py_bend_function != Py_None) {
        if (PyList_Check(py_bend_function)) {
            if ((uint64_t)PyList_GET_SIZE(py_bend_function) != num_elements) {
                flexpath_cleanup(self);
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Sequence bend_function must have the same length as the number of paths.");
                return -1;
            }
            FlexPathElement* el = flexpath->elements;
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_bend_function, i);
                if (item == NULL || !PyCallable_Check(item)) {
                    flexpath_cleanup(self);
                    PyErr_Format(
                        PyExc_RuntimeError,
                        "Unable to get callable from item %" PRIu64 " from bend_function list.", i);
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
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                el->bend_type = BendType::Function;
                el->bend_function = (BendFunction)custom_bend_function;
                el->bend_function_data = (void*)py_bend_function;
                Py_INCREF(py_bend_function);
            }
        }
    }

    flexpath->spine.tolerance = tolerance;
    flexpath->simple_path = simple_path > 0;
    flexpath->scale_width = scale_width > 0;
    flexpath->owner = self;
    return 0;
}

static PyObject* flexpath_object_copy(FlexPathObject* self, PyObject*) {
    FlexPathObject* result = PyObject_New(FlexPathObject, &flexpath_object_type);
    result = (FlexPathObject*)PyObject_Init((PyObject*)result, &flexpath_object_type);
    result->flexpath = (FlexPath*)allocate_clear(sizeof(FlexPath));
    result->flexpath->copy_from(*self->flexpath);
    result->flexpath->owner = result;
    return (PyObject*)result;
}

static PyObject* flexpath_object_deepcopy(FlexPathObject* self, PyObject* arg) {
    return flexpath_object_copy(self, NULL);
}

static PyObject* flexpath_object_spine(FlexPathObject* self, PyObject*) {
    const Array<Vec2>* point_array = &self->flexpath->spine.point_array;
    npy_intp dims[] = {(npy_intp)point_array->count, 2};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    memcpy(data, point_array->items, sizeof(double) * point_array->count * 2);
    return (PyObject*)result;
}

static PyObject* flexpath_object_path_spines(FlexPathObject* self, PyObject*) {
    Array<Vec2> point_array = {};
    FlexPath* path = self->flexpath;
    PyObject* result = PyList_New(path->num_elements);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        return NULL;
    }
    FlexPathElement* el = path->elements;
    for (uint64_t i = 0; i < path->num_elements; i++) {
        path->element_center(el++, point_array);
        npy_intp dims[] = {(npy_intp)point_array.count, 2};
        PyObject* spine = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (!spine) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create return array.");
            Py_DECREF(result);
            point_array.clear();
            return NULL;
        }
        PyList_SET_ITEM(result, i, spine);
        double* data = (double*)PyArray_DATA((PyArrayObject*)spine);
        memcpy(data, point_array.items, sizeof(double) * point_array.count * 2);
        point_array.count = 0;
    }
    point_array.clear();
    return (PyObject*)result;
}

static PyObject* flexpath_object_widths(FlexPathObject* self, PyObject*) {
    const FlexPath* flexpath = self->flexpath;
    npy_intp dims[] = {(npy_intp)flexpath->spine.point_array.count,
                       (npy_intp)flexpath->num_elements};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    double* d = data;
    for (uint64_t j = 0; j < flexpath->spine.point_array.count; j++) {
        const FlexPathElement* el = flexpath->elements;
        for (uint64_t i = 0; i < flexpath->num_elements; i++)
            *d++ = 2 * (el++)->half_width_and_offset[j].u;
    }
    return (PyObject*)result;
}

static PyObject* flexpath_object_offsets(FlexPathObject* self, PyObject*) {
    const FlexPath* flexpath = self->flexpath;
    npy_intp dims[] = {(npy_intp)flexpath->spine.point_array.count,
                       (npy_intp)flexpath->num_elements};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    double* d = data;
    for (uint64_t j = 0; j < flexpath->spine.point_array.count; j++) {
        const FlexPathElement* el = flexpath->elements;
        for (uint64_t i = 0; i < flexpath->num_elements; i++)
            *d++ = (el++)->half_width_and_offset[j].v;
    }
    return (PyObject*)result;
}

static PyObject* flexpath_object_to_polygons(FlexPathObject* self, PyObject*) {
    Array<Polygon*> array = {};
    if (return_error(self->flexpath->to_polygons(false, 0, array))) {
        for (uint64_t i = 0; i < array.count; i++) {
            array[i]->clear();
            free_allocation(array[i]);
        }
        array.clear();
        return NULL;
    }
    PyObject* result = PyList_New(array.count);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return array.");
        for (uint64_t i = 0; i < array.count; i++) {
            array[i]->clear();
            free_allocation(array[i]);
        }
        array.clear();
        return NULL;
    }
    for (uint64_t i = 0; i < array.count; i++) {
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
    uint64_t len = PySequence_Length(arg);
    FlexPath* flexpath = self->flexpath;
    if (len != flexpath->num_elements) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Length of layer sequence must match the number of paths.");
        return NULL;
    }
    for (uint64_t i = 0; i < len; i++) {
        PyObject* item = PySequence_ITEM(arg, i);
        if (item == NULL) {
            PyErr_Format(PyExc_RuntimeError, "Unable to get item %" PRIu64 " from sequence.", i);
            return NULL;
        }
        set_layer(flexpath->elements[i].tag, (uint32_t)PyLong_AsUnsignedLongLong(item));
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_Format(PyExc_RuntimeError, "Unable to convert sequence item %" PRIu64 " to int.",
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
    uint64_t len = PySequence_Length(arg);
    FlexPath* flexpath = self->flexpath;
    if (len != flexpath->num_elements) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Length of datatype sequence must match the number of paths.");
        return NULL;
    }
    for (uint64_t i = 0; i < len; i++) {
        PyObject* item = PySequence_ITEM(arg, i);
        if (item == NULL) {
            PyErr_Format(PyExc_RuntimeError, "Unable to get item %" PRIu64 " from sequence.", i);
            return NULL;
        }
        set_type(flexpath->elements[i].tag, (uint32_t)PyLong_AsUnsignedLongLong(item));
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_Format(PyExc_TypeError, "Unable to convert sequence item %" PRIu64 " to int.", i);
            return NULL;
        }
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_set_joins(FlexPathObject* self, PyObject* arg) {
    if (!PySequence_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a sequence of join types.");
        return NULL;
    }
    uint64_t len = PySequence_Length(arg);
    FlexPath* flexpath = self->flexpath;
    if (len != flexpath->num_elements) {
        PyErr_SetString(PyExc_RuntimeError, "Length of sequence must match the number of paths.");
        return NULL;
    }
    for (uint64_t i = 0; i < len; i++) {
        FlexPathElement* el = flexpath->elements + i;
        if (el->join_type == JoinType::Function) {
            el->join_type = JoinType::Natural;
            el->join_function = NULL;
            Py_DECREF(el->join_function_data);
            el->join_function_data = NULL;
        }
        PyObject* item = PySequence_ITEM(arg, i);
        if (item == NULL) {
            PyErr_Format(PyExc_RuntimeError, "Unable to get item %" PRIu64 " from sequence.", i);
            return NULL;
        }
        if (PyCallable_Check(item)) {
            el->join_type = JoinType::Function;
            el->join_function = (JoinFunction)custom_join_function;
            el->join_function_data = (void*)item;
        } else {
            if (!PyUnicode_Check(item)) {
                Py_DECREF(item);
                PyErr_SetString(
                    PyExc_TypeError,
                    "Joins must be one of 'natural', 'miter', 'bevel', 'round', 'smooth', or a callable.");
                return NULL;
            }
            JoinType jt = JoinType::Natural;
            if (PyUnicode_CompareWithASCIIString(item, "miter") == 0) {
                jt = JoinType::Miter;
            } else if (PyUnicode_CompareWithASCIIString(item, "bevel") == 0) {
                jt = JoinType::Bevel;
            } else if (PyUnicode_CompareWithASCIIString(item, "round") == 0) {
                jt = JoinType::Round;
            } else if (PyUnicode_CompareWithASCIIString(item, "smooth") == 0) {
                jt = JoinType::Smooth;
            } else if (PyUnicode_CompareWithASCIIString(item, "natural") != 0) {
                flexpath_cleanup(self);
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Joins must be one of 'natural', 'miter', 'bevel', 'round', 'smooth', a callable, or a list of those.");
                return NULL;
            }
            el->join_type = jt;
            Py_DECREF(item);
        }
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_set_ends(FlexPathObject* self, PyObject* arg) {
    if (!PySequence_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a sequence of end types.");
        return NULL;
    }
    uint64_t len = PySequence_Length(arg);
    FlexPath* flexpath = self->flexpath;
    if (len != flexpath->num_elements) {
        PyErr_SetString(PyExc_RuntimeError, "Length of sequence must match the number of paths.");
        return NULL;
    }
    for (uint64_t i = 0; i < len; i++) {
        FlexPathElement* el = flexpath->elements + i;
        if (el->end_type == EndType::Function) {
            el->end_type = EndType::Flush;
            el->end_function = NULL;
            Py_DECREF(el->end_function_data);
            el->end_function_data = NULL;
        }
        PyObject* item = PySequence_ITEM(arg, i);
        if (item == NULL) {
            PyErr_Format(PyExc_RuntimeError, "Unable to get item %" PRIu64 " from sequence.", i);
            return NULL;
        }
        if (PyCallable_Check(item)) {
            el->end_type = EndType::Function;
            el->end_function = (EndFunction)custom_end_function;
            el->end_function_data = (void*)item;
        } else {
            EndType et = EndType::Flush;
            if (PyUnicode_Check(item)) {
                if (PyUnicode_CompareWithASCIIString(item, "extended") == 0) {
                    et = EndType::HalfWidth;
                } else if (PyUnicode_CompareWithASCIIString(item, "round") == 0) {
                    et = EndType::Round;
                } else if (PyUnicode_CompareWithASCIIString(item, "smooth") == 0) {
                    et = EndType::Smooth;
                } else if (PyUnicode_CompareWithASCIIString(item, "flush") != 0) {
                    Py_DECREF(item);
                    PyErr_SetString(
                        PyExc_RuntimeError,
                        "Ends must be one of 'flush', 'extended', 'round', 'smooth', a 2-tuple, or a callable.");
                    return NULL;
                }
            } else {
                et = EndType::Extended;
                if (!PyTuple_Check(item) || PyArg_ParseTuple(item, "dd", &el->end_extensions.u,
                                                             &el->end_extensions.v) < 0) {
                    Py_DECREF(item);
                    PyErr_SetString(
                        PyExc_RuntimeError,
                        "Ends must be one of 'flush', 'extended', 'round', 'smooth', a 2-tuple, or a callable.");
                    return NULL;
                }
            }
            el->end_type = et;
            Py_DECREF(item);
        }
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_set_bend_radius(FlexPathObject* self, PyObject* arg) {
    if (!PySequence_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a sequence of radii.");
        return NULL;
    }
    uint64_t len = PySequence_Length(arg);
    FlexPath* flexpath = self->flexpath;
    if (len != flexpath->num_elements) {
        PyErr_SetString(PyExc_RuntimeError, "Length of sequence must match the number of paths.");
        return NULL;
    }
    for (uint64_t i = 0; i < len; i++) {
        FlexPathElement* el = flexpath->elements + i;
        PyObject* item = PySequence_ITEM(arg, i);
        if (item == NULL) {
            PyErr_Format(PyExc_RuntimeError, "Unable to get item %" PRIu64 " from sequence.", i);
            return NULL;
        }
        const double bend_radius = item == Py_None ? 0 : PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_Format(PyExc_RuntimeError,
                         "Unable to convert item %" PRIu64 " to a callable or float.", i);
            return NULL;
        }
        if (bend_radius > 0) {
            if (el->bend_type == BendType::None) {
                el->bend_type = BendType::Circular;
            }
            el->bend_radius = bend_radius;
        } else if (el->bend_type == BendType::Circular) {
            el->bend_type = BendType::None;
            el->bend_radius = 0;
        }
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_set_bend_function(FlexPathObject* self, PyObject* arg) {
    if (!PySequence_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a sequence of callables or None.");
        return NULL;
    }
    uint64_t len = PySequence_Length(arg);
    FlexPath* flexpath = self->flexpath;
    if (len != flexpath->num_elements) {
        PyErr_SetString(PyExc_RuntimeError, "Length of sequence must match the number of paths.");
        return NULL;
    }
    for (uint64_t i = 0; i < len; i++) {
        FlexPathElement* el = flexpath->elements + i;
        if (el->bend_type == BendType::Function) {
            el->bend_type = el->bend_radius > 0 ? BendType::Circular : BendType::None;
            el->bend_function = NULL;
            Py_DECREF(el->bend_function_data);
            el->bend_function_data = NULL;
        }
        PyObject* item = PySequence_ITEM(arg, i);
        if (item == NULL) {
            PyErr_Format(PyExc_RuntimeError, "Unable to get item %" PRIu64 " from sequence.", i);
            return NULL;
        }
        if (PyCallable_Check(item)) {
            el->bend_type = BendType::Function;
            el->bend_function = (BendFunction)custom_bend_function;
            el->bend_function_data = (void*)item;
        } else {
            Py_DECREF(item);
        }
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

// If offset is a single number, it's the new distance between paths (analogous to what is used in
// init).
static int parse_flexpath_offset(const FlexPath flexpath, PyObject* py_offset, double* offset) {
    if (PySequence_Check(py_offset)) {
        if ((uint64_t)PySequence_Length(py_offset) < flexpath.num_elements) {
            PyErr_SetString(PyExc_RuntimeError, "Sequence offset doesn't have enough elements.");
            return -1;
        }
        for (uint64_t i = 0; i < flexpath.num_elements; i++) {
            PyObject* item = PySequence_ITEM(py_offset, i);
            if (item == NULL) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get item %" PRIu64 " from sequence offset.", i);
                return -1;
            }
            *offset++ = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to convert item %" PRIu64 " from sequence offset to float.",
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
        for (uint64_t i = 0; i < flexpath.num_elements; i++)
            *offset++ = (i - 0.5 * (flexpath.num_elements - 1)) * value;
    }
    return 0;
}

static int parse_flexpath_width(const FlexPath flexpath, PyObject* py_width, double* width) {
    if (PySequence_Check(py_width)) {
        if ((uint64_t)PySequence_Length(py_width) < flexpath.num_elements) {
            PyErr_SetString(PyExc_RuntimeError, "Sequence width doesn't have enough elements.");
            return -1;
        }
        for (uint64_t i = 0; i < flexpath.num_elements; i++) {
            PyObject* item = PySequence_ITEM(py_width, i);
            if (item == NULL) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get item %" PRIu64 " from sequence width.", i);
                return -1;
            }
            const double value = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to convert item %" PRIu64 " from sequence width to float.", i);
                return -1;
            }
            if (value < 0) {
                PyErr_Format(PyExc_ValueError,
                             "Negative width value not allowed: width[%" PRIu64 "].", i);
                return -1;
            }
            *width++ = value;
        }
    } else {
        const double value = PyFloat_AsDouble(py_width);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert width to float.");
            return -1;
        }
        if (value < 0) {
            PyErr_SetString(PyExc_ValueError, "Negative width value not allowed.");
            return -1;
        }
        for (uint64_t i = 0; i < flexpath.num_elements; i++) *width++ = value;
    }
    return 0;
}

static PyObject* flexpath_object_horizontal(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_coord;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 0;
    const char* keywords[] = {"x", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:horizontal", (char**)keywords, &py_coord,
                                     &py_width, &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    double* buffer = (double*)allocate(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            free_allocation(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            free_allocation(buffer);
            return NULL;
        }
    }
    if (PySequence_Check(py_coord)) {
        Array<double> coord = {};
        if (parse_double_sequence(py_coord, coord, "x") < 0) {
            free_allocation(buffer);
            return NULL;
        }
        flexpath->horizontal(coord, width, offset, relative > 0);
        coord.clear();
    } else {
        double single = PyFloat_AsDouble(py_coord);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert coordinate to float.");
            free_allocation(buffer);
            return NULL;
        }
        flexpath->horizontal(single, width, offset, relative > 0);
    }
    free_allocation(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_vertical(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_coord;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 0;
    const char* keywords[] = {"y", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:vertical", (char**)keywords, &py_coord,
                                     &py_width, &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    double* buffer = (double*)allocate(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            free_allocation(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            free_allocation(buffer);
            return NULL;
        }
    }
    if (PySequence_Check(py_coord)) {
        Array<double> coord = {};
        if (parse_double_sequence(py_coord, coord, "y") < 0) {
            free_allocation(buffer);
            return NULL;
        }
        flexpath->vertical(coord, width, offset, relative > 0);
        coord.clear();
    } else {
        double single = PyFloat_AsDouble(py_coord);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert coordinate to float.");
            free_allocation(buffer);
            return NULL;
        }
        flexpath->vertical(single, width, offset, relative > 0);
    }
    free_allocation(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_segment(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:segment", (char**)keywords, &xy, &py_width,
                                     &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    Array<Vec2> point_array = {};
    point_array.ensure_slots(1);
    if (parse_point(xy, *point_array.items, "xy") == 0)
        point_array.count = 1;
    else {
        PyErr_Clear();
        if (parse_point_sequence(xy, point_array, "xy") < 0) {
            point_array.clear();
            return NULL;
        }
    }
    double* buffer = (double*)allocate(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            point_array.clear();
            free_allocation(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            point_array.clear();
            free_allocation(buffer);
            return NULL;
        }
    }
    flexpath->segment(point_array, width, offset, relative > 0);
    point_array.clear();
    free_allocation(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_cubic(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:cubic", (char**)keywords, &xy, &py_width,
                                     &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    Array<Vec2> point_array = {};
    if (parse_point_sequence(xy, point_array, "xy") < 0 || point_array.count < 3) {
        point_array.clear();
        PyErr_SetString(PyExc_RuntimeError,
                        "Argument xy must be a sequence of at least 3 coordinates.");
        return NULL;
    }
    double* buffer = (double*)allocate(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            point_array.clear();
            free_allocation(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            point_array.clear();
            free_allocation(buffer);
            return NULL;
        }
    }
    flexpath->cubic(point_array, width, offset, relative > 0);
    point_array.clear();
    free_allocation(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_cubic_smooth(FlexPathObject* self, PyObject* args,
                                              PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:cubic_smooth", (char**)keywords, &xy,
                                     &py_width, &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    Array<Vec2> point_array = {};
    if (parse_point_sequence(xy, point_array, "xy") < 0 || point_array.count < 2) {
        point_array.clear();
        PyErr_SetString(PyExc_RuntimeError,
                        "Argument xy must be a sequence of at least 2 coordinates.");
        return NULL;
    }
    double* buffer = (double*)allocate(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            point_array.clear();
            free_allocation(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            point_array.clear();
            free_allocation(buffer);
            return NULL;
        }
    }
    flexpath->cubic_smooth(point_array, width, offset, relative > 0);
    point_array.clear();
    free_allocation(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_quadratic(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:quadratic", (char**)keywords, &xy,
                                     &py_width, &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    Array<Vec2> point_array = {};
    point_array.ensure_slots(1);
    if (parse_point_sequence(xy, point_array, "xy") < 0 || point_array.count < 2) {
        point_array.clear();
        PyErr_SetString(PyExc_RuntimeError,
                        "Argument xy must be a sequence of at least 2 coordinates.");
        return NULL;
    }
    double* buffer = (double*)allocate(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            point_array.clear();
            free_allocation(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            point_array.clear();
            free_allocation(buffer);
            return NULL;
        }
    }
    flexpath->quadratic(point_array, width, offset, relative > 0);
    point_array.clear();
    free_allocation(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_quadratic_smooth(FlexPathObject* self, PyObject* args,
                                                  PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:quadratic_smooth", (char**)keywords, &xy,
                                     &py_width, &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    Array<Vec2> point_array = {};
    point_array.ensure_slots(1);
    if (parse_point(xy, *point_array.items, "xy") == 0)
        point_array.count = 1;
    else {
        PyErr_Clear();
        if (parse_point_sequence(xy, point_array, "xy") < 0) {
            point_array.clear();
            return NULL;
        }
    }
    double* buffer = (double*)allocate(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            point_array.clear();
            free_allocation(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            point_array.clear();
            free_allocation(buffer);
            return NULL;
        }
    }
    flexpath->quadratic_smooth(point_array, width, offset, relative > 0);
    point_array.clear();
    free_allocation(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_bezier(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:bezier", (char**)keywords, &xy, &py_width,
                                     &py_offset, &relative))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    Array<Vec2> point_array = {};
    point_array.ensure_slots(1);
    if (parse_point(xy, *point_array.items, "xy") == 0)
        point_array.count = 1;
    else {
        PyErr_Clear();
        if (parse_point_sequence(xy, point_array, "xy") < 0) {
            point_array.clear();
            return NULL;
        }
    }
    double* buffer = (double*)allocate(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            point_array.clear();
            free_allocation(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            point_array.clear();
            free_allocation(buffer);
            return NULL;
        }
    }
    flexpath->bezier(point_array, width, offset, relative > 0);
    point_array.clear();
    free_allocation(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_intepolation(FlexPathObject* self, PyObject* args,
                                              PyObject* kwds) {
    PyObject* py_points = NULL;
    PyObject* py_angles = Py_None;
    PyObject* py_tension_in = NULL;
    PyObject* py_tension_out = NULL;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
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
    Array<Vec2> point_array = {};
    if (parse_point_sequence(py_points, point_array, "points") < 0) {
        point_array.clear();
        return NULL;
    }
    const uint64_t count = point_array.count;

    tension = (Vec2*)allocate((sizeof(Vec2) + sizeof(double) + sizeof(bool)) * (count + 1));
    angles = (double*)(tension + (count + 1));
    angle_constraints = (bool*)(angles + (count + 1));

    if (py_angles == Py_None) {
        memset(angle_constraints, 0, sizeof(bool) * (count + 1));
    } else {
        if ((uint64_t)PySequence_Length(py_angles) != count + 1) {
            free_allocation(tension);
            point_array.clear();
            PyErr_SetString(
                PyExc_TypeError,
                "Argument angles must be None or a sequence with count len(points) + 1.");
            return NULL;
        }
        for (uint64_t i = 0; i < count + 1; i++) {
            PyObject* item = PySequence_ITEM(py_angles, i);
            if (!item) {
                free_allocation(tension);
                point_array.clear();
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get item %" PRIu64 " from angles sequence.", i);
                return NULL;
            }
            if (item == Py_None)
                angle_constraints[i] = false;
            else {
                angle_constraints[i] = true;
                angles[i] = PyFloat_AsDouble(item);
                if (PyErr_Occurred()) {
                    free_allocation(tension);
                    point_array.clear();
                    Py_DECREF(item);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert angle[%" PRIu64 "] to float.", i);
                    return NULL;
                }
            }
            Py_DECREF(item);
        }
    }

    if (!py_tension_in) {
        Vec2* t = tension;
        for (uint64_t i = 0; i < count + 1; i++) (t++)->u = 1;
    } else if (!PySequence_Check(py_tension_in)) {
        double t_in = PyFloat_AsDouble(py_tension_in);
        if (PyErr_Occurred()) {
            free_allocation(tension);
            point_array.clear();
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert tension_in to float.");
            return NULL;
        }
        Vec2* t = tension;
        for (uint64_t i = 0; i < count + 1; i++) (t++)->u = t_in;
    } else {
        if ((uint64_t)PySequence_Length(py_tension_in) != count + 1) {
            free_allocation(tension);
            point_array.clear();
            PyErr_SetString(
                PyExc_TypeError,
                "Argument tension_in must be a number or a sequence with count len(points) + 1.");
            return NULL;
        }
        for (uint64_t i = 0; i < count + 1; i++) {
            PyObject* item = PySequence_ITEM(py_tension_in, i);
            if (!item) {
                free_allocation(tension);
                point_array.clear();
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get item %" PRIu64 " from tension_in sequence.", i);
                return NULL;
            }
            tension[i].u = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                free_allocation(tension);
                point_array.clear();
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to convert tension_in[%" PRIu64 "] to float.", i);
                return NULL;
            }
        }
    }

    if (!py_tension_out) {
        Vec2* t = tension;
        for (uint64_t i = 0; i < count + 1; i++) (t++)->v = 1;
    } else if (!PySequence_Check(py_tension_out)) {
        double t_out = PyFloat_AsDouble(py_tension_out);
        if (PyErr_Occurred()) {
            free_allocation(tension);
            point_array.clear();
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert tension_out to float.");
            return NULL;
        }
        Vec2* t = tension;
        for (uint64_t i = 0; i < count + 1; i++) (t++)->v = t_out;
    } else {
        if ((uint64_t)PySequence_Length(py_tension_out) != count + 1) {
            free_allocation(tension);
            point_array.clear();
            PyErr_SetString(
                PyExc_TypeError,
                "Argument tension_out must be a number or a sequence with count len(points) + 1.");
            return NULL;
        }
        for (uint64_t i = 0; i < count + 1; i++) {
            PyObject* item = PySequence_ITEM(py_tension_out, i);
            if (!item) {
                free_allocation(tension);
                point_array.clear();
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get item %" PRIu64 " from tension_out sequence.", i);
                return NULL;
            }
            tension[i].v = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                free_allocation(tension);
                point_array.clear();
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to convert tension_out[%" PRIu64 "] to float.", i);
                return NULL;
            }
        }
    }

    double* buffer = (double*)allocate(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            free_allocation(tension);
            point_array.clear();
            free_allocation(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            free_allocation(tension);
            point_array.clear();
            free_allocation(buffer);
            return NULL;
        }
    }

    flexpath->interpolation(point_array, angles, angle_constraints, tension, initial_curl,
                            final_curl, cycle > 0, width, offset, relative > 0);

    point_array.clear();
    free_allocation(tension);
    free_allocation(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_arc(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_radius;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
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
    double* buffer = (double*)allocate(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            free_allocation(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            free_allocation(buffer);
            return NULL;
        }
    }

    if (radius_x <= 0 || radius_y <= 0) {
        PyErr_SetString(PyExc_ValueError, "Arc radius must be positive.");
        free_allocation(buffer);
        return NULL;
    }

    flexpath->arc(radius_x, radius_y, initial_angle, final_angle, rotation, width, offset);
    free_allocation(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_turn(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    double radius;
    double angle;
    const char* keywords[] = {"radius", "angle", "width", "offset", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dd|OO:turn", (char**)keywords, &radius, &angle,
                                     &py_width, &py_offset))
        return NULL;
    FlexPath* flexpath = self->flexpath;
    double* buffer = (double*)allocate(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            free_allocation(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            free_allocation(buffer);
            return NULL;
        }
    }
    if (radius <= 0) {
        PyErr_SetString(PyExc_ValueError, "Turn radius must be positive.");
        free_allocation(buffer);
        return NULL;
    }
    flexpath->turn(radius, angle, width, offset);
    free_allocation(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_parametric(FlexPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_function;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
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
    double* buffer = (double*)allocate(sizeof(double) * flexpath->num_elements * 2);
    double* width = NULL;
    if (py_width != Py_None) {
        width = buffer;
        if (parse_flexpath_width(*flexpath, py_width, width) < 0) {
            free_allocation(buffer);
            return NULL;
        }
    }
    double* offset = NULL;
    if (py_offset != Py_None) {
        offset = buffer + flexpath->num_elements;
        if (parse_flexpath_offset(*flexpath, py_offset, offset) < 0) {
            free_allocation(buffer);
            return NULL;
        }
    }
    Py_INCREF(py_function);
    flexpath->parametric((ParametricVec2)eval_parametric_vec2, (void*)py_function, width, offset,
                         relative > 0);
    Py_DECREF(py_function);
    free_allocation(buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_commands(FlexPathObject* self, PyObject* args) {
    uint64_t count = PyTuple_GET_SIZE(args);
    CurveInstruction* instructions =
        (CurveInstruction*)allocate_clear(sizeof(CurveInstruction) * count * 2);
    CurveInstruction* instr = instructions;

    for (uint64_t i = 0; i < count; i++) {
        PyObject* item = PyTuple_GET_ITEM(args, i);
        if (PyUnicode_Check(item)) {
            Py_ssize_t len = 0;
            const char* command = PyUnicode_AsUTF8AndSize(item, &len);
            if (len != 1) {
                PyErr_SetString(PyExc_RuntimeError,
                                "Path instructions must be single characters or numbers.");
                free_allocation(instructions);
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
                free_allocation(instructions);
                return NULL;
            }
        }
    }

    uint64_t instr_size = instr - instructions;
    uint64_t processed = self->flexpath->commands(instructions, instr_size);
    if (processed < instr_size) {
        PyErr_Format(PyExc_RuntimeError,
                     "Error parsing argument %" PRIu64 " in curve construction.", processed);
        free_allocation(instructions);
        return NULL;
    }

    free_allocation(instructions);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_translate(FlexPathObject* self, PyObject* args) {
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

static PyObject* flexpath_object_apply_repetition(FlexPathObject* self, PyObject*) {
    Array<FlexPath*> array = {};
    self->flexpath->apply_repetition(array);
    PyObject* result = PyList_New(array.count);
    for (uint64_t i = 0; i < array.count; i++) {
        FlexPathObject* obj = PyObject_New(FlexPathObject, &flexpath_object_type);
        obj = (FlexPathObject*)PyObject_Init((PyObject*)obj, &flexpath_object_type);
        obj->flexpath = array[i];
        array[i]->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }
    array.clear();
    return result;
}

static PyObject* flexpath_object_set_property(FlexPathObject* self, PyObject* args) {
    if (!parse_property(self->flexpath->properties, args)) return NULL;
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_get_property(FlexPathObject* self, PyObject* args) {
    return build_property(self->flexpath->properties, args);
}

static PyObject* flexpath_object_delete_property(FlexPathObject* self, PyObject* args) {
    char* name;
    if (!PyArg_ParseTuple(args, "s:delete_property", &name)) return NULL;
    remove_property(self->flexpath->properties, name, false);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_set_gds_property(FlexPathObject* self, PyObject* args) {
    uint16_t attribute;
    char* value;
    Py_ssize_t count;
    if (!PyArg_ParseTuple(args, "Hs#:set_gds_property", &attribute, &value, &count)) return NULL;
    if (count >= 0) set_gds_property(self->flexpath->properties, attribute, value, (uint64_t)count);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* flexpath_object_get_gds_property(FlexPathObject* self, PyObject* args) {
    uint16_t attribute;
    if (!PyArg_ParseTuple(args, "H:get_gds_property", &attribute)) return NULL;
    const PropertyValue* value = get_gds_property(self->flexpath->properties, attribute);
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

static PyObject* flexpath_object_delete_gds_property(FlexPathObject* self, PyObject* args) {
    uint16_t attribute;
    if (!PyArg_ParseTuple(args, "H:delete_gds_property", &attribute)) return NULL;
    remove_gds_property(self->flexpath->properties, attribute);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyMethodDef flexpath_object_methods[] = {
    {"copy", (PyCFunction)flexpath_object_copy, METH_NOARGS, flexpath_object_copy_doc},
    {"__deepcopy__", (PyCFunction)flexpath_object_deepcopy, METH_VARARGS | METH_KEYWORDS,
     flexpath_object_deepcopy_doc},
    {"spine", (PyCFunction)flexpath_object_spine, METH_NOARGS, flexpath_object_spine_doc},
    {"path_spines", (PyCFunction)flexpath_object_path_spines, METH_NOARGS,
     flexpath_object_path_spines_doc},
    {"widths", (PyCFunction)flexpath_object_widths, METH_NOARGS, flexpath_object_widths_doc},
    {"offsets", (PyCFunction)flexpath_object_offsets, METH_NOARGS, flexpath_object_offsets_doc},
    {"to_polygons", (PyCFunction)flexpath_object_to_polygons, METH_NOARGS,
     flexpath_object_to_polygons_doc},
    {"set_layers", (PyCFunction)flexpath_object_set_layers, METH_VARARGS,
     flexpath_object_set_layers_doc},
    {"set_datatypes", (PyCFunction)flexpath_object_set_datatypes, METH_VARARGS,
     flexpath_object_set_datatypes_doc},
    {"set_joins", (PyCFunction)flexpath_object_set_joins, METH_VARARGS,
     flexpath_object_set_joins_doc},
    {"set_ends", (PyCFunction)flexpath_object_set_ends, METH_VARARGS, flexpath_object_set_ends_doc},
    {"set_bend_radius", (PyCFunction)flexpath_object_set_bend_radius, METH_VARARGS,
     flexpath_object_set_bend_radius_doc},
    {"set_bend_function", (PyCFunction)flexpath_object_set_bend_function, METH_VARARGS,
     flexpath_object_set_bend_function_doc},
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
    {"apply_repetition", (PyCFunction)flexpath_object_apply_repetition, METH_NOARGS,
     flexpath_object_apply_repetition_doc},
    {"set_property", (PyCFunction)flexpath_object_set_property, METH_VARARGS,
     object_set_property_doc},
    {"get_property", (PyCFunction)flexpath_object_get_property, METH_VARARGS,
     object_get_property_doc},
    {"delete_property", (PyCFunction)flexpath_object_delete_property, METH_VARARGS,
     object_delete_property_doc},
    {"set_gds_property", (PyCFunction)flexpath_object_set_gds_property, METH_VARARGS,
     object_set_gds_property_doc},
    {"get_gds_property", (PyCFunction)flexpath_object_get_gds_property, METH_VARARGS,
     object_get_gds_property_doc},
    {"delete_gds_property", (PyCFunction)flexpath_object_delete_gds_property, METH_VARARGS,
     object_delete_gds_property_doc},
    {NULL}};

static PyObject* flexpath_object_get_layers(FlexPathObject* self, void*) {
    FlexPath* flexpath = self->flexpath;
    PyObject* result = PyTuple_New(flexpath->num_elements);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return tuple.");
        return NULL;
    }
    for (uint64_t i = 0; i < flexpath->num_elements; i++) {
        PyObject* item = PyLong_FromUnsignedLongLong(get_layer(flexpath->elements[i].tag));
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
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return tuple.");
        return NULL;
    }
    for (uint64_t i = 0; i < flexpath->num_elements; i++) {
        PyObject* item = PyLong_FromUnsignedLongLong(get_type(flexpath->elements[i].tag));
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
    return PyLong_FromUnsignedLongLong(self->flexpath->num_elements);
}

static PyObject* flexpath_object_get_size(FlexPathObject* self, void*) {
    return PyLong_FromUnsignedLongLong(self->flexpath->spine.point_array.count);
}

static PyObject* flexpath_object_get_joins(FlexPathObject* self, void*) {
    FlexPath* flexpath = self->flexpath;
    PyObject* result = PyTuple_New(flexpath->num_elements);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return tuple.");
        return NULL;
    }
    for (uint64_t i = 0; i < flexpath->num_elements; i++) {
        FlexPathElement* element = flexpath->elements + i;
        PyObject* item = NULL;
        switch (element->join_type) {
            case JoinType::Natural:
                item = PyUnicode_FromString("natural");
                break;
            case JoinType::Miter:
                item = PyUnicode_FromString("miter");
                break;
            case JoinType::Bevel:
                item = PyUnicode_FromString("bevel");
                break;
            case JoinType::Round:
                item = PyUnicode_FromString("round");
                break;
            case JoinType::Smooth:
                item = PyUnicode_FromString("smooth");
                break;
            case JoinType::Function:
                item = (PyObject*)element->join_function_data;
                Py_INCREF(item);
                break;
        }
        if (item == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create return object item.");
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* flexpath_object_get_ends(FlexPathObject* self, void*) {
    FlexPath* flexpath = self->flexpath;
    PyObject* result = PyTuple_New(flexpath->num_elements);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return tuple.");
        return NULL;
    }
    for (uint64_t i = 0; i < flexpath->num_elements; i++) {
        FlexPathElement* element = flexpath->elements + i;
        PyObject* item = NULL;
        switch (element->end_type) {
            case EndType::Flush:
                item = PyUnicode_FromString("flush");
                break;
            case EndType::Round:
                item = PyUnicode_FromString("round");
                break;
            case EndType::HalfWidth:
                item = PyUnicode_FromString("extendend");
                break;
            case EndType::Smooth:
                item = PyUnicode_FromString("smooth");
                break;
            case EndType::Extended: {
                item = PyTuple_New(2);
                if (item == NULL) {
                    PyErr_SetString(PyExc_RuntimeError, "Unable to create return object item.");
                    Py_DECREF(result);
                    return NULL;
                }
                PyObject* value = PyFloat_FromDouble(element->end_extensions.u);
                if (PyErr_Occurred()) {
                    PyErr_SetString(PyExc_RuntimeError, "Unable to create return object item.");
                    Py_DECREF(item);
                    Py_DECREF(result);
                    return NULL;
                }
                PyTuple_SET_ITEM(item, 0, value);
                value = PyFloat_FromDouble(element->end_extensions.v);
                if (PyErr_Occurred()) {
                    PyErr_SetString(PyExc_RuntimeError, "Unable to create return object item.");
                    Py_DECREF(item);
                    Py_DECREF(result);
                    return NULL;
                }
                PyTuple_SET_ITEM(item, 1, value);
            } break;
            case EndType::Function:
                item = (PyObject*)element->end_function_data;
                Py_INCREF(item);
                break;
        }
        if (item == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create return object item.");
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* flexpath_object_get_bend_radius(FlexPathObject* self, void*) {
    FlexPath* flexpath = self->flexpath;
    PyObject* result = PyTuple_New(flexpath->num_elements);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return tuple.");
        return NULL;
    }
    for (uint64_t i = 0; i < flexpath->num_elements; i++) {
        PyObject* item = PyFloat_FromDouble(flexpath->elements[i].bend_radius);
        if (item == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create return object item.");
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* flexpath_object_get_bend_function(FlexPathObject* self, void*) {
    FlexPath* flexpath = self->flexpath;
    PyObject* result = PyTuple_New(flexpath->num_elements);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return tuple.");
        return NULL;
    }
    for (uint64_t i = 0; i < flexpath->num_elements; i++) {
        FlexPathElement* element = flexpath->elements + i;
        PyObject* item = element->bend_type == BendType::Function
                             ? (PyObject*)element->bend_function_data
                             : Py_None;
        Py_INCREF(item);
        PyTuple_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* flexpath_object_get_tolerance(FlexPathObject* self, void*) {
    return PyFloat_FromDouble(self->flexpath->spine.tolerance);
}

int flexpath_object_set_tolerance(FlexPathObject* self, PyObject* arg, void*) {
    double tolerance = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert value to float.");
        return -1;
    }
    if (tolerance <= 0) {
        PyErr_SetString(PyExc_ValueError, "Tolerance must be positive.");
        return -1;
    }
    self->flexpath->spine.tolerance = tolerance;
    return 0;
}

static PyObject* flexpath_object_get_simple_path(FlexPathObject* self, void*) {
    PyObject* result = self->flexpath->simple_path ? Py_True : Py_False;
    Py_INCREF(result);
    return result;
}

int flexpath_object_set_simple_path(FlexPathObject* self, PyObject* arg, void*) {
    self->flexpath->simple_path = PyObject_IsTrue(arg) > 0;
    return 0;
}

static PyObject* flexpath_object_get_scale_width(FlexPathObject* self, void*) {
    PyObject* result = self->flexpath->scale_width ? Py_True : Py_False;
    Py_INCREF(result);
    return result;
}

int flexpath_object_set_scale_width(FlexPathObject* self, PyObject* arg, void*) {
    self->flexpath->scale_width = PyObject_IsTrue(arg) > 0;
    return 0;
}

static PyObject* flexpath_object_get_properties(FlexPathObject* self, void*) {
    return build_properties(self->flexpath->properties);
}

int flexpath_object_set_properties(FlexPathObject* self, PyObject* arg, void*) {
    return parse_properties(self->flexpath->properties, arg);
}

static PyObject* flexpath_object_get_repetition(FlexPathObject* self, void*) {
    RepetitionObject* obj = PyObject_New(RepetitionObject, &repetition_object_type);
    obj = (RepetitionObject*)PyObject_Init((PyObject*)obj, &repetition_object_type);
    obj->repetition.copy_from(self->flexpath->repetition);
    return (PyObject*)obj;
}

int flexpath_object_set_repetition(FlexPathObject* self, PyObject* arg, void*) {
    if (arg == Py_None) {
        self->flexpath->repetition.clear();
        return 0;
    } else if (!RepetitionObject_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Value must be a Repetition object.");
        return -1;
    }
    RepetitionObject* repetition_obj = (RepetitionObject*)arg;
    self->flexpath->repetition.clear();
    self->flexpath->repetition.copy_from(repetition_obj->repetition);
    return 0;
}

static PyObject* flexpath_object_get_raith_data(FlexPathObject* self, void*) {
    RaithDataObject* obj = PyObject_New(RaithDataObject, &raithdata_object_type);
    obj = (RaithDataObject*)PyObject_Init((PyObject*)obj, &raithdata_object_type);
    obj->raith_data.base_cell_name = NULL;
    obj->raith_data.copy_from(self->flexpath->raith_data);
    return (PyObject*)obj;
}

int flexpath_object_set_raith_data(FlexPathObject* self, PyObject* arg, void*) {
    if (arg == Py_None) {
        self->flexpath->raith_data.clear();
        return 0;
    }
    if (!RaithDataObject_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Value must be a RaithData object.");
        return -1;
    }
    RaithDataObject* raith_data_obj = (RaithDataObject*)arg;
    self->flexpath->raith_data.copy_from(raith_data_obj->raith_data);
    return 0;
}

static PyGetSetDef flexpath_object_getset[] = {
    {"layers", (getter)flexpath_object_get_layers, NULL, flexpath_object_layers_doc, NULL},
    {"datatypes", (getter)flexpath_object_get_datatypes, NULL, flexpath_object_datatypes_doc, NULL},
    {"num_paths", (getter)flexpath_object_get_num_paths, NULL, flexpath_object_num_paths_doc, NULL},
    {"size", (getter)flexpath_object_get_size, NULL, flexpath_object_size_doc, NULL},
    {"joins", (getter)flexpath_object_get_joins, NULL, flexpath_object_joins_doc, NULL},
    {"ends", (getter)flexpath_object_get_ends, NULL, flexpath_object_ends_doc, NULL},
    {"bend_radius", (getter)flexpath_object_get_bend_radius, NULL, flexpath_object_bend_radius_doc,
     NULL},
    {"bend_function", (getter)flexpath_object_get_bend_function, NULL,
     flexpath_object_bend_function_doc, NULL},
    {"tolerance", (getter)flexpath_object_get_tolerance, (setter)flexpath_object_set_tolerance,
     path_object_tolerance_doc, NULL},
    {"simple_path", (getter)flexpath_object_get_simple_path,
     (setter)flexpath_object_set_simple_path, path_object_simple_path_doc, NULL},
    {"scale_width", (getter)flexpath_object_get_scale_width,
     (setter)flexpath_object_set_scale_width, path_object_scale_width_doc, NULL},
    {"properties", (getter)flexpath_object_get_properties, (setter)flexpath_object_set_properties,
     object_properties_doc, NULL},
    {"repetition", (getter)flexpath_object_get_repetition, (setter)flexpath_object_set_repetition,
     object_repetition_doc, NULL},
    {"raith_data", (getter)flexpath_object_get_raith_data, (setter)flexpath_object_set_raith_data,
     flexpath_object_raith_data_doc, NULL},
    {NULL}};
