/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* robustpath_object_str(RobustPathObject* self) {
    char buffer[GDSTK_PRINT_BUFFER_COUNT];
    snprintf(buffer, COUNT(buffer), "RobustPath with %" PRIu64 " paths and %" PRIu64 " sections",
             self->robustpath->num_elements, self->robustpath->subpath_array.count);
    return PyUnicode_FromString(buffer);
}

static void robustpath_cleanup(RobustPathObject* self) {
    RobustPath* path = self->robustpath;
    RobustPathElement* el = path->elements;
    for (uint64_t j = path->num_elements; j > 0; j--, el++) {
        Py_XDECREF(el->end_function_data);
        Interpolation* interp = el->width_array.items;
        for (uint64_t i = el->width_array.count; i > 0; i--, interp++)
            if (interp->type == InterpolationType::Parametric) Py_XDECREF(interp->data);
        interp = el->offset_array.items;
        for (uint64_t i = el->offset_array.count; i > 0; i--, interp++)
            if (interp->type == InterpolationType::Parametric) Py_XDECREF(interp->data);
    }
    SubPath* sub = path->subpath_array.items;
    for (uint64_t j = path->subpath_array.count; j > 0; j--, sub++)
        if (sub->type == SubPathType::Parametric) {
            Py_XDECREF(sub->func_data);
            if (sub->path_gradient != NULL) Py_XDECREF(sub->grad_data);
        }
    path->clear();
    free_allocation(path);
    self->robustpath = NULL;
}

static void robustpath_object_dealloc(RobustPathObject* self) {
    if (self->robustpath) robustpath_cleanup(self);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int robustpath_object_init(RobustPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_point = NULL;
    PyObject* py_width = NULL;
    PyObject* py_offset = NULL;
    PyObject* py_ends = NULL;
    PyObject* py_layer = NULL;
    PyObject* py_datatype = NULL;
    double tolerance = 1e-2;
    uint64_t max_evals = 1000;
    int simple_path = 0;
    int scale_width = 1;
    const char* keywords[] = {"initial_point", "width",     "offset",      "ends",
                              "tolerance",     "max_evals", "simple_path", "scale_width",
                              "layer",         "datatype",  NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OOdlppOO:RobustPath", (char**)keywords,
                                     &py_point, &py_width, &py_offset, &py_ends, &tolerance,
                                     &max_evals, &simple_path, &scale_width, &py_layer,
                                     &py_datatype))
        return -1;

    if (tolerance <= 0) {
        PyErr_SetString(PyExc_ValueError, "Tolerance must be positive.");
        return -1;
    }

    if (max_evals < 1) {
        PyErr_SetString(PyExc_ValueError, "Argument max_evals must be greater than 0.");
        return -1;
    }

    if (self->robustpath) {
        RobustPath* robustpath = self->robustpath;
        RobustPathElement* el = robustpath->elements;
        for (uint64_t j = 0; j < robustpath->num_elements; j++, el++) {
            Py_XDECREF(el->end_function_data);
            Interpolation* interp = el->width_array.items;
            for (uint64_t i = el->width_array.count; i > 0; i--, interp++)
                if (interp->type == InterpolationType::Parametric) Py_XDECREF(interp->data);
            interp = el->offset_array.items;
            for (uint64_t i = el->offset_array.count; i > 0; i--, interp++)
                if (interp->type == InterpolationType::Parametric) Py_XDECREF(interp->data);
        }
        SubPath* sub = self->robustpath->subpath_array.items;
        for (uint64_t j = self->robustpath->subpath_array.count; j > 0; j--, sub++)
            if (sub->type == SubPathType::Parametric) {
                Py_XDECREF(sub->func_data);
                if (sub->path_gradient != NULL) Py_XDECREF(sub->grad_data);
            }
        robustpath->clear();
    } else {
        self->robustpath = (RobustPath*)allocate_clear(sizeof(RobustPath));
    }
    RobustPath* robustpath = self->robustpath;

    if (parse_point(py_point, robustpath->end_point, "point") < 0) {
        robustpath_cleanup(self);
        return -1;
    }
    uint64_t num_elements = 1;

    if (PySequence_Check(py_width)) {
        num_elements = PySequence_Length(py_width);
        robustpath->num_elements = num_elements;
        robustpath->elements =
            (RobustPathElement*)allocate_clear(num_elements * sizeof(RobustPathElement));
        if (py_offset && PySequence_Check(py_offset)) {
            if ((uint64_t)PySequence_Length(py_offset) != num_elements) {
                robustpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError,
                                "Sequences width and offset must have the same length.");
                return -1;
            }

            // Case 1: width and offset are sequences with the same length
            RobustPathElement* el = robustpath->elements;
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PySequence_ITEM(py_width, i);
                if (item == NULL) {
                    robustpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRIu64 " from width sequence.", i);
                    return -1;
                }
                el->end_width = PyFloat_AsDouble(item);
                Py_DECREF(item);
                if (PyErr_Occurred()) {
                    robustpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert width[%" PRIu64 "] to float.", i);
                    return -1;
                }
                if (el->end_width < 0) {
                    robustpath_cleanup(self);
                    PyErr_Format(PyExc_ValueError,
                                 "Negative width value not allowed: width[%" PRIu64 "].", i);
                    return -1;
                }

                item = PySequence_ITEM(py_offset, i);
                if (item == NULL) {
                    robustpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRIu64 " from offset sequence.", i);
                    return -1;
                }
                el->end_offset = PyFloat_AsDouble(item);
                Py_DECREF(item);
                if (PyErr_Occurred()) {
                    robustpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert offset[%" PRIu64 "] to float.", i);
                    return -1;
                }
            }
        } else {
            // Case 2: width is a sequence, offset a number
            const double offset = py_offset == NULL ? 0 : PyFloat_AsDouble(py_offset);
            if (PyErr_Occurred()) {
                robustpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert offset to float.");
                return -1;
            }

            RobustPathElement* el = robustpath->elements;
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                el->end_offset = (i - 0.5 * (num_elements - 1)) * offset;

                PyObject* item = PySequence_ITEM(py_width, i);
                if (item == NULL) {
                    robustpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to retrieve item %" PRIu64 " from width sequence.", i);
                    return -1;
                }
                el->end_width = PyFloat_AsDouble(item);
                Py_DECREF(item);
                if (PyErr_Occurred()) {
                    robustpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert width[%" PRIu64 "] to float.", i);
                    return -1;
                }
                if (el->end_width < 0) {
                    robustpath_cleanup(self);
                    PyErr_Format(PyExc_ValueError,
                                 "Negative width value not allowed: width[%" PRIu64 "].", i);
                    return -1;
                }
            }
        }
    } else if (py_offset && PySequence_Check(py_offset)) {
        // Case 3: offset is a sequence, width a number
        num_elements = PySequence_Length(py_offset);
        robustpath->num_elements = num_elements;
        robustpath->elements =
            (RobustPathElement*)allocate_clear(num_elements * sizeof(RobustPathElement));
        const double width = PyFloat_AsDouble(py_width);
        if (PyErr_Occurred()) {
            robustpath_cleanup(self);
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert width to float.");
            return -1;
        }
        if (width < 0) {
            robustpath_cleanup(self);
            PyErr_SetString(PyExc_ValueError, "Negative width value not allowed.");
            return -1;
        }

        RobustPathElement* el = robustpath->elements;
        for (uint64_t i = 0; i < num_elements; i++, el++) {
            el->end_width = width;

            PyObject* item = PySequence_ITEM(py_offset, i);
            if (item == NULL) {
                robustpath_cleanup(self);
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to retrieve item %" PRIu64 " from offset sequence.", i);
                return -1;
            }
            el->end_offset = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                robustpath_cleanup(self);
                PyErr_Format(PyExc_RuntimeError, "Unable to convert offset[%" PRIu64 "] to float.",
                             i);
                return -1;
            }
        }
    } else {
        // Case 4: width and offset are numbers
        robustpath->num_elements = 1;
        robustpath->elements = (RobustPathElement*)allocate_clear(sizeof(RobustPathElement));
        robustpath->elements[0].end_width = PyFloat_AsDouble(py_width);
        if (PyErr_Occurred()) {
            robustpath_cleanup(self);
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert width to float.");
            return -1;
        }
        if (robustpath->elements[0].end_width < 0) {
            robustpath_cleanup(self);
            PyErr_SetString(PyExc_ValueError, "Negative width value not allowed.");
            return -1;
        }
        if (py_offset != NULL) {
            robustpath->elements[0].end_offset = PyFloat_AsDouble(py_offset);
            if (PyErr_Occurred()) {
                robustpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert offset to float.");
                return -1;
            }
        }
    }

    if (py_layer) {
        if (PyList_Check(py_layer)) {
            if ((uint64_t)PyList_GET_SIZE(py_layer) != num_elements) {
                robustpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError,
                                "List layer must have the same length as the number of paths.");
                return -1;
            }
            RobustPathElement* el = robustpath->elements;
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_layer, i);
                if (item == NULL) {
                    robustpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to get item %" PRIu64 " from layer list.", i);
                    return -1;
                }
                set_layer(el->tag, (uint32_t)PyLong_AsUnsignedLongLong(item));
                if (PyErr_Occurred()) {
                    robustpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError, "Unable to convert layer[%" PRIu64 "] to int.",
                                 i);
                    return -1;
                }
            }
        } else {
            const uint32_t layer = (uint32_t)PyLong_AsUnsignedLongLong(py_layer);
            if (PyErr_Occurred()) {
                robustpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert layer to int.");
                return -1;
            }
            RobustPathElement* el = robustpath->elements;
            for (uint64_t i = 0; i < num_elements; i++) set_layer((el++)->tag, layer);
        }
    }

    if (py_datatype) {
        if (PyList_Check(py_datatype)) {
            if ((uint64_t)PyList_GET_SIZE(py_datatype) != num_elements) {
                robustpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError,
                                "List datatype must have the same length as the number of paths.");
                return -1;
            }
            RobustPathElement* el = robustpath->elements;
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_datatype, i);
                if (item == NULL) {
                    robustpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to get item %" PRIu64 " from datatype list.", i);
                    return -1;
                }
                set_type(el->tag, (uint32_t)PyLong_AsUnsignedLongLong(item));
                if (PyErr_Occurred()) {
                    robustpath_cleanup(self);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert datatype[%" PRIu64 "] to int.", i);
                    return -1;
                }
            }
        } else {
            const uint32_t datatype = (uint32_t)PyLong_AsUnsignedLongLong(py_datatype);
            if (PyErr_Occurred()) {
                robustpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert datatype to int.");
                return -1;
            }
            RobustPathElement* el = robustpath->elements;
            for (uint64_t i = 0; i < num_elements; i++) set_type((el++)->tag, datatype);
        }
    }

    if (py_ends) {
        if (PyList_Check(py_ends)) {
            if ((uint64_t)PyList_GET_SIZE(py_ends) != num_elements) {
                robustpath_cleanup(self);
                PyErr_SetString(PyExc_RuntimeError,
                                "List ends must have the same length as the number of paths.");
                return -1;
            }
            RobustPathElement* el = robustpath->elements;
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                PyObject* item = PyList_GET_ITEM(py_ends, i);
                if (item == NULL) {
                    robustpath_cleanup(self);
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
                            robustpath_cleanup(self);
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
                            robustpath_cleanup(self);
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
            RobustPathElement* el = robustpath->elements;
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
                    robustpath_cleanup(self);
                    PyErr_SetString(
                        PyExc_RuntimeError,
                        "Argument ends must be one of 'flush', 'extended', 'round', 'smooth', a 2-tuple, a callable, or a list of those.");
                    return -1;
                }
            } else {
                et = EndType::Extended;
                if (!PyTuple_Check(py_ends) || PyArg_ParseTuple(py_ends, "dd", &ex.u, &ex.v) < 0) {
                    robustpath_cleanup(self);
                    PyErr_SetString(
                        PyExc_RuntimeError,
                        "Argument ends must be one of 'flush', 'extended', 'round', 'smooth', a 2-tuple, a callable, or a list of those.");
                    return -1;
                }
            }
            RobustPathElement* el = robustpath->elements;
            for (uint64_t i = 0; i < num_elements; i++, el++) {
                el->end_type = et;
                el->end_extensions = ex;
            }
        }
    }

    robustpath->tolerance = tolerance;
    robustpath->max_evals = max_evals;
    robustpath->width_scale = 1;
    robustpath->offset_scale = 1;
    robustpath->trafo[0] = 1;
    robustpath->trafo[4] = 1;
    robustpath->simple_path = simple_path > 0;
    robustpath->scale_width = scale_width > 0;
    robustpath->owner = self;
    return 0;
}

static PyObject* robustpath_object_copy(RobustPathObject* self, PyObject*) {
    RobustPathObject* result = PyObject_New(RobustPathObject, &robustpath_object_type);
    result = (RobustPathObject*)PyObject_Init((PyObject*)result, &robustpath_object_type);
    result->robustpath = (RobustPath*)allocate_clear(sizeof(RobustPath));
    result->robustpath->copy_from(*self->robustpath);
    result->robustpath->owner = result;
    return (PyObject*)result;
}

static PyObject* robustpath_object_deepcopy(RobustPathObject* self, PyObject* arg) {
    return robustpath_object_copy(self, NULL);
}

static PyObject* robustpath_object_spine(RobustPathObject* self, PyObject*) {
    Array<Vec2> point_array = {};
    if (return_error(self->robustpath->spine(point_array))) return NULL;

    npy_intp dims[] = {(npy_intp)point_array.count, 2};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) {
        point_array.clear();
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    memcpy(data, point_array.items, sizeof(double) * point_array.count * 2);
    point_array.clear();
    return (PyObject*)result;
}

static PyObject* robustpath_object_path_spines(RobustPathObject* self, PyObject*) {
    Array<Vec2> point_array = {};
    RobustPath* path = self->robustpath;
    PyObject* result = PyList_New(path->num_elements);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        return NULL;
    }
    RobustPathElement* el = path->elements;
    for (uint64_t i = 0; i < path->num_elements; i++) {
        if (return_error(path->element_center(el++, point_array))) {
            Py_DECREF(result);
            point_array.clear();
            return NULL;
        }
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

static PyObject* robustpath_object_widths(RobustPathObject* self, PyObject* args, PyObject* kwds) {
    double u = 0;
    int from_below = 1;
    const char* keywords[] = {"u", "from_below", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|p:widths", (char**)keywords, &u, &from_below))
        return NULL;
    const RobustPath* robustpath = self->robustpath;
    npy_intp dims[] = {(npy_intp)robustpath->num_elements};
    PyObject* result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    robustpath->width(u, from_below > 0, data);
    return (PyObject*)result;
}

static PyObject* robustpath_object_offsets(RobustPathObject* self, PyObject* args, PyObject* kwds) {
    double u = 0;
    int from_below = 1;
    const char* keywords[] = {"u", "from_below", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|p:offsets", (char**)keywords, &u, &from_below))
        return NULL;
    const RobustPath* robustpath = self->robustpath;
    npy_intp dims[] = {(npy_intp)robustpath->num_elements};
    PyObject* result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    robustpath->offset(u, from_below > 0, data);
    return (PyObject*)result;
}

static PyObject* robustpath_object_position(RobustPathObject* self, PyObject* args,
                                            PyObject* kwds) {
    double u = 0;
    int from_below = 1;
    const char* keywords[] = {"u", "from_below", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|p:position", (char**)keywords, &u, &from_below))
        return NULL;
    const RobustPath* robustpath = self->robustpath;
    npy_intp dims[] = {2};
    PyObject* result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return array.");
        return NULL;
    }
    Vec2* data = (Vec2*)PyArray_DATA((PyArrayObject*)result);
    *data = robustpath->position(u, from_below > 0);
    return (PyObject*)result;
}

static PyObject* robustpath_object_gradient(RobustPathObject* self, PyObject* args,
                                            PyObject* kwds) {
    double u = 0;
    int from_below = 1;
    const char* keywords[] = {"u", "from_below", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|p:gradient", (char**)keywords, &u, &from_below))
        return NULL;
    const RobustPath* robustpath = self->robustpath;
    npy_intp dims[] = {2};
    PyObject* result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return array.");
        return NULL;
    }
    Vec2* data = (Vec2*)PyArray_DATA((PyArrayObject*)result);
    *data = robustpath->gradient(u, from_below > 0);
    return (PyObject*)result;
}

static PyObject* robustpath_object_to_polygons(RobustPathObject* self, PyObject*) {
    Array<Polygon*> array = {};
    if (return_error(self->robustpath->to_polygons(false, 0, array))) {
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

static PyObject* robustpath_object_set_layers(RobustPathObject* self, PyObject* arg) {
    if (!PySequence_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Value must be a sequence of layer numbers.");
        return NULL;
    }
    uint64_t len = PySequence_Length(arg);
    RobustPath* robustpath = self->robustpath;
    if (len != robustpath->num_elements) {
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
        set_layer(robustpath->elements[i].tag, (uint32_t)PyLong_AsUnsignedLongLong(item));
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

static PyObject* robustpath_object_set_datatypes(RobustPathObject* self, PyObject* arg) {
    if (!PySequence_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Value must be a sequence of datatype numbers.");
        return NULL;
    }
    uint64_t len = PySequence_Length(arg);
    RobustPath* robustpath = self->robustpath;
    if (len != robustpath->num_elements) {
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
        set_type(robustpath->elements[i].tag, (uint32_t)PyLong_AsUnsignedLongLong(item));
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_Format(PyExc_TypeError, "Unable to convert sequence item %" PRIu64 " to int.", i);
            return NULL;
        }
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_set_ends(RobustPathObject* self, PyObject* arg) {
    if (!PySequence_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a sequence of end types.");
        return NULL;
    }
    uint64_t len = PySequence_Length(arg);
    RobustPath* robustpath = self->robustpath;
    if (len != robustpath->num_elements) {
        PyErr_SetString(PyExc_RuntimeError, "Length of sequence must match the number of paths.");
        return NULL;
    }
    for (uint64_t i = 0; i < len; i++) {
        RobustPathElement* el = robustpath->elements + i;
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

// Note: offset must be an array of count at least robustpath.num_elements.  If py_offset is a
// single number, it's the new distance between paths (analogous to what is used in init).
static int parse_robustpath_offset(RobustPath& robustpath, PyObject* py_offset,
                                   Interpolation* offset) {
    if (PyList_Check(py_offset)) {
        if ((uint64_t)PyList_GET_SIZE(py_offset) < robustpath.num_elements) {
            PyErr_SetString(PyExc_RuntimeError, "List offset doesn't have enough elements.");
            return -1;
        }
        for (uint64_t i = 0; i < robustpath.num_elements; i++, offset++) {
            PyObject* item = PyList_GET_ITEM(py_offset, i);
            if (item == NULL) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get item [%" PRIu64 "] from sequence offset.", i);
                return -1;
            }
            if (PyTuple_Check(item)) {
                double value;
                const char* type;
                if (!PyArg_ParseTuple(item, "ds", &value, &type)) {
                    PyErr_SetString(
                        PyExc_RuntimeError,
                        "Offset tuple must contain a number and the interpolation specification ('constant', 'linear', or 'smooth').");
                    return -1;
                }
                if (strcmp(type, "constant") == 0) {
                    offset->type = InterpolationType::Constant;
                    offset->value = value;
                } else {
                    offset->initial_value = robustpath.elements[i].end_offset;
                    offset->final_value = value;
                    if (strcmp(type, "linear") == 0)
                        offset->type = InterpolationType::Linear;
                    else if (strcmp(type, "smooth") == 0)
                        offset->type = InterpolationType::Smooth;
                    else {
                        PyErr_SetString(
                            PyExc_RuntimeError,
                            "Offset tuple must contain a number and the interpolation specification ('constant', 'linear', or 'smooth').");
                        return -1;
                    }
                }
            } else if (PyCallable_Check(item)) {
                offset->type = InterpolationType::Parametric;
                offset->function = (ParametricDouble)eval_parametric_double;
                offset->data = (void*)item;
                Py_INCREF(item);
            } else {
                offset->type = InterpolationType::Linear;
                offset->initial_value = robustpath.elements[i].end_offset;
                offset->final_value = PyFloat_AsDouble(item);
                if (PyErr_Occurred()) {
                    PyErr_SetString(
                        PyExc_RuntimeError,
                        "Argument offset must be a number, a 2-tuple, a callable, or a list of those.");
                    return -1;
                }
            }
        }
    } else {
        if (PyTuple_Check(py_offset)) {
            double value;
            const char* type;
            if (!PyArg_ParseTuple(py_offset, "ds", &value, &type)) {
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Offset tuple must contain a number and the interpolation specification ('constant', 'linear', or 'smooth').");
                return -1;
            }
            if (strcmp(type, "constant") == 0) {
                for (uint64_t i = 0; i < robustpath.num_elements; i++, offset++) {
                    offset->type = InterpolationType::Constant;
                    offset->value = (i - 0.5 * (robustpath.num_elements - 1)) * value;
                }
                return 0;
            } else {
                InterpolationType interp_type;
                if (strcmp(type, "linear") == 0)
                    interp_type = InterpolationType::Linear;
                else if (strcmp(type, "smooth") == 0)
                    interp_type = InterpolationType::Smooth;
                else {
                    PyErr_SetString(
                        PyExc_RuntimeError,
                        "Offset tuple must contain a number and the interpolation specification ('constant', 'linear', or 'smooth').");
                    return -1;
                }
                for (uint64_t i = 0; i < robustpath.num_elements; i++, offset++) {
                    offset->type = interp_type;
                    offset->initial_value = robustpath.elements[i].end_offset;
                    offset->final_value = (i - 0.5 * (robustpath.num_elements - 1)) * value;
                }
            }
        } else if (PyCallable_Check(py_offset)) {
            for (uint64_t i = 0; i < robustpath.num_elements; i++, offset++) {
                offset->type = InterpolationType::Parametric;
                offset->function = (ParametricDouble)eval_parametric_double;
                offset->data = (void*)py_offset;
                Py_INCREF(py_offset);
            }
        } else {
            double value = PyFloat_AsDouble(py_offset);
            if (PyErr_Occurred()) {
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Argument offset must be a number, a 2-tuple, a callable, or a list of those.");
                return -1;
            }
            for (uint64_t i = 0; i < robustpath.num_elements; i++, offset++) {
                offset->type = InterpolationType::Linear;
                offset->initial_value = robustpath.elements[i].end_offset;
                offset->final_value = (i - 0.5 * (robustpath.num_elements - 1)) * value;
            }
        }
    }
    return 0;
}

// Note: width must be an array of count at least robustpath.num_elements.
static int parse_robustpath_width(RobustPath& robustpath, PyObject* py_width,
                                  Interpolation* width) {
    if (PyList_Check(py_width)) {
        if ((uint64_t)PyList_GET_SIZE(py_width) < robustpath.num_elements) {
            PyErr_SetString(PyExc_RuntimeError, "List width doesn't have enough elements.");
            return -1;
        }
        for (uint64_t i = 0; i < robustpath.num_elements; i++, width++) {
            PyObject* item = PyList_GET_ITEM(py_width, i);
            if (item == NULL) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get item %" PRIu64 " from sequence width.", i);
                return -1;
            }
            if (PyTuple_Check(item)) {
                double value;
                const char* type;
                if (!PyArg_ParseTuple(item, "ds", &value, &type)) {
                    PyErr_SetString(
                        PyExc_RuntimeError,
                        "Width tuple must contain a number and the interpolation specification ('constant', 'linear', or 'smooth').");
                    return -1;
                }
                if (value < 0) {
                    PyErr_Format(PyExc_ValueError,
                                 "Negative width value not allowed: width[%" PRIu64 "].", i);
                    return -1;
                }
                if (strcmp(type, "constant") == 0) {
                    width->type = InterpolationType::Constant;
                    width->value = value;
                } else {
                    width->initial_value = robustpath.elements[i].end_width;
                    width->final_value = value;
                    if (strcmp(type, "linear") == 0)
                        width->type = InterpolationType::Linear;
                    else if (strcmp(type, "smooth") == 0)
                        width->type = InterpolationType::Smooth;
                    else {
                        PyErr_SetString(
                            PyExc_RuntimeError,
                            "Width tuple must contain a number and the interpolation specification ('constant', 'linear', or 'smooth').");
                        return -1;
                    }
                }
            } else if (PyCallable_Check(item)) {
                width->type = InterpolationType::Parametric;
                width->function = (ParametricDouble)eval_parametric_double;
                width->data = (void*)item;
                Py_INCREF(item);
            } else {
                double value = PyFloat_AsDouble(item);
                if (PyErr_Occurred()) {
                    PyErr_SetString(
                        PyExc_RuntimeError,
                        "Argument width must be a number, a 2-tuple, a callable, or a list of those.");
                    return -1;
                }
                if (value < 0) {
                    PyErr_Format(PyExc_ValueError,
                                 "Negative width value not allowed: width[%" PRIu64 "].", i);
                    return -1;
                }
                width->type = InterpolationType::Linear;
                width->initial_value = robustpath.elements[i].end_width;
                width->final_value = value;
            }
        }
    } else {
        if (PyTuple_Check(py_width)) {
            double value;
            const char* type;
            if (!PyArg_ParseTuple(py_width, "ds", &value, &type)) {
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Width tuple must contain a number and the interpolation specification ('constant', 'linear', or 'smooth').");
                return -1;
            }
            if (value < 0) {
                PyErr_SetString(PyExc_ValueError, "Negative width value not allowed.");
                return -1;
            }
            if (strcmp(type, "constant") == 0) {
                for (uint64_t i = 0; i < robustpath.num_elements; i++, width++) {
                    width->type = InterpolationType::Constant;
                    width->value = value;
                }
                return 0;
            } else {
                InterpolationType interp_type;
                if (strcmp(type, "linear") == 0)
                    interp_type = InterpolationType::Linear;
                else if (strcmp(type, "smooth") == 0)
                    interp_type = InterpolationType::Smooth;
                else {
                    PyErr_SetString(
                        PyExc_RuntimeError,
                        "Width tuple must contain a number and the interpolation specification ('constant', 'linear', or 'smooth').");
                    return -1;
                }
                for (uint64_t i = 0; i < robustpath.num_elements; i++, width++) {
                    width->type = interp_type;
                    width->initial_value = robustpath.elements[i].end_width;
                    width->final_value = value;
                }
            }
        } else if (PyCallable_Check(py_width)) {
            for (uint64_t i = 0; i < robustpath.num_elements; i++, width++) {
                width->type = InterpolationType::Parametric;
                width->function = (ParametricDouble)eval_parametric_double;
                width->data = (void*)py_width;
                Py_INCREF(py_width);
            }
        } else {
            double value = PyFloat_AsDouble(py_width);
            if (PyErr_Occurred()) {
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Argument width must be a number, a 2-tuple, a callable, or a list of those.");
                return -1;
            }
            if (value < 0) {
                PyErr_SetString(PyExc_ValueError, "Negative width value not allowed.");
                return -1;
            }
            for (uint64_t i = 0; i < robustpath.num_elements; i++, width++) {
                width->type = InterpolationType::Linear;
                width->initial_value = robustpath.elements[i].end_width;
                width->final_value = value;
            }
        }
    }
    return 0;
}

static PyObject* robustpath_object_horizontal(RobustPathObject* self, PyObject* args,
                                              PyObject* kwds) {
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    double coord = 0;
    int relative = 0;
    const char* keywords[] = {"x", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|OOp:horizontal", (char**)keywords, &coord,
                                     &py_width, &py_offset, &relative))
        return NULL;
    RobustPath* robustpath = self->robustpath;
    Interpolation* o_buffer =
        (Interpolation*)allocate(sizeof(Interpolation) * 2 * robustpath->num_elements);
    Interpolation* w_buffer = o_buffer + robustpath->num_elements;
    Interpolation* offset = NULL;
    Interpolation* width = NULL;
    if (py_offset != Py_None) {
        offset = o_buffer;
        if (parse_robustpath_offset(*robustpath, py_offset, offset) < 0) {
            free_allocation(o_buffer);
            return NULL;
        }
    }
    if (py_width != Py_None) {
        width = w_buffer;
        if (parse_robustpath_width(*robustpath, py_width, width) < 0) {
            free_allocation(o_buffer);
            return NULL;
        }
    }
    robustpath->horizontal(coord, width, offset, relative > 0);
    free_allocation(o_buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_vertical(RobustPathObject* self, PyObject* args,
                                            PyObject* kwds) {
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    double coord = 0;
    int relative = 0;
    const char* keywords[] = {"y", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|OOp:vertical", (char**)keywords, &coord,
                                     &py_width, &py_offset, &relative))
        return NULL;
    RobustPath* robustpath = self->robustpath;
    Interpolation* o_buffer =
        (Interpolation*)allocate(sizeof(Interpolation) * 2 * robustpath->num_elements);
    Interpolation* w_buffer = o_buffer + robustpath->num_elements;
    Interpolation* offset = NULL;
    Interpolation* width = NULL;
    if (py_offset != Py_None) {
        offset = o_buffer;
        if (parse_robustpath_offset(*robustpath, py_offset, offset) < 0) {
            free_allocation(o_buffer);
            return NULL;
        }
    }
    if (py_width != Py_None) {
        width = w_buffer;
        if (parse_robustpath_width(*robustpath, py_width, width) < 0) {
            free_allocation(o_buffer);
            return NULL;
        }
    }
    robustpath->vertical(coord, width, offset, relative > 0);
    free_allocation(o_buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_segment(RobustPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:segment", (char**)keywords, &xy, &py_width,
                                     &py_offset, &relative))
        return NULL;
    Vec2 end_point;
    if (parse_point(xy, end_point, "xy") != 0) return NULL;
    RobustPath* robustpath = self->robustpath;
    Interpolation* o_buffer =
        (Interpolation*)allocate(sizeof(Interpolation) * 2 * robustpath->num_elements);
    Interpolation* w_buffer = o_buffer + robustpath->num_elements;
    Interpolation* offset = NULL;
    Interpolation* width = NULL;
    if (py_offset != Py_None) {
        offset = o_buffer;
        if (parse_robustpath_offset(*robustpath, py_offset, offset) < 0) {
            free_allocation(o_buffer);
            return NULL;
        }
    }
    if (py_width != Py_None) {
        width = w_buffer;
        if (parse_robustpath_width(*robustpath, py_width, width) < 0) {
            free_allocation(o_buffer);
            return NULL;
        }
    }
    robustpath->segment(end_point, width, offset, relative > 0);
    free_allocation(o_buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_cubic(RobustPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:cubic", (char**)keywords, &xy, &py_width,
                                     &py_offset, &relative))
        return NULL;
    Array<Vec2> point_array = {};
    if (parse_point_sequence(xy, point_array, "xy") < 0 || point_array.count != 3) {
        point_array.clear();
        PyErr_SetString(PyExc_RuntimeError, "Argument xy must be a sequence of 3 coordinates.");
        return NULL;
    }
    RobustPath* robustpath = self->robustpath;
    Interpolation* o_buffer =
        (Interpolation*)allocate(sizeof(Interpolation) * 2 * robustpath->num_elements);
    Interpolation* w_buffer = o_buffer + robustpath->num_elements;
    Interpolation* offset = NULL;
    Interpolation* width = NULL;
    if (py_offset != Py_None) {
        offset = o_buffer;
        if (parse_robustpath_offset(*robustpath, py_offset, offset) < 0) {
            point_array.clear();
            free_allocation(o_buffer);
            return NULL;
        }
    }
    if (py_width != Py_None) {
        width = w_buffer;
        if (parse_robustpath_width(*robustpath, py_width, width) < 0) {
            point_array.clear();
            free_allocation(o_buffer);
            return NULL;
        }
    }
    robustpath->cubic(point_array[0], point_array[1], point_array[2], width, offset, relative > 0);
    point_array.clear();
    free_allocation(o_buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_cubic_smooth(RobustPathObject* self, PyObject* args,
                                                PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:cubic_smooth", (char**)keywords, &xy,
                                     &py_width, &py_offset, &relative))
        return NULL;
    Array<Vec2> point_array = {};
    if (parse_point_sequence(xy, point_array, "xy") < 0 || point_array.count != 2) {
        point_array.clear();
        PyErr_SetString(PyExc_RuntimeError, "Argument xy must be a sequence of 2 coordinates.");
        return NULL;
    }
    RobustPath* robustpath = self->robustpath;
    Interpolation* o_buffer =
        (Interpolation*)allocate(sizeof(Interpolation) * 2 * robustpath->num_elements);
    Interpolation* w_buffer = o_buffer + robustpath->num_elements;
    Interpolation* offset = NULL;
    Interpolation* width = NULL;
    if (py_offset != Py_None) {
        offset = o_buffer;
        if (parse_robustpath_offset(*robustpath, py_offset, offset) < 0) {
            point_array.clear();
            free_allocation(o_buffer);
            return NULL;
        }
    }
    if (py_width != Py_None) {
        width = w_buffer;
        if (parse_robustpath_width(*robustpath, py_width, width) < 0) {
            point_array.clear();
            free_allocation(o_buffer);
            return NULL;
        }
    }
    robustpath->cubic_smooth(point_array[0], point_array[1], width, offset, relative > 0);
    point_array.clear();
    free_allocation(o_buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_quadratic(RobustPathObject* self, PyObject* args,
                                             PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:quadratic", (char**)keywords, &xy,
                                     &py_width, &py_offset, &relative))
        return NULL;
    Array<Vec2> point_array = {};
    if (parse_point_sequence(xy, point_array, "xy") < 0 || point_array.count != 2) {
        point_array.clear();
        PyErr_SetString(PyExc_RuntimeError, "Argument xy must be a sequence of 2 coordinates.");
        return NULL;
    }
    RobustPath* robustpath = self->robustpath;
    Interpolation* o_buffer =
        (Interpolation*)allocate(sizeof(Interpolation) * 2 * robustpath->num_elements);
    Interpolation* w_buffer = o_buffer + robustpath->num_elements;
    Interpolation* offset = NULL;
    Interpolation* width = NULL;
    if (py_offset != Py_None) {
        offset = o_buffer;
        if (parse_robustpath_offset(*robustpath, py_offset, offset) < 0) {
            point_array.clear();
            free_allocation(o_buffer);
            return NULL;
        }
    }
    if (py_width != Py_None) {
        width = w_buffer;
        if (parse_robustpath_width(*robustpath, py_width, width) < 0) {
            point_array.clear();
            free_allocation(o_buffer);
            return NULL;
        }
    }
    robustpath->quadratic(point_array[0], point_array[1], width, offset, relative > 0);
    point_array.clear();
    free_allocation(o_buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_quadratic_smooth(RobustPathObject* self, PyObject* args,
                                                    PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:quadratic_smooth", (char**)keywords, &xy,
                                     &py_width, &py_offset, &relative))
        return NULL;
    Vec2 end_point;
    if (parse_point(xy, end_point, "xy") != 0) return NULL;
    RobustPath* robustpath = self->robustpath;
    Interpolation* o_buffer =
        (Interpolation*)allocate(sizeof(Interpolation) * 2 * robustpath->num_elements);
    Interpolation* w_buffer = o_buffer + robustpath->num_elements;
    Interpolation* offset = NULL;
    Interpolation* width = NULL;
    if (py_offset != Py_None) {
        offset = o_buffer;
        if (parse_robustpath_offset(*robustpath, py_offset, offset) < 0) {
            free_allocation(o_buffer);
            return NULL;
        }
    }
    if (py_width != Py_None) {
        width = w_buffer;
        if (parse_robustpath_width(*robustpath, py_width, width) < 0) {
            free_allocation(o_buffer);
            return NULL;
        }
    }
    robustpath->quadratic_smooth(end_point, width, offset, relative > 0);
    Py_INCREF(self);
    free_allocation(o_buffer);
    return (PyObject*)self;
}

static PyObject* robustpath_object_bezier(RobustPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 0;
    const char* keywords[] = {"xy", "width", "offset", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:bezier", (char**)keywords, &xy, &py_width,
                                     &py_offset, &relative))
        return NULL;
    Array<Vec2> point_array = {};
    if (parse_point_sequence(xy, point_array, "xy") < 0 || point_array.count < 1) {
        point_array.clear();
        PyErr_SetString(PyExc_RuntimeError, "Argument xy must be a sequence of coordinates.");
        return NULL;
    }
    RobustPath* robustpath = self->robustpath;
    Interpolation* o_buffer =
        (Interpolation*)allocate(sizeof(Interpolation) * 2 * robustpath->num_elements);
    Interpolation* w_buffer = o_buffer + robustpath->num_elements;
    Interpolation* offset = NULL;
    Interpolation* width = NULL;
    if (py_offset != Py_None) {
        offset = o_buffer;
        if (parse_robustpath_offset(*robustpath, py_offset, offset) < 0) {
            point_array.clear();
            free_allocation(o_buffer);
            return NULL;
        }
    }
    if (py_width != Py_None) {
        width = w_buffer;
        if (parse_robustpath_width(*robustpath, py_width, width) < 0) {
            point_array.clear();
            free_allocation(o_buffer);
            return NULL;
        }
    }
    robustpath->bezier(point_array, width, offset, relative > 0);
    point_array.clear();
    free_allocation(o_buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_intepolation(RobustPathObject* self, PyObject* args,
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

    RobustPath* robustpath = self->robustpath;
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

    Interpolation* o_buffer =
        (Interpolation*)allocate(sizeof(Interpolation) * 2 * robustpath->num_elements);
    Interpolation* w_buffer = o_buffer + robustpath->num_elements;
    Interpolation* offset = NULL;
    Interpolation* width = NULL;
    if (py_offset != Py_None) {
        offset = o_buffer;
        if (parse_robustpath_offset(*robustpath, py_offset, offset) < 0) {
            point_array.clear();
            free_allocation(tension);
            free_allocation(o_buffer);
            return NULL;
        }
    }
    if (py_width != Py_None) {
        width = w_buffer;
        if (parse_robustpath_width(*robustpath, py_width, width) < 0) {
            point_array.clear();
            free_allocation(tension);
            free_allocation(o_buffer);
            return NULL;
        }
    }

    robustpath->interpolation(point_array, angles, angle_constraints, tension, initial_curl,
                              final_curl, cycle > 0, width, offset, relative > 0);

    point_array.clear();
    free_allocation(tension);
    free_allocation(o_buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_arc(RobustPathObject* self, PyObject* args, PyObject* kwds) {
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
    RobustPath* robustpath = self->robustpath;
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
    Interpolation* o_buffer =
        (Interpolation*)allocate(sizeof(Interpolation) * 2 * robustpath->num_elements);
    Interpolation* w_buffer = o_buffer + robustpath->num_elements;
    Interpolation* offset = NULL;
    Interpolation* width = NULL;
    if (py_offset != Py_None) {
        offset = o_buffer;
        if (parse_robustpath_offset(*robustpath, py_offset, offset) < 0) {
            free_allocation(o_buffer);
            return NULL;
        }
    }
    if (py_width != Py_None) {
        width = w_buffer;
        if (parse_robustpath_width(*robustpath, py_width, width) < 0) {
            free_allocation(o_buffer);
            return NULL;
        }
    }

    if (radius_x <= 0 || radius_y <= 0) {
        PyErr_SetString(PyExc_ValueError, "Arc radius must be positive.");
        free_allocation(o_buffer);
        return NULL;
    }

    robustpath->arc(radius_x, radius_y, initial_angle, final_angle, rotation, width, offset);
    free_allocation(o_buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_turn(RobustPathObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    double radius;
    double angle;
    const char* keywords[] = {"radius", "angle", "width", "offset", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dd|OO:turn", (char**)keywords, &radius, &angle,
                                     &py_width, &py_offset))
        return NULL;
    RobustPath* robustpath = self->robustpath;
    Interpolation* o_buffer =
        (Interpolation*)allocate(sizeof(Interpolation) * 2 * robustpath->num_elements);
    Interpolation* w_buffer = o_buffer + robustpath->num_elements;
    Interpolation* offset = NULL;
    Interpolation* width = NULL;
    if (py_offset != Py_None) {
        offset = o_buffer;
        if (parse_robustpath_offset(*robustpath, py_offset, offset) < 0) {
            free_allocation(o_buffer);
            return NULL;
        }
    }
    if (py_width != Py_None) {
        width = w_buffer;
        if (parse_robustpath_width(*robustpath, py_width, width) < 0) {
            free_allocation(o_buffer);
            return NULL;
        }
    }
    if (radius <= 0) {
        PyErr_SetString(PyExc_ValueError, "Turn radius must be positive.");
        free_allocation(o_buffer);
        return NULL;
    }
    robustpath->turn(radius, angle, width, offset);
    free_allocation(o_buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_parametric(RobustPathObject* self, PyObject* args,
                                              PyObject* kwds) {
    PyObject* py_function;
    PyObject* py_gradient = Py_None;
    PyObject* py_width = Py_None;
    PyObject* py_offset = Py_None;
    int relative = 1;
    const char* keywords[] = {"path_function", "path_gradient", "width",
                              "offset",        "relative",      NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOOp:parametric", (char**)keywords,
                                     &py_function, &py_gradient, &py_width, &py_offset, &relative))
        return NULL;
    if (!PyCallable_Check(py_function)) {
        PyErr_SetString(PyExc_TypeError, "Argument path_function must be callable.");
        return NULL;
    }
    if (py_gradient != Py_None && !PyCallable_Check(py_gradient)) {
        PyErr_SetString(PyExc_TypeError, "Argument path_gradient must be callable.");
        return NULL;
    }
    RobustPath* robustpath = self->robustpath;
    Interpolation* o_buffer =
        (Interpolation*)allocate(sizeof(Interpolation) * 2 * robustpath->num_elements);
    Interpolation* w_buffer = o_buffer + robustpath->num_elements;
    Interpolation* offset = NULL;
    Interpolation* width = NULL;
    if (py_offset != Py_None) {
        offset = o_buffer;
        if (parse_robustpath_offset(*robustpath, py_offset, offset) < 0) {
            free_allocation(o_buffer);
            return NULL;
        }
    }
    if (py_width != Py_None) {
        width = w_buffer;
        if (parse_robustpath_width(*robustpath, py_width, width) < 0) {
            free_allocation(o_buffer);
            return NULL;
        }
    }
    Py_INCREF(py_function);
    if (py_gradient == Py_None) {
        robustpath->parametric((ParametricVec2)eval_parametric_vec2, (void*)py_function, NULL, NULL,
                               width, offset, relative > 0);
    } else {
        Py_INCREF(py_gradient);
        robustpath->parametric((ParametricVec2)eval_parametric_vec2, (void*)py_function,
                               (ParametricVec2)eval_parametric_vec2, (void*)py_gradient, width,
                               offset, relative > 0);
    }
    free_allocation(o_buffer);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_commands(RobustPathObject* self, PyObject* args) {
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
    uint64_t processed = self->robustpath->commands(instructions, instr_size);
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

static PyObject* robustpath_object_translate(RobustPathObject* self, PyObject* args) {
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
    self->robustpath->translate(v);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_scale(RobustPathObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"s", "center", NULL};
    double scale = 0;
    Vec2 center = {0, 0};
    PyObject* center_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|O:scale", (char**)keywords, &scale,
                                     &center_obj))
        return NULL;
    if (parse_point(center_obj, center, "center") < 0) return NULL;
    self->robustpath->scale(scale, center);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_mirror(RobustPathObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"p1", "p2", NULL};
    Vec2 p1;
    Vec2 p2 = {0, 0};
    PyObject* p1_obj = NULL;
    PyObject* p2_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O:mirror", (char**)keywords, &p1_obj, &p2_obj))
        return NULL;
    if (parse_point(p1_obj, p1, "p1") < 0) return NULL;
    if (parse_point(p2_obj, p2, "p2") < 0) return NULL;
    self->robustpath->mirror(p1, p2);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_rotate(RobustPathObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"angle", "center", NULL};
    double angle;
    Vec2 center = {0, 0};
    PyObject* center_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|O:rotate", (char**)keywords, &angle,
                                     &center_obj))
        return NULL;
    if (parse_point(center_obj, center, "center") < 0) return NULL;
    self->robustpath->rotate(angle, center);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_apply_repetition(RobustPathObject* self, PyObject*) {
    Array<RobustPath*> array = {};
    self->robustpath->apply_repetition(array);
    PyObject* result = PyList_New(array.count);
    for (uint64_t i = 0; i < array.count; i++) {
        RobustPathObject* obj = PyObject_New(RobustPathObject, &robustpath_object_type);
        obj = (RobustPathObject*)PyObject_Init((PyObject*)obj, &robustpath_object_type);
        obj->robustpath = array[i];
        array[i]->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }
    array.clear();
    return result;
}

static PyObject* robustpath_object_set_property(RobustPathObject* self, PyObject* args) {
    if (!parse_property(self->robustpath->properties, args)) return NULL;
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_get_property(RobustPathObject* self, PyObject* args) {
    return build_property(self->robustpath->properties, args);
}

static PyObject* robustpath_object_delete_property(RobustPathObject* self, PyObject* args) {
    char* name;
    if (!PyArg_ParseTuple(args, "s:delete_property", &name)) return NULL;
    remove_property(self->robustpath->properties, name, false);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_set_gds_property(RobustPathObject* self, PyObject* args) {
    uint16_t attribute;
    char* value;
    Py_ssize_t count;
    if (!PyArg_ParseTuple(args, "Hs#:set_gds_property", &attribute, &value, &count)) return NULL;
    if (count >= 0)
        set_gds_property(self->robustpath->properties, attribute, value, (uint64_t)count);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* robustpath_object_get_gds_property(RobustPathObject* self, PyObject* args) {
    uint16_t attribute;
    if (!PyArg_ParseTuple(args, "H:get_gds_property", &attribute)) return NULL;
    const PropertyValue* value = get_gds_property(self->robustpath->properties, attribute);
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

static PyObject* robustpath_object_delete_gds_property(RobustPathObject* self, PyObject* args) {
    uint16_t attribute;
    if (!PyArg_ParseTuple(args, "H:delete_gds_property", &attribute)) return NULL;
    remove_gds_property(self->robustpath->properties, attribute);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyMethodDef robustpath_object_methods[] = {
    {"copy", (PyCFunction)robustpath_object_copy, METH_NOARGS, robustpath_object_copy_doc},
    {"__deepcopy__", (PyCFunction)robustpath_object_deepcopy, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_deepcopy_doc},
    {"spine", (PyCFunction)robustpath_object_spine, METH_NOARGS, robustpath_object_spine_doc},
    {"path_spines", (PyCFunction)robustpath_object_path_spines, METH_NOARGS,
     robustpath_object_path_spines_doc},
    {"widths", (PyCFunction)robustpath_object_widths, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_widths_doc},
    {"offsets", (PyCFunction)robustpath_object_offsets, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_offsets_doc},
    {"position", (PyCFunction)robustpath_object_position, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_position_doc},
    {"gradient", (PyCFunction)robustpath_object_gradient, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_gradient_doc},
    {"to_polygons", (PyCFunction)robustpath_object_to_polygons, METH_NOARGS,
     robustpath_object_to_polygons_doc},
    {"set_layers", (PyCFunction)robustpath_object_set_layers, METH_VARARGS,
     robustpath_object_set_layers_doc},
    {"set_datatypes", (PyCFunction)robustpath_object_set_datatypes, METH_VARARGS,
     robustpath_object_set_datatypes_doc},
    {"set_ends", (PyCFunction)robustpath_object_set_ends, METH_VARARGS,
     robustpath_object_set_ends_doc},
    {"horizontal", (PyCFunction)robustpath_object_horizontal, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_horizontal_doc},
    {"vertical", (PyCFunction)robustpath_object_vertical, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_vertical_doc},
    {"segment", (PyCFunction)robustpath_object_segment, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_segment_doc},
    {"cubic", (PyCFunction)robustpath_object_cubic, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_cubic_doc},
    {"cubic_smooth", (PyCFunction)robustpath_object_cubic_smooth, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_cubic_smooth_doc},
    {"quadratic", (PyCFunction)robustpath_object_quadratic, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_quadratic_doc},
    {"quadratic_smooth", (PyCFunction)robustpath_object_quadratic_smooth,
     METH_VARARGS | METH_KEYWORDS, robustpath_object_quadratic_smooth_doc},
    {"bezier", (PyCFunction)robustpath_object_bezier, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_bezier_doc},
    {"interpolation", (PyCFunction)robustpath_object_intepolation, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_intepolation_doc},
    {"arc", (PyCFunction)robustpath_object_arc, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_arc_doc},
    {"turn", (PyCFunction)robustpath_object_turn, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_turn_doc},
    {"parametric", (PyCFunction)robustpath_object_parametric, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_parametric_doc},
    {"commands", (PyCFunction)robustpath_object_commands, METH_VARARGS,
     robustpath_object_commands_doc},
    {"translate", (PyCFunction)robustpath_object_translate, METH_VARARGS,
     robustpath_object_translate_doc},
    {"scale", (PyCFunction)robustpath_object_scale, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_scale_doc},
    {"mirror", (PyCFunction)robustpath_object_mirror, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_mirror_doc},
    {"rotate", (PyCFunction)robustpath_object_rotate, METH_VARARGS | METH_KEYWORDS,
     robustpath_object_rotate_doc},
    {"apply_repetition", (PyCFunction)robustpath_object_apply_repetition, METH_NOARGS,
     robustpath_object_apply_repetition_doc},
    {"set_property", (PyCFunction)robustpath_object_set_property, METH_VARARGS,
     object_set_property_doc},
    {"get_property", (PyCFunction)robustpath_object_get_property, METH_VARARGS,
     object_get_property_doc},
    {"delete_property", (PyCFunction)robustpath_object_delete_property, METH_VARARGS,
     object_delete_property_doc},
    {"set_gds_property", (PyCFunction)robustpath_object_set_gds_property, METH_VARARGS,
     object_set_gds_property_doc},
    {"get_gds_property", (PyCFunction)robustpath_object_get_gds_property, METH_VARARGS,
     object_get_gds_property_doc},
    {"delete_gds_property", (PyCFunction)robustpath_object_delete_gds_property, METH_VARARGS,
     object_delete_gds_property_doc},
    {NULL}};

static PyObject* robustpath_object_get_layers(RobustPathObject* self, void*) {
    RobustPath* robustpath = self->robustpath;
    PyObject* result = PyTuple_New(robustpath->num_elements);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return tuple.");
        return NULL;
    }
    for (uint64_t i = 0; i < robustpath->num_elements; i++) {
        PyObject* item = PyLong_FromUnsignedLongLong(get_layer(robustpath->elements[i].tag));
        if (item == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create int from layer");
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* robustpath_object_get_datatypes(RobustPathObject* self, void*) {
    RobustPath* robustpath = self->robustpath;
    PyObject* result = PyTuple_New(robustpath->num_elements);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return tuple.");
        return NULL;
    }
    for (uint64_t i = 0; i < robustpath->num_elements; i++) {
        PyObject* item = PyLong_FromUnsignedLongLong(get_type(robustpath->elements[i].tag));
        if (item == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create int from datatype");
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* robustpath_object_get_num_paths(RobustPathObject* self, void*) {
    return PyLong_FromUnsignedLongLong(self->robustpath->num_elements);
}

static PyObject* robustpath_object_get_size(RobustPathObject* self, void*) {
    return PyLong_FromUnsignedLongLong(self->robustpath->subpath_array.count);
}

static PyObject* robustpath_object_get_ends(RobustPathObject* self, void*) {
    RobustPath* robustpath = self->robustpath;
    PyObject* result = PyTuple_New(robustpath->num_elements);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return tuple.");
        return NULL;
    }
    for (uint64_t i = 0; i < robustpath->num_elements; i++) {
        RobustPathElement* element = robustpath->elements + i;
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

static PyObject* robustpath_object_get_tolerance(RobustPathObject* self, void*) {
    return PyFloat_FromDouble(self->robustpath->tolerance);
}

int robustpath_object_set_tolerance(RobustPathObject* self, PyObject* arg, void*) {
    double tolerance = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert value to float.");
        return -1;
    }
    if (tolerance <= 0) {
        PyErr_SetString(PyExc_ValueError, "Tolerance must be positive.");
        return -1;
    }
    self->robustpath->tolerance = tolerance;
    return 0;
}

static PyObject* robustpath_object_get_max_evals(RobustPathObject* self, void*) {
    return PyLong_FromUnsignedLongLong(self->robustpath->max_evals);
}

int robustpath_object_set_max_evals(RobustPathObject* self, PyObject* arg, void*) {
    uint64_t max_evals = PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert value to unsigned integer.");
        return -1;
    }
    if (max_evals < 1) {
        PyErr_SetString(PyExc_ValueError, "Value must be greater than 0.");
        return -1;
    }
    self->robustpath->max_evals = max_evals;
    return 0;
}

static PyObject* robustpath_object_get_simple_path(RobustPathObject* self, void*) {
    PyObject* result = self->robustpath->simple_path ? Py_True : Py_False;
    Py_INCREF(result);
    return result;
}

int robustpath_object_set_simple_path(RobustPathObject* self, PyObject* arg, void*) {
    self->robustpath->simple_path = PyObject_IsTrue(arg) > 0;
    return 0;
}

static PyObject* robustpath_object_get_scale_width(RobustPathObject* self, void*) {
    PyObject* result = self->robustpath->scale_width ? Py_True : Py_False;
    Py_INCREF(result);
    return result;
}

int robustpath_object_set_scale_width(RobustPathObject* self, PyObject* arg, void*) {
    self->robustpath->scale_width = PyObject_IsTrue(arg) > 0;
    return 0;
}

static PyObject* robustpath_object_get_properties(RobustPathObject* self, void*) {
    return build_properties(self->robustpath->properties);
}

int robustpath_object_set_properties(RobustPathObject* self, PyObject* arg, void*) {
    return parse_properties(self->robustpath->properties, arg);
}

static PyObject* robustpath_object_get_repetition(RobustPathObject* self, void*) {
    RepetitionObject* obj = PyObject_New(RepetitionObject, &repetition_object_type);
    obj = (RepetitionObject*)PyObject_Init((PyObject*)obj, &repetition_object_type);
    obj->repetition.copy_from(self->robustpath->repetition);
    return (PyObject*)obj;
}

int robustpath_object_set_repetition(RobustPathObject* self, PyObject* arg, void*) {
    if (arg == Py_None) {
        self->robustpath->repetition.clear();
        return 0;
    } else if (!RepetitionObject_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Value must be a Repetition object.");
        return -1;
    }
    RepetitionObject* repetition_obj = (RepetitionObject*)arg;
    self->robustpath->repetition.clear();
    self->robustpath->repetition.copy_from(repetition_obj->repetition);
    return 0;
}

static PyGetSetDef robustpath_object_getset[] = {
    {"layers", (getter)robustpath_object_get_layers, NULL, robustpath_object_layers_doc, NULL},
    {"datatypes", (getter)robustpath_object_get_datatypes, NULL, robustpath_object_datatypes_doc,
     NULL},
    {"num_paths", (getter)robustpath_object_get_num_paths, NULL, robustpath_object_num_paths_doc,
     NULL},
    {"size", (getter)robustpath_object_get_size, NULL, robustpath_object_size_doc, NULL},
    {"ends", (getter)robustpath_object_get_ends, NULL, robustpath_object_ends_doc, NULL},
    {"tolerance", (getter)robustpath_object_get_tolerance, (setter)robustpath_object_set_tolerance,
     path_object_tolerance_doc, NULL},
    {"max_evals", (getter)robustpath_object_get_max_evals, (setter)robustpath_object_set_max_evals,
     robustpath_object_max_evals_doc, NULL},
    {"simple_path", (getter)robustpath_object_get_simple_path,
     (setter)robustpath_object_set_simple_path, path_object_simple_path_doc, NULL},
    {"scale_width", (getter)robustpath_object_get_scale_width,
     (setter)robustpath_object_set_scale_width, path_object_scale_width_doc, NULL},
    {"properties", (getter)robustpath_object_get_properties,
     (setter)robustpath_object_set_properties, object_properties_doc, NULL},
    {"repetition", (getter)robustpath_object_get_repetition,
     (setter)robustpath_object_set_repetition, object_repetition_doc, NULL},
    {NULL}};
