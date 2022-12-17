/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* curve_object_str(CurveObject* self) {
    char buffer[GDSTK_PRINT_BUFFER_COUNT];
    snprintf(buffer, COUNT(buffer), "Curve with %" PRIu64 " points",
             self->curve->point_array.count);
    return PyUnicode_FromString(buffer);
}

static void curve_object_dealloc(CurveObject* self) {
    if (self->curve) {
        self->curve->clear();
        free_allocation(self->curve);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int curve_object_init(CurveObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy = NULL;
    double tolerance = 0.01;
    const char* keywords[] = {"xy", "tolerance", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|d:Curve", (char**)keywords, &xy, &tolerance))
        return -1;
    if (tolerance <= 0) {
        PyErr_SetString(PyExc_ValueError, "Tolerance must be positive.");
        return -1;
    }
    Vec2 v;
    if (parse_point(xy, v, "xy") != 0) return -1;
    if (self->curve) {
        self->curve->clear();
    } else {
        self->curve = (Curve*)allocate_clear(sizeof(Curve));
    }
    Curve* curve = self->curve;
    curve->tolerance = tolerance;
    curve->append(v);
    curve->owner = self;
    return 0;
}

static PyObject* curve_object_points(CurveObject* self, PyObject*) {
    const Curve* curve = self->curve;
    npy_intp dims[] = {(npy_intp)curve->point_array.count, 2};
    if (curve->closed()) dims[0] -= 1;

    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_MemoryError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    memcpy(data, curve->point_array.items, sizeof(double) * dims[0] * 2);
    return (PyObject*)result;
}

static PyObject* curve_object_horizontal(CurveObject* self, PyObject* args, PyObject* kwds) {
    PyObject* x;
    int relative = 0;
    const char* keywords[] = {"x", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p:horizontal", (char**)keywords, &x, &relative))
        return NULL;
    if (PySequence_Check(x)) {
        Array<double> points = {};
        if (parse_double_sequence(x, points, "x") < 0) return NULL;
        self->curve->horizontal(points, relative > 0);
        points.clear();
    } else {
        double point = PyFloat_AsDouble(x);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Unable to convert first argument to float.");
            return NULL;
        }
        self->curve->horizontal(point, relative > 0);
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* curve_object_vertical(CurveObject* self, PyObject* args, PyObject* kwds) {
    PyObject* y;
    int relative = 0;
    const char* keywords[] = {"y", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p:vertical", (char**)keywords, &y, &relative))
        return NULL;
    if (PySequence_Check(y)) {
        Array<double> points = {};
        if (parse_double_sequence(y, points, "y") < 0) return NULL;
        self->curve->vertical(points, relative > 0);
        points.clear();
    } else {
        double point = PyFloat_AsDouble(y);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Unable to convert first argument to float.");
            return NULL;
        }
        self->curve->vertical(point, relative > 0);
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* curve_object_segment(CurveObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    int relative = 0;
    const char* keywords[] = {"xy", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p:segment", (char**)keywords, &xy, &relative))
        return NULL;
    Vec2 point;
    if (parse_point(xy, point, "xy") == 0) {
        self->curve->segment(point, relative > 0);
    } else {
        PyErr_Clear();
        Array<Vec2> array = {};
        if (parse_point_sequence(xy, array, "xy") < 0) {
            array.clear();
            return NULL;
        }
        self->curve->segment(array, relative > 0);
        array.clear();
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* curve_object_cubic(CurveObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    int relative = 0;
    const char* keywords[] = {"xy", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p:cubic", (char**)keywords, &xy, &relative))
        return NULL;
    Array<Vec2> array = {};
    if (parse_point_sequence(xy, array, "xy") < 0 || array.count < 3) {
        array.clear();
        PyErr_SetString(PyExc_RuntimeError,
                        "Argument xy must be a sequence of at least 3 coordinates.");
        return NULL;
    }
    self->curve->cubic(array, relative > 0);
    array.clear();
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* curve_object_cubic_smooth(CurveObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    int relative = 0;
    const char* keywords[] = {"xy", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p:cubic_smooth", (char**)keywords, &xy,
                                     &relative))
        return NULL;
    Array<Vec2> array = {};
    if (parse_point_sequence(xy, array, "xy") < 0 || array.count < 2) {
        array.clear();
        PyErr_SetString(PyExc_RuntimeError,
                        "Argument xy must be a sequence of at least 2 coordinates.");
        return NULL;
    }
    self->curve->cubic_smooth(array, relative > 0);
    array.clear();
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* curve_object_quadratic(CurveObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    int relative = 0;
    const char* keywords[] = {"xy", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p:quadratic", (char**)keywords, &xy, &relative))
        return NULL;
    Array<Vec2> array = {};
    if (parse_point_sequence(xy, array, "xy") < 0 || array.count < 2) {
        array.clear();
        PyErr_SetString(PyExc_RuntimeError,
                        "Argument xy must be a sequence of at least 2 coordinates.");
        return NULL;
    }
    self->curve->quadratic(array, relative > 0);
    array.clear();
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* curve_object_quadratic_smooth(CurveObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    int relative = 0;
    const char* keywords[] = {"xy", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p:quadratic_smooth", (char**)keywords, &xy,
                                     &relative))
        return NULL;
    Vec2 point;
    if (parse_point(xy, point, "xy") == 0) {
        self->curve->quadratic_smooth(point, relative > 0);
    } else {
        Array<Vec2> array = {};
        PyErr_Clear();
        if (parse_point_sequence(xy, array, "xy") < 0) {
            array.clear();
            return NULL;
        }
        self->curve->quadratic_smooth(array, relative > 0);
        array.clear();
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* curve_object_bezier(CurveObject* self, PyObject* args, PyObject* kwds) {
    PyObject* xy;
    int relative = 0;
    const char* keywords[] = {"xy", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p:bezier", (char**)keywords, &xy, &relative))
        return NULL;
    Array<Vec2> array = {};
    if (parse_point_sequence(xy, array, "xy") < 0) {
        array.clear();
        return NULL;
    }
    self->curve->bezier(array, relative > 0);
    array.clear();
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* curve_object_interpolation(CurveObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_points = NULL;
    PyObject* py_angles = Py_None;
    PyObject* py_tension_in = NULL;
    PyObject* py_tension_out = NULL;
    double initial_curl = 1;
    double final_curl = 1;
    int cycle = 0;
    int relative = 0;
    Vec2* tension;
    double* angles;
    bool* angle_constraints;
    const char* keywords[] = {"points",     "angles", "tension_in", "tension_out", "initial_curl",
                              "final_curl", "cycle",  "relative",   NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOOddpp:interpolation", (char**)keywords,
                                     &py_points, &py_angles, &py_tension_in, &py_tension_out,
                                     &initial_curl, &final_curl, &cycle, &relative))
        return NULL;

    Array<Vec2> array = {};
    if (parse_point_sequence(py_points, array, "points") < 0) {
        array.clear();
        return NULL;
    }
    const uint64_t count = array.count;

    tension = (Vec2*)allocate((sizeof(Vec2) + sizeof(double) + sizeof(bool)) * (count + 1));
    angles = (double*)(tension + (count + 1));
    angle_constraints = (bool*)(angles + (count + 1));

    if (py_angles == Py_None) {
        memset(angle_constraints, 0, sizeof(bool) * (count + 1));
    } else {
        if ((uint64_t)PySequence_Length(py_angles) != count + 1) {
            free_allocation(tension);
            array.clear();
            PyErr_SetString(
                PyExc_TypeError,
                "Argument angles must be None or a sequence with count len(points) + 1.");
            return NULL;
        }
        for (uint64_t i = 0; i < count + 1; i++) {
            PyObject* item = PySequence_ITEM(py_angles, i);
            if (!item) {
                free_allocation(tension);
                array.clear();
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
                    array.clear();
                    Py_DECREF(item);
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to convert angles[%" PRIu64 "] to float.", i);
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
            array.clear();
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert tension_in to float.");
            return NULL;
        }
        Vec2* t = tension;
        for (uint64_t i = 0; i < count + 1; i++) (t++)->u = t_in;
    } else {
        if ((uint64_t)PySequence_Length(py_tension_in) != count + 1) {
            free_allocation(tension);
            array.clear();
            PyErr_SetString(
                PyExc_TypeError,
                "Argument tension_in must be a number or a sequence with count len(points) + 1.");
            return NULL;
        }
        for (uint64_t i = 0; i < count + 1; i++) {
            PyObject* item = PySequence_ITEM(py_tension_in, i);
            if (!item) {
                free_allocation(tension);
                array.clear();
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get item %" PRIu64 " from tension_in sequence.", i);
                return NULL;
            }
            tension[i].u = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                free_allocation(tension);
                array.clear();
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
            array.clear();
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert tension_out to float.");
            return NULL;
        }
        Vec2* t = tension;
        for (uint64_t i = 0; i < count + 1; i++) (t++)->v = t_out;
    } else {
        if ((uint64_t)PySequence_Length(py_tension_out) != count + 1) {
            free_allocation(tension);
            array.clear();
            PyErr_SetString(
                PyExc_TypeError,
                "Argument tension_out must be a number or a sequence with count len(points) + 1.");
            return NULL;
        }
        for (uint64_t i = 0; i < count + 1; i++) {
            PyObject* item = PySequence_ITEM(py_tension_out, i);
            if (!item) {
                free_allocation(tension);
                array.clear();
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get item %" PRIu64 " from tension_out sequence.", i);
                return NULL;
            }
            tension[i].v = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                free_allocation(tension);
                array.clear();
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to convert tension_out[%" PRIu64 "] to float.", i);
                return NULL;
            }
        }
    }

    self->curve->interpolation(array, angles, angle_constraints, tension, initial_curl, final_curl,
                               cycle > 0, relative > 0);

    array.clear();
    free_allocation(tension);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* curve_object_arc(CurveObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_radius;
    double radius_x;
    double radius_y;
    double initial_angle;
    double final_angle;
    double rotation = 0;
    const char* keywords[] = {"radius", "initial_angle", "final_angle", "rotation", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Odd|d:arc", (char**)keywords, &py_radius,
                                     &initial_angle, &final_angle, &rotation))
        return NULL;

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

    if (radius_x <= 0 || radius_y <= 0) {
        PyErr_SetString(PyExc_ValueError, "Arc radius must be positive.");
        return NULL;
    }

    self->curve->arc(radius_x, radius_y, initial_angle, final_angle, rotation);

    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* curve_object_turn(CurveObject* self, PyObject* args, PyObject* kwds) {
    double radius;
    double angle;
    const char* keywords[] = {"radius", "angle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dd:turn", (char**)keywords, &radius, &angle))
        return NULL;
    if (radius <= 0) {
        PyErr_SetString(PyExc_ValueError, "Turn radius must be positive.");
        return NULL;
    }
    self->curve->turn(radius, angle);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* curve_object_parametric(CurveObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_function;
    int relative = 1;
    const char* keywords[] = {"curve_function", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p:parametric", (char**)keywords, &py_function,
                                     &relative))
        return NULL;
    if (!PyCallable_Check(py_function)) {
        PyErr_SetString(PyExc_TypeError, "Argument curve_function must be callable.");
        return NULL;
    }
    Py_INCREF(py_function);
    self->curve->parametric((ParametricVec2)eval_parametric_vec2, (void*)py_function, relative > 0);
    Py_DECREF(py_function);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* curve_object_commands(CurveObject* self, PyObject* args) {
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
                                "Curve instructions must be single characters or numbers.");
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
                                "Curve instructions must be single characters or numbers.");
                free_allocation(instructions);
                return NULL;
            }
        }
    }

    uint64_t instr_size = instr - instructions;
    uint64_t processed = self->curve->commands(instructions, instr_size);
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

static PyMethodDef curve_object_methods[] = {
    {"points", (PyCFunction)curve_object_points, METH_NOARGS, curve_object_points_doc},
    {"horizontal", (PyCFunction)curve_object_horizontal, METH_VARARGS | METH_KEYWORDS,
     curve_object_horizontal_doc},
    {"vertical", (PyCFunction)curve_object_vertical, METH_VARARGS | METH_KEYWORDS,
     curve_object_vertical_doc},
    {"segment", (PyCFunction)curve_object_segment, METH_VARARGS | METH_KEYWORDS,
     curve_object_segment_doc},
    {"cubic", (PyCFunction)curve_object_cubic, METH_VARARGS | METH_KEYWORDS,
     curve_object_cubic_doc},
    {"cubic_smooth", (PyCFunction)curve_object_cubic_smooth, METH_VARARGS | METH_KEYWORDS,
     curve_object_cubic_smooth_doc},
    {"quadratic", (PyCFunction)curve_object_quadratic, METH_VARARGS | METH_KEYWORDS,
     curve_object_quadratic_doc},
    {"quadratic_smooth", (PyCFunction)curve_object_quadratic_smooth, METH_VARARGS | METH_KEYWORDS,
     curve_object_quadratic_smooth_doc},
    {"bezier", (PyCFunction)curve_object_bezier, METH_VARARGS | METH_KEYWORDS,
     curve_object_bezier_doc},
    {"interpolation", (PyCFunction)curve_object_interpolation, METH_VARARGS | METH_KEYWORDS,
     curve_object_interpolation_doc},
    {"arc", (PyCFunction)curve_object_arc, METH_VARARGS | METH_KEYWORDS, curve_object_arc_doc},
    {"turn", (PyCFunction)curve_object_turn, METH_VARARGS | METH_KEYWORDS, curve_object_turn_doc},
    {"parametric", (PyCFunction)curve_object_parametric, METH_VARARGS | METH_KEYWORDS,
     curve_object_parametric_doc},
    {"commands", (PyCFunction)curve_object_commands, METH_VARARGS, curve_object_commands_doc},
    {NULL}};

PyObject* curve_object_get_tolerance(CurveObject* self, void*) {
    return PyFloat_FromDouble(self->curve->tolerance);
}

int curve_object_set_tolerance(CurveObject* self, PyObject* arg, void*) {
    double tolerance = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert tolerance to float.");
        return -1;
    }
    if (tolerance <= 0) {
        PyErr_SetString(PyExc_ValueError, "Tolerance must be positive.");
        return -1;
    }
    self->curve->tolerance = tolerance;
    return 0;
}

static PyGetSetDef curve_object_getset[] = {
    {"tolerance", (getter)curve_object_get_tolerance, (setter)curve_object_set_tolerance,
     curve_object_tolerance_doc, NULL},
    {NULL}};
