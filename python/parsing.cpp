/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static int parse_point(PyObject* point, Vec2& v, const char* name) {
    if (!point) return 0;

    if (PyComplex_Check(point)) {
        v.x = PyComplex_RealAsDouble(point);
        v.y = PyComplex_ImagAsDouble(point);
        return 0;
    }

    if (!PySequence_Check(point) || PySequence_Length(point) != 2) {
        PyErr_Format(PyExc_TypeError,
                     "Argument %s must be a sequence of 2 numbers or a complex value.", name);
        return -1;
    }

    PyObject* coord = PySequence_ITEM(point, 0);
    if (!coord) {
        PyErr_Format(PyExc_RuntimeError, "Unable to get first item from %s.", name);
        return -1;
    }
    v.x = PyFloat_AsDouble(coord);
    Py_DECREF(coord);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_RuntimeError, "Error parsing first number from %s.", name);
        return -1;
    }

    coord = PySequence_ITEM(point, 1);
    if (!coord) {
        PyErr_Format(PyExc_RuntimeError, "Unable to get second item from %s.", name);
        return -1;
    }
    v.y = PyFloat_AsDouble(coord);
    Py_DECREF(coord);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_RuntimeError, "Error parsing second number from %s.", name);
        return -1;
    }

    return 0;
}

static int64_t parse_point_sequence(PyObject* py_polygon, Array<Vec2>& dest, const char* name) {
    if (!PySequence_Check(py_polygon)) {
        PyErr_Format(PyExc_TypeError, "Argument %s must be a sequence of points.", name);
        return -1;
    }

    const int64_t len = PySequence_Length(py_polygon);
    dest.ensure_slots(len);
    Vec2* p = dest.items;
    for (Py_ssize_t j = 0; j < len; ++j) {
        PyObject* py_point = PySequence_ITEM(py_polygon, j);
        if (py_point == NULL || parse_point(py_point, *p, "") != 0) {
            Py_XDECREF(py_point);
            PyErr_Format(PyExc_TypeError,
                         "Item %" PRId64
                         " in %s must be a sequence of 2 numbers or a complex value.",
                         j, name);
            return -1;
        }
        Py_DECREF(py_point);
        p++;
    }
    dest.size = len;
    return len;
}

static int64_t parse_double_sequence(PyObject* sequence, Array<double>& dest, const char* name) {
    const int64_t len = PySequence_Length(sequence);
    if (len <= 0) {
        PyErr_Format(PyExc_RuntimeError,
                     "Argument %s is a sequence with invalid length (%" PRId64 ").", name, len);
        return -1;
    }
    dest.ensure_slots(len);
    double* v = dest.items;
    for (Py_ssize_t j = 0; j < len; j++) {
        PyObject* item = PySequence_ITEM(sequence, j);
        *v++ = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_Format(PyExc_RuntimeError, "Unable to convert item %" PRId64 " in %s to float.",
                         j, name);
            return -1;
        }
    }
    dest.size = len;
    return len;
}

// polygon_array should be zero-initialized
static int64_t parse_polygons(PyObject* py_polygons, Array<Polygon*>& polygon_array,
                              const char* name) {
    if (PolygonObject_Check(py_polygons)) {
        Polygon* polygon = (Polygon*)allocate_clear(sizeof(Polygon));
        polygon->copy_from(*((PolygonObject*)py_polygons)->polygon);
        polygon_array.append(polygon);
    } else if (FlexPathObject_Check(py_polygons)) {
        ((FlexPathObject*)py_polygons)->flexpath->to_polygons(polygon_array);
    } else if (RobustPathObject_Check(py_polygons)) {
        ((RobustPathObject*)py_polygons)->robustpath->to_polygons(polygon_array);
    } else if (ReferenceObject_Check(py_polygons)) {
        ((ReferenceObject*)py_polygons)->reference->polygons(true, true, -1, polygon_array);
    } else if (PySequence_Check(py_polygons)) {
        for (int64_t i = PySequence_Length(py_polygons) - 1; i >= 0; i--) {
            PyObject* arg = PySequence_ITEM(py_polygons, i);
            if (arg == NULL) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to retrieve item %" PRId64 " from sequence %s.", i, name);
                for (int64_t j = polygon_array.size - 1; j >= 0; j--) {
                    polygon_array[j]->clear();
                    free_allocation(polygon_array[j]);
                }
                polygon_array.clear();
                return -1;
            }
            if (PolygonObject_Check(arg)) {
                Polygon* polygon = (Polygon*)allocate_clear(sizeof(Polygon));
                polygon->copy_from(*((PolygonObject*)arg)->polygon);
                polygon_array.append(polygon);
            } else if (FlexPathObject_Check(arg)) {
                ((FlexPathObject*)arg)->flexpath->to_polygons(polygon_array);
            } else if (RobustPathObject_Check(arg)) {
                ((RobustPathObject*)arg)->robustpath->to_polygons(polygon_array);
            } else if (ReferenceObject_Check(arg)) {
                ((ReferenceObject*)arg)->reference->polygons(true, true, -1, polygon_array);
            } else {
                Polygon* polygon = (Polygon*)allocate_clear(sizeof(Polygon));
                if (parse_point_sequence(arg, polygon->point_array, "") < 0) {
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to parse item %" PRId64 " from sequence %s.", i, name);
                    return -1;
                }
                polygon_array.append(polygon);
            }
            Py_DECREF(arg);
        }
    } else {
        PyErr_Format(
            PyExc_TypeError,
            "Argument %s must be a Polygon, FlexPath, RobustPath, References. "
            "It can also be a sequence where each item is one of those or a sequence of points.",
            name);
        return -1;
    }
    return polygon_array.size;
}

int update_style(PyObject* dict, StyleMap& map, const char* name) {
    Array<char> buffer = {0};
    buffer.ensure_slots(4096);

    if (!PyDict_Check(dict)) {
        PyErr_Format(PyExc_TypeError, "Argument %s must be a dictionary.", name);
        return -1;
    }

    PyObject* lttuple;
    PyObject* css_dict;
    Py_ssize_t j = 0;
    while (PyDict_Next(dict, &j, &lttuple, &css_dict)) {
        if (!(PyDict_Check(css_dict) && PyTuple_Check(lttuple) && PyTuple_GET_SIZE(lttuple) == 2)) {
            PyErr_Format(PyExc_TypeError,
                         "Item %" PRId64
                         " in %s must have a 2-element tuple as key and a dictionary as value.",
                         j, name);
            return -1;
        }

        int16_t layer = (int16_t)PyLong_AsLong(PyTuple_GET_ITEM(lttuple, 0));
        int16_t type = (int16_t)PyLong_AsLong(PyTuple_GET_ITEM(lttuple, 1));
        if (PyErr_Occurred()) {
            PyErr_Format(PyExc_TypeError,
                         "Unable to retrieve layer and type from the key in item %" PRId64
                         " in %s.",
                         j, name);
            return -1;
        }

        buffer.size = 0;
        PyObject* key;
        PyObject* value;
        Py_ssize_t i = 0;
        while (PyDict_Next(css_dict, &i, &key, &value)) {
            if (!(PyUnicode_Check(key) && PyUnicode_Check(value))) {
                PyErr_Format(PyExc_TypeError,
                             "Keys and values in dictionary %" PRId64 " in %s are not strings.", j,
                             name);
                return -1;
            }

            Py_ssize_t key_len = 0;
            Py_ssize_t value_len = 0;
            const char* key_str = PyUnicode_AsUTF8AndSize(key, &key_len);
            if (!key_str) {
                fputs("Unable to load key from string.", stderr);
                break;
            }
            const char* value_str = PyUnicode_AsUTF8AndSize(value, &value_len);
            if (!value_str) {
                fputs("Unable to load value from string.", stderr);
                break;
            }

            buffer.ensure_slots(key_len + value_len + 2);
            memcpy(buffer.items + buffer.size, key_str, key_len);
            buffer.size += key_len;
            buffer.append(':');
            memcpy(buffer.items + buffer.size, value_str, value_len);
            buffer.size += value_len;
            buffer.append(';');
        }
        buffer.append('\0');
        map.set(layer, type, buffer.items);
    }
    buffer.clear();
    return 0;
}
