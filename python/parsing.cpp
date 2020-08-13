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
        PyErr_Format(PyExc_RuntimeError, "Unable to get first item from %s.");
        return -1;
    }
    v.x = PyFloat_AsDouble(coord);
    Py_DECREF(coord);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_RuntimeError, "Error parsing first number from %s.");
        return -1;
    }

    coord = PySequence_ITEM(point, 1);
    if (!coord) {
        PyErr_Format(PyExc_RuntimeError, "Unable to get second item from %s.");
        return -1;
    }
    v.y = PyFloat_AsDouble(coord);
    Py_DECREF(coord);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_RuntimeError, "Error parsing second number from %s.");
        return -1;
    }

    return 0;
}

static Py_ssize_t parse_point_sequence(PyObject* py_polygon, bool fix_orientation,
                                       Array<Vec2>& dest, const char* name) {
    if (!PySequence_Check(py_polygon)) {
        PyErr_Format(PyExc_TypeError, "Argument %s must be a sequence of points.", name);
        return -1;
    }

    const Py_ssize_t len = PySequence_Length(py_polygon);
    dest.size = 0;
    dest.ensure_slots(len);

    double orientation = 0;
    Vec2* p = dest.items;
    Vec2 v0 = {0};
    Vec2 v1 = {0};
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
        if (fix_orientation) {
            if (j == 0)
                v0 = *p;
            else if (j == 1)
                v1 = *p - v0;
            else {
                Vec2 v2 = *p - v0;
                orientation += v1.cross(v2);
                v1 = v2;
            }
        }
        p++;
    }
    dest.size = len;

    if (fix_orientation && orientation < 0) {
        Py_ssize_t i, j;
        for (i = 0, j = len - 1; i < len / 2; i++, j--) {
            Vec2 p = dest[i];
            dest[i] = dest[j];
            dest[j] = p;
        }
    }
    return len;
}

static double* parse_sequence_double(PyObject* sequence, int64_t& len, const char* name) {
    len = PySequence_Length(sequence);
    if (len <= 0) {
        PyErr_Format(PyExc_RuntimeError,
                     "Argument %s is a sequence with invalid length (%" PRId64 ").", name, len);
        return NULL;
    }
    double* values = (double*)malloc(sizeof(double) * len);
    double* v = values;
    for (Py_ssize_t j = 0; j < len; j++) {
        PyObject* item = PySequence_ITEM(sequence, j);
        *v++ = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            free(values);
            PyErr_Format(PyExc_RuntimeError, "Unable to convert item %" PRId64 " in %s to float.",
                         j, name);
            return NULL;
        }
    }
    return values;
}

// static int64_t* parse_sequence_int(PyObject* sequence, Py_ssize_t& len) {
//     len = PySequence_Length(sequence);
//     if (len <= 0) {
//         PyErr_SetString(PyExc_TypeError, "Sequence with invalid length.");
//         return NULL;
//     }
//     int64_t* values = (int64_t*)malloc(sizeof(int64_t) * len);
//     int64_t* v = values;
//     for (Py_ssize_t j = 0; j < len; j++) {
//         PyObject* item = PySequence_ITEM(sequence, j);
//         if (item == NULL) {
//             PyErr_SetString(PyExc_RuntimeError, "Unable to get item from sequence.");
//             return NULL;
//         }
//         *v++ = PyLong_AsLong(item);
//         Py_DECREF(item);
//         if (PyErr_Occurred()) {
//             free(values);
//             PyErr_SetString(PyExc_TypeError, "Unable to convert sequence value to int.");
//             return NULL;
//         }
//     }
//     return values;
// }

int update_style(PyObject* dict, StyleMap& map, const char* name) {
    // TODO: buffer overflow error.
    char buffer[4096];

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

        buffer[0] = 0;
        Py_ssize_t free = COUNT(buffer) - 1;
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
            const char* value_str = PyUnicode_AsUTF8AndSize(value, &value_len);
            if (!key_str || !value_str || free < key_len + value_len + 2) break;

            strcat(buffer, key_str);
            strcat(buffer, ":");
            strcat(buffer, value_str);
            strcat(buffer, ";");
            free -= key_len + value_len + 2;
        }

        map.set(layer, type, buffer);
    }
    return 0;
}
