/*
Copyright 2020 Lucas Heitzmann Gabrielli.
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
    for (int64_t j = 0; j < len; ++j) {
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
    dest.count = len;
    return len;
}

static int64_t parse_double_sequence(PyObject* sequence, Array<double>& dest, const char* name) {
    if (!PySequence_Check(sequence)) {
        PyErr_Format(PyExc_RuntimeError, "Argument %s must be a sequence.", name);
        return -1;
    }
    const int64_t len = PySequence_Length(sequence);
    if (len <= 0) {
        PyErr_Format(PyExc_RuntimeError,
                     "Argument %s is a sequence with invalid length (%" PRIu64 ").", name, len);
        return -1;
    }
    dest.ensure_slots(len);
    double* v = dest.items;
    for (int64_t j = 0; j < len; j++) {
        PyObject* item = PySequence_ITEM(sequence, j);
        *v++ = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_Format(PyExc_RuntimeError, "Unable to convert item %" PRId64 " in %s to float.",
                         j, name);
            return -1;
        }
    }
    dest.count = len;
    return len;
}

/*
static int64_t parse_uint_sequence(PyObject* sequence, Array<uint32_t>& dest, const char* name) {
    if (!PySequence_Check(sequence)) {
        PyErr_Format(PyExc_RuntimeError, "Argument %s must be a sequence.", name);
        return -1;
    }
    const int64_t len = PySequence_Length(sequence);
    if (len < 0) {
        PyErr_Format(PyExc_RuntimeError,
                     "Argument %s is a sequence with invalid length (%" PRIu64 ").", name, len);
        return -1;
    }
    if (len > 0) {
        dest.ensure_slots(len);
        uint32_t* v = dest.items;
        for (int64_t j = 0; j < len; j++) {
            PyObject* item = PySequence_ITEM(sequence, j);
            *v++ = (uint32_t)PyLong_AsUnsignedLong(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to convert item %" PRId64 " in %s to positive integer.", j,
                             name);
                return -1;
            }
        }
        dest.count = len;
    }
    return len;
}
*/

static bool parse_tag(PyObject* py_tag, Tag& tag) {
    if (!PySequence_Check(py_tag) || PySequence_Length(py_tag) != 2) return false;
    PyObject* value = PySequence_ITEM(py_tag, 0);
    if (!value) return false;
    uint32_t layer = (uint32_t)PyLong_AsUnsignedLong(value);
    Py_DECREF(value);
    value = PySequence_ITEM(py_tag, 1);
    if (!value) return false;
    uint32_t type = (uint32_t)PyLong_AsUnsignedLong(value);
    Py_DECREF(value);
    if (PyErr_Occurred()) return false;
    tag = make_tag(layer, type);
    return true;
}

static int64_t parse_tag_sequence(PyObject* iterable, Set<Tag>& dest, const char* name) {
    PyObject* iterator = PyObject_GetIter(iterable);
    if (!iterator) {
        PyErr_Format(PyExc_RuntimeError, "Unable to get an iterator from %s.", name);
        return -1;
    }
    int64_t count = 0;
    PyObject* item;
    while ((item = PyIter_Next(iterator))) {
        Tag tag;
        if (!parse_tag(item, tag)) {
            PyErr_Format(PyExc_TypeError, "Items in argument %s must be a 2-element sequence of non-negative integers (layer, type).",
                         name);
            Py_DECREF(item);
            Py_DECREF(iterator);
            return -1;
        }
        dest.add(tag);
        count++;
    }
    Py_DECREF(iterator);
    return count;
}

// polygon_array should be zero-initialized
static int64_t parse_polygons(PyObject* py_polygons, Array<Polygon*>& polygon_array,
                              const char* name) {
    if (PolygonObject_Check(py_polygons)) {
        Polygon* polygon = (Polygon*)allocate_clear(sizeof(Polygon));
        polygon->copy_from(*((PolygonObject*)py_polygons)->polygon);
        polygon_array.append(polygon);
    } else if (FlexPathObject_Check(py_polygons)) {
        ErrorCode error_code =
            ((FlexPathObject*)py_polygons)->flexpath->to_polygons(false, 0, polygon_array);
        if (return_error(error_code)) {
            for (int64_t j = polygon_array.count - 1; j >= 0; j--) {
                polygon_array[j]->clear();
                free_allocation(polygon_array[j]);
            }
            polygon_array.clear();
            return -1;
        }
    } else if (RobustPathObject_Check(py_polygons)) {
        ErrorCode error_code =
            ((RobustPathObject*)py_polygons)->robustpath->to_polygons(false, 0, polygon_array);
        if (return_error(error_code)) {
            for (int64_t j = polygon_array.count - 1; j >= 0; j--) {
                polygon_array[j]->clear();
                free_allocation(polygon_array[j]);
            }
            polygon_array.clear();
            return -1;
        }
    } else if (ReferenceObject_Check(py_polygons)) {
        ((ReferenceObject*)py_polygons)
            ->reference->get_polygons(true, true, -1, false, 0, polygon_array);
    } else if (PySequence_Check(py_polygons)) {
        for (int64_t i = PySequence_Length(py_polygons) - 1; i >= 0; i--) {
            PyObject* arg = PySequence_ITEM(py_polygons, i);
            if (arg == NULL) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to retrieve item %" PRIu64 " from sequence %s.", i, name);
                for (int64_t j = polygon_array.count - 1; j >= 0; j--) {
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
                ErrorCode error_code =
                    ((FlexPathObject*)arg)->flexpath->to_polygons(false, 0, polygon_array);
                if (return_error(error_code)) {
                    for (int64_t j = polygon_array.count - 1; j >= 0; j--) {
                        polygon_array[j]->clear();
                        free_allocation(polygon_array[j]);
                    }
                    polygon_array.clear();
                    return -1;
                }
            } else if (RobustPathObject_Check(arg)) {
                ErrorCode error_code =
                    ((RobustPathObject*)arg)->robustpath->to_polygons(false, 0, polygon_array);
                if (return_error(error_code)) {
                    for (int64_t j = polygon_array.count - 1; j >= 0; j--) {
                        polygon_array[j]->clear();
                        free_allocation(polygon_array[j]);
                    }
                    polygon_array.clear();
                    return -1;
                }
            } else if (ReferenceObject_Check(arg)) {
                ((ReferenceObject*)arg)
                    ->reference->get_polygons(true, true, -1, false, 0, polygon_array);
            } else {
                Polygon* polygon = (Polygon*)allocate_clear(sizeof(Polygon));
                if (parse_point_sequence(arg, polygon->point_array, "") <= 0) {
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to parse item %" PRIu64 " from sequence %s.", i, name);
                    for (int64_t j = polygon_array.count - 1; j >= 0; j--) {
                        polygon_array[j]->clear();
                        free_allocation(polygon_array[j]);
                    }
                    polygon_array.clear();
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
    return polygon_array.count;
}

static int update_style(PyObject* dict, StyleMap& map, const char* name) {
    Array<char> buffer = {};
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
                         (int64_t)j, name);
            return -1;
        }

        uint32_t layer = (uint32_t)PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(lttuple, 0));
        uint32_t type = (uint32_t)PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(lttuple, 1));
        if (PyErr_Occurred()) {
            PyErr_Format(PyExc_TypeError,
                         "Unable to retrieve layer and type from the key in item %" PRId64
                         " in %s.",
                         (int64_t)j, name);
            return -1;
        }

        buffer.count = 0;
        PyObject* key;
        PyObject* value;
        Py_ssize_t i = 0;
        while (PyDict_Next(css_dict, &i, &key, &value)) {
            if (!(PyUnicode_Check(key) && PyUnicode_Check(value))) {
                PyErr_Format(PyExc_TypeError,
                             "Keys and values in dictionary %" PRId64 " in %s are not strings.",
                             (int64_t)j, name);
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
            memcpy(buffer.items + buffer.count, key_str, key_len);
            buffer.count += key_len;
            buffer.append(':');
            memcpy(buffer.items + buffer.count, value_str, value_len);
            buffer.count += value_len;
            buffer.append(';');
        }
        buffer.append('\0');
        map.set(make_tag(layer, type), buffer.items);
    }
    buffer.clear();
    return 0;
}

static PyObject* build_properties(Property* properties) {
    uint64_t i = 0;
    for (Property* property = properties; property; property = property->next) i++;
    PyObject* result = PyList_New(i);

    i = 0;
    for (Property* property = properties; property; property = property->next) {
        PyObject* name = PyUnicode_FromString(property->name);
        if (!name) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert name to string.");
            Py_DECREF(result);
            return NULL;
        }
        uint64_t j = 0;
        for (PropertyValue* value = property->value; value; value = value->next) j++;
        PyObject* py_property = PyList_New(1 + j);
        PyList_SET_ITEM(result, i++, py_property);
        PyList_SET_ITEM(py_property, 0, name);

        j = 1;
        for (PropertyValue* value = property->value; value; value = value->next) {
            PyObject* py_value = NULL;
            switch (value->type) {
                case PropertyType::UnsignedInteger:
                    py_value = PyLong_FromUnsignedLongLong(value->unsigned_integer);
                    break;
                case PropertyType::Integer:
                    py_value = PyLong_FromLongLong(value->integer);
                    break;
                case PropertyType::Real:
                    py_value = PyFloat_FromDouble(value->real);
                    break;
                case PropertyType::String:
                    py_value =
                        PyBytes_FromStringAndSize((char*)value->bytes, (Py_ssize_t)value->count);
            }
            if (!py_value) {
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert property value to object.");
                Py_DECREF(result);
                return NULL;
            }
            PyList_SET_ITEM(py_property, j++, py_value);
        }
    }
    return result;
}

static PyObject* build_property(Property* properties, PyObject* args) {
    char* name;
    if (!PyArg_ParseTuple(args, "s:get_property", &name)) return NULL;
    PropertyValue* value = get_property(properties, name);
    if (value == NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    uint64_t i = 0;
    for (PropertyValue* v = value; v; v = v->next) i++;
    PyObject* result = PyList_New(i);
    for (i = 0; value; value = value->next) {
        PyObject* py_value = NULL;
        switch (value->type) {
            case PropertyType::UnsignedInteger:
                py_value = PyLong_FromUnsignedLongLong(value->unsigned_integer);
                break;
            case PropertyType::Integer:
                py_value = PyLong_FromLongLong(value->integer);
                break;
            case PropertyType::Real:
                py_value = PyFloat_FromDouble(value->real);
                break;
            case PropertyType::String:
                py_value = PyBytes_FromStringAndSize((char*)value->bytes, (Py_ssize_t)value->count);
        }
        if (!py_value) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert property value to object.");
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i++, py_value);
    }
    return result;
}

static bool add_value(PropertyValue* value, PyObject* item) {
    Py_ssize_t string_len;
    if (PyLong_Check(item)) {
        PyObject* zero = PyLong_FromLong(0);
        if (PyObject_RichCompareBool(item, zero, Py_GE)) {
            value->type = PropertyType::UnsignedInteger;
            value->unsigned_integer = PyLong_AsUnsignedLongLong(item);
        } else {
            value->type = PropertyType::Integer;
            value->integer = PyLong_AsLongLong(item);
        }
        Py_DECREF(zero);
        if (PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        return true;
    } else if (PyFloat_Check(item)) {
        value->type = PropertyType::Real;
        value->real = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        return true;
    } else if (PyUnicode_Check(item)) {
        const char* string = PyUnicode_AsUTF8AndSize(item, &string_len);
        if (!string) return false;
        value->type = PropertyType::String;
        value->count = (uint64_t)string_len;
        value->bytes = (uint8_t*)allocate(string_len);
        memcpy(value->bytes, string, string_len);
        return true;
    } else if (PyBytes_Check(item)) {
        char* string = NULL;
        PyBytes_AsStringAndSize(item, &string, &string_len);
        value->type = PropertyType::String;
        value->count = (uint64_t)string_len;
        value->bytes = (uint8_t*)allocate(string_len);
        memcpy(value->bytes, string, string_len);
        return true;
    }
    return false;
}

static int parse_properties(Property*& properties, PyObject* arg) {
    properties_clear(properties);
    if (!PySequence_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Properties must be a sequence.");
        return -1;
    }
    int64_t count = PySequence_Size(arg);
    if (count < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to get sequence count.");
        return -1;
    }
    for (count--; count >= 0; count--) {
        PyObject* py_property = PySequence_ITEM(arg, count);
        if (!py_property) {
            PyErr_Format(PyExc_RuntimeError, "Unable to get sequence item %" PRId64 ".", count);
            return -1;
        }
        if (!PySequence_Check(py_property)) {
            PyErr_SetString(PyExc_TypeError, "Properties must be sequences of name and values.");
            Py_DECREF(py_property);
            return -1;
        }
        int64_t num_values = PySequence_Size(py_property) - 1;
        if (num_values < 1) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Properties must be a sequence with length 2 or more.");
            Py_DECREF(py_property);
            return -1;
        }

        PyObject* item = PySequence_ITEM(py_property, 0);
        if (!item) {
            PyErr_Format(PyExc_RuntimeError, "Unable to get property %" PRId64 " name.", count);
            Py_DECREF(py_property);
            return -1;
        }
        if (!PyUnicode_Check(item)) {
            PyErr_Format(PyExc_RuntimeError,
                         "First item in property %" PRId64 " must be a string.");
            Py_DECREF(py_property);
            Py_DECREF(item);
            return -1;
        }
        Py_ssize_t string_len = 0;
        const char* name = PyUnicode_AsUTF8AndSize(item, &string_len);
        if (!name) {
            PyErr_Format(PyExc_RuntimeError, "Unable to get name from property %" PRId64 ".",
                         count);
            Py_DECREF(py_property);
            Py_DECREF(item);
            return -1;
        }
        Py_DECREF(item);

        Property* property = (Property*)allocate(sizeof(Property));
        property->name = (char*)allocate(++string_len);
        memcpy(property->name, name, string_len);
        property->value = NULL;
        property->next = properties;
        properties = property;

        for (; num_values >= 1; num_values--) {
            item = PySequence_ITEM(py_property, num_values);
            if (!item) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to get property %" PRId64 " item %" PRId64 ".", count,
                             num_values);
                Py_DECREF(py_property);
                return -1;
            }
            PropertyValue* value = (PropertyValue*)allocate_clear(sizeof(PropertyValue));
            value->next = property->value;
            property->value = value;
            if (!add_value(value, item)) {
                PyErr_Format(PyExc_RuntimeError,
                             "Item %" PRId64 " from property %" PRId64
                             " could not be converted to integer, float, or string.",
                             num_values, count);
                Py_DECREF(item);
                Py_DECREF(py_property);
                return -1;
            }
            Py_DECREF(item);
        }
        Py_DECREF(py_property);
    }
    return 0;
}

static bool parse_property(Property*& properties, PyObject* args) {
    char* name;
    PyObject* py_value;
    if (!PyArg_ParseTuple(args, "sO:set_property", &name, &py_value)) return false;
    Property* property = (Property*)allocate(sizeof(Property));
    property->name = copy_string(name, NULL);
    property->next = properties;
    properties = property;
    property->value = (PropertyValue*)allocate_clear(sizeof(PropertyValue));
    if (add_value(property->value, py_value)) return true;
    if (!PySequence_Check(py_value)) {
        PyErr_SetString(
            PyExc_TypeError,
            "Property value must be integer, float, string, bytes, or sequence of those.");
        return false;
    }
    int64_t count = PySequence_Size(py_value);
    if (count < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to get sequence count.");
        return false;
    } else if (count == 0) {
        PyErr_SetString(PyExc_RuntimeError, "No values found in property sequence.");
        return false;
    }
    for (count--; count >= 0; count--) {
        PyObject* item = PySequence_ITEM(py_value, count);
        if (!item) {
            PyErr_Format(PyExc_RuntimeError, "Unable to get item %" PRId64 ".", count);
            return false;
        }
        if (!add_value(property->value, item)) {
            PyErr_Format(PyExc_RuntimeError,
                         "Item %" PRId64
                         " from could not be converted to integer, float, or string.",
                         count);
            Py_DECREF(item);
            return false;
        }
        Py_DECREF(item);
        if (count > 0) {
            PropertyValue* value = (PropertyValue*)allocate_clear(sizeof(PropertyValue));
            value->next = property->value;
            property->value = value;
        }
    }
    return true;
}

static PyObject* build_tag_set(const Set<Tag>& tags) {
    PyObject* result = PySet_New(NULL);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create set object.");
        return NULL;
    }
    for (SetItem<Tag>* item = tags.next(NULL); item; item = tags.next(item)) {
        PyObject* value = PyTuple_New(2);
        if (!value) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create (layer, datatype) tuple.");
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(value, 0, PyLong_FromUnsignedLong(get_layer(item->value)));
        PyTuple_SET_ITEM(value, 1, PyLong_FromUnsignedLong(get_type(item->value)));
        if (PySet_Add(result, value) < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to add item to set.");
            Py_DECREF(value);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(value);
    }
    return result;
}
