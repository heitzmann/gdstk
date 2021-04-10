/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* cell_object_str(CellObject* self) {
    char buffer[256];
    snprintf(buffer, COUNT(buffer),
             "Cell '%s' with %" PRIu64 " polygons, %" PRIu64 " flexpaths, %" PRIu64
             " robustpaths, %" PRIu64 " references, and %" PRIu64 " labels",
             self->cell->name, self->cell->polygon_array.count, self->cell->flexpath_array.count,
             self->cell->robustpath_array.count, self->cell->reference_array.count,
             self->cell->label_array.count);
    return PyUnicode_FromString(buffer);
}

static void cell_object_dealloc(CellObject* self) {
    Cell* cell = self->cell;
    if (cell) {
        for (uint64_t i = 0; i < cell->polygon_array.count; i++)
            Py_DECREF(cell->polygon_array[i]->owner);
        for (uint64_t i = 0; i < cell->reference_array.count; i++)
            Py_DECREF(cell->reference_array[i]->owner);
        for (uint64_t i = 0; i < cell->flexpath_array.count; i++)
            Py_DECREF(cell->flexpath_array[i]->owner);
        for (uint64_t i = 0; i < cell->robustpath_array.count; i++)
            Py_DECREF(cell->robustpath_array[i]->owner);
        for (uint64_t i = 0; i < cell->label_array.count; i++)
            Py_DECREF(cell->label_array[i]->owner);
        cell->clear();
        free_allocation(cell);
    }
    PyObject_Del(self);
}

static int cell_object_init(CellObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"name", NULL};
    char* name = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:Cell", (char**)keywords, &name)) return -1;
    Cell* cell = self->cell;
    if (cell) {
        for (uint64_t i = 0; i < cell->polygon_array.count; i++)
            Py_XDECREF(cell->polygon_array[i]->owner);
        for (uint64_t i = 0; i < cell->reference_array.count; i++)
            Py_XDECREF(cell->reference_array[i]->owner);
        for (uint64_t i = 0; i < cell->flexpath_array.count; i++)
            Py_XDECREF(cell->flexpath_array[i]->owner);
        for (uint64_t i = 0; i < cell->robustpath_array.count; i++)
            Py_XDECREF(cell->robustpath_array[i]->owner);
        for (uint64_t i = 0; i < cell->label_array.count; i++)
            Py_XDECREF(cell->label_array[i]->owner);
        cell->clear();
    } else {
        self->cell = (Cell*)allocate_clear(sizeof(Cell));
        cell = self->cell;
    }
    uint64_t len;
    cell->name = copy_string(name, len);
    cell->owner = self;
    return 0;
}

static PyObject* cell_object_add(CellObject* self, PyObject* args) {
    uint64_t len = PyTuple_GET_SIZE(args);
    Cell* cell = self->cell;
    for (uint64_t i = 0; i < len; i++) {
        PyObject* arg = PyTuple_GET_ITEM(args, i);
        Py_INCREF(arg);
        if (PolygonObject_Check(arg)) {
            cell->polygon_array.append(((PolygonObject*)arg)->polygon);
        } else if (ReferenceObject_Check(arg)) {
            cell->reference_array.append(((ReferenceObject*)arg)->reference);
        } else if (FlexPathObject_Check(arg)) {
            cell->flexpath_array.append(((FlexPathObject*)arg)->flexpath);
        } else if (RobustPathObject_Check(arg)) {
            cell->robustpath_array.append(((RobustPathObject*)arg)->robustpath);
        } else if (LabelObject_Check(arg)) {
            cell->label_array.append(((LabelObject*)arg)->label);
        } else if (PyIter_Check(arg)) {
            PyObject* item = PyIter_Next(arg);
            while (item) {
                if (PolygonObject_Check(item)) {
                    cell->polygon_array.append(((PolygonObject*)item)->polygon);
                } else if (ReferenceObject_Check(item)) {
                    cell->reference_array.append(((ReferenceObject*)item)->reference);
                } else if (FlexPathObject_Check(item)) {
                    cell->flexpath_array.append(((FlexPathObject*)item)->flexpath);
                } else if (RobustPathObject_Check(item)) {
                    cell->robustpath_array.append(((RobustPathObject*)item)->robustpath);
                } else if (LabelObject_Check(item)) {
                    cell->label_array.append(((LabelObject*)item)->label);
                } else {
                    PyErr_SetString(
                        PyExc_TypeError,
                        "Arguments must be Polygon, FlexPath, RobustPath, Label or Reference.");
                    Py_DECREF(item);
                    Py_DECREF(arg);
                    return NULL;
                }
                item = PyIter_Next(arg);
            }
            Py_DECREF(arg);
        } else {
            PyErr_SetString(PyExc_TypeError,
                            "Arguments must be Polygon, FlexPath, RobustPath, Label or Reference.");
            Py_DECREF(arg);
            return NULL;
        }
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* cell_object_area(CellObject* self, PyObject* args) {
    PyObject* result;
    int by_spec = 0;
    if (!PyArg_ParseTuple(args, "|p:area", &by_spec)) return NULL;
    Array<Polygon*> array = {0};
    self->cell->get_polygons(true, true, -1, array);
    if (by_spec) {
        result = PyDict_New();
        if (!result) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create dictionary.");
            return NULL;
        }
        Polygon** p_item = array.items;
        for (uint64_t i = 0; i < array.count; i++, p_item++) {
            Polygon* poly = *p_item;
            PyObject* area = PyFloat_FromDouble(poly->area());
            if (!area) {
                PyErr_SetString(PyExc_RuntimeError, "Could not convert area to float.");
                Py_DECREF(result);
                array.clear();
                return NULL;
            }
            PyObject* key = Py_BuildValue("(hh)", poly->layer, poly->datatype);
            if (!key) {
                PyErr_SetString(PyExc_RuntimeError, "Unable to build key.");
                Py_DECREF(area);
                Py_DECREF(result);
                array.clear();
                return NULL;
            }
            PyObject* current = PyDict_GetItem(result, key);
            if (current) {
                PyObject* sum = PyNumber_Add(area, current);
                if (!sum) {
                    PyErr_SetString(PyExc_RuntimeError, "Unable to perform sum.");
                    Py_DECREF(key);
                    Py_DECREF(area);
                    Py_DECREF(result);
                    array.clear();
                    return NULL;
                }
                if (PyDict_SetItem(result, key, sum) < 0) {
                    PyErr_SetString(PyExc_RuntimeError, "Unable to insert value.");
                    Py_DECREF(key);
                    Py_DECREF(area);
                    Py_DECREF(result);
                    array.clear();
                    return NULL;
                }
                Py_DECREF(sum);
            } else {
                if (PyDict_SetItem(result, key, area) < 0) {
                    PyErr_SetString(PyExc_RuntimeError, "Unable to insert value.");
                    Py_DECREF(key);
                    Py_DECREF(area);
                    Py_DECREF(result);
                    array.clear();
                    return NULL;
                }
            }
            Py_DECREF(key);
            Py_DECREF(area);
        }
    } else {
        double area = 0;
        Polygon** poly = array.items;
        for (uint64_t i = 0; i < array.count; i++, poly++) area += (*poly)->area();
        result = PyFloat_FromDouble(area);
    }
    array.clear();
    return result;
}

static PyObject* cell_object_bounding_box(CellObject* self, PyObject* args) {
    Vec2 min, max;
    self->cell->bounding_box(min, max);
    if (min.x > max.x) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return Py_BuildValue("((dd)(dd))", min.x, min.y, max.x, max.y);
}

static PyObject* cell_object_convex_hull(CellObject* self, PyObject* args) {
    Array<Vec2> points = {0};
    self->cell->convex_hull(points);
    npy_intp dims[] = {(npy_intp)points.count, 2};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_MemoryError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    memcpy(data, points.items, sizeof(double) * points.count * 2);
    points.clear();
    return (PyObject*)result;
}

static PyObject* cell_object_flatten(CellObject* self, PyObject* args, PyObject* kwds) {
    int apply_repetitions = 1;
    const char* keywords[] = {"apply_repetitions", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|p:flatten", (char**)keywords,
                                     &apply_repetitions))
        return NULL;

    Cell* cell = self->cell;

    Array<Reference*> reference_array = {0};
    cell->flatten(apply_repetitions > 0, reference_array);
    Reference** ref = reference_array.items;
    for (uint64_t i = reference_array.count; i > 0; i--, ref++) Py_XDECREF((*ref)->owner);
    reference_array.clear();

    Polygon** p_item = cell->polygon_array.items;
    for (uint64_t i = cell->polygon_array.count; i > 0; i--, p_item++) {
        Polygon* poly = *p_item;
        if (!poly->owner) {
            PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
            obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
            obj->polygon = poly;
            obj->polygon->owner = obj;
        }
    }

    FlexPath** fp_item = cell->flexpath_array.items;
    for (uint64_t i = cell->flexpath_array.count; i > 0; i--, fp_item++) {
        FlexPath* flexpath = *fp_item;
        if (!flexpath->owner) {
            FlexPathObject* obj = PyObject_New(FlexPathObject, &flexpath_object_type);
            obj = (FlexPathObject*)PyObject_Init((PyObject*)obj, &flexpath_object_type);
            obj->flexpath = flexpath;
            obj->flexpath->owner = obj;
        }
    }

    RobustPath** rp_item = cell->robustpath_array.items;
    for (uint64_t i = cell->robustpath_array.count; i > 0; i--, rp_item++) {
        RobustPath* robustpath = *rp_item;
        if (!robustpath->owner) {
            RobustPathObject* obj = PyObject_New(RobustPathObject, &robustpath_object_type);
            obj = (RobustPathObject*)PyObject_Init((PyObject*)obj, &robustpath_object_type);
            obj->robustpath = robustpath;
            obj->robustpath->owner = obj;
        }
    }

    Label** l_item = cell->label_array.items;
    for (uint64_t i = cell->label_array.count; i > 0; i--, l_item++) {
        Label* label = *l_item;
        if (!label->owner) {
            LabelObject* obj = PyObject_New(LabelObject, &label_object_type);
            obj = (LabelObject*)PyObject_Init((PyObject*)obj, &label_object_type);
            obj->label = label;
            obj->label->owner = obj;
        }
    }

    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* cell_object_copy(CellObject* self, PyObject* args, PyObject* kwds) {
    char* name = NULL;
    int deep_copy = 1;
    PyObject* py_trans = NULL;
    double rotation = 0;
    double magnification = 1;
    int x_reflection = 0;
    const char* keywords[] = {"name",         "translation", "rotation", "magnification",
                              "x_reflection", "deep_copy",   NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|Oddpp:copy", (char**)keywords, &name, &py_trans,
                                     &rotation, &magnification, &x_reflection, &deep_copy))
        return NULL;

    Vec2 translation = {0, 0};
    if (py_trans && parse_point(py_trans, translation, "translation") != 0) return NULL;

    bool transform = (translation.x != 0 || translation.y != 0 || rotation != 0 ||
                      magnification != 1 || x_reflection > 0);
    if (transform) deep_copy = 1;

    CellObject* result = PyObject_New(CellObject, &cell_object_type);
    result = (CellObject*)PyObject_Init((PyObject*)result, &cell_object_type);
    result->cell = (Cell*)allocate_clear(sizeof(Cell));
    Cell* cell = result->cell;
    cell->owner = result;
    cell->copy_from(*self->cell, name, deep_copy > 0);

    Array<Polygon*>* polygon_array = &cell->polygon_array;
    if (deep_copy) {
        for (uint64_t i = 0; i < polygon_array->count; i++) {
            PolygonObject* new_obj = PyObject_New(PolygonObject, &polygon_object_type);
            new_obj = (PolygonObject*)PyObject_Init((PyObject*)new_obj, &polygon_object_type);
            Polygon* polygon = (*polygon_array)[i];
            polygon->owner = new_obj;
            new_obj->polygon = polygon;
            if (transform) {
                polygon->transform(magnification, x_reflection > 0, rotation, translation);
                polygon->repetition.transform(magnification, x_reflection > 0, rotation);
            }
        }
    } else {
        for (uint64_t i = 0; i < polygon_array->count; i++) Py_INCREF((*polygon_array)[i]->owner);
    }

    Array<Reference*>* reference_array = &cell->reference_array;
    if (deep_copy) {
        for (uint64_t i = 0; i < reference_array->count; i++) {
            ReferenceObject* new_obj = PyObject_New(ReferenceObject, &reference_object_type);
            new_obj = (ReferenceObject*)PyObject_Init((PyObject*)new_obj, &reference_object_type);
            Reference* reference = (*reference_array)[i];
            reference->owner = new_obj;
            new_obj->reference = reference;
            if (reference->type == ReferenceType::Cell)
                Py_INCREF(reference->cell->owner);
            else if (reference->type == ReferenceType::RawCell)
                Py_INCREF(reference->rawcell->owner);
            if (transform) {
                reference->transform(magnification, x_reflection > 0, rotation, translation);
                reference->repetition.transform(magnification, x_reflection > 0, rotation);
            }
        }
    } else {
        for (uint64_t i = 0; i < reference_array->count; i++)
            Py_INCREF((*reference_array)[i]->owner);
    }

    Array<FlexPath*>* flexpath_array = &cell->flexpath_array;
    if (deep_copy) {
        for (uint64_t i = 0; i < flexpath_array->count; i++) {
            FlexPathObject* new_obj = PyObject_New(FlexPathObject, &flexpath_object_type);
            new_obj = (FlexPathObject*)PyObject_Init((PyObject*)new_obj, &flexpath_object_type);
            FlexPath* path = (*flexpath_array)[i];
            path->owner = new_obj;
            new_obj->flexpath = path;
            if (transform) {
                path->transform(magnification, x_reflection > 0, rotation, translation);
                path->repetition.transform(magnification, x_reflection > 0, rotation);
            }
        }
    } else {
        for (uint64_t i = 0; i < flexpath_array->count; i++) Py_INCREF((*flexpath_array)[i]->owner);
    }

    Array<RobustPath*>* robustpath_array = &cell->robustpath_array;
    if (deep_copy) {
        for (uint64_t i = 0; i < robustpath_array->count; i++) {
            RobustPathObject* new_obj = PyObject_New(RobustPathObject, &robustpath_object_type);
            new_obj = (RobustPathObject*)PyObject_Init((PyObject*)new_obj, &robustpath_object_type);
            RobustPath* path = (*robustpath_array)[i];
            path->owner = new_obj;
            new_obj->robustpath = path;
            if (transform) {
                path->transform(magnification, x_reflection > 0, rotation, translation);
                path->repetition.transform(magnification, x_reflection > 0, rotation);
            }
        }
    } else {
        for (uint64_t i = 0; i < robustpath_array->count; i++)
            Py_INCREF((*robustpath_array)[i]->owner);
    }

    Array<Label*>* label_array = &cell->label_array;
    if (deep_copy) {
        for (uint64_t i = 0; i < label_array->count; i++) {
            LabelObject* new_obj = PyObject_New(LabelObject, &label_object_type);
            new_obj = (LabelObject*)PyObject_Init((PyObject*)new_obj, &label_object_type);
            Label* label = (*label_array)[i];
            label->owner = new_obj;
            new_obj->label = label;
            if (transform) {
                label->transform(magnification, x_reflection > 0, rotation, translation);
                label->repetition.transform(magnification, x_reflection > 0, rotation);
            }
        }
    } else {
        for (uint64_t i = 0; i < label_array->count; i++) Py_INCREF((*label_array)[i]->owner);
    }

    return (PyObject*)result;
}

static PyObject* cell_object_write_svg(CellObject* self, PyObject* args, PyObject* kwds) {
    double scaling = 10;
    PyObject* pybytes = NULL;
    PyObject* style_obj = Py_None;
    PyObject* label_style_obj = Py_None;
    PyObject* pad_obj = NULL;
    PyObject* sort_obj = Py_None;
    const char* background = "#222222";
    const char* keywords[] = {"outfile",    "scaling", "style",         "fontstyle",
                              "background", "pad",     "sort_function", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|dOOzOO:write_svg", (char**)keywords,
                                     PyUnicode_FSConverter, &pybytes, &scaling, &style_obj,
                                     &label_style_obj, &background, &pad_obj, &sort_obj))
        return NULL;

    double pad = 5;
    bool pad_as_percentage = true;
    if (pad_obj) {
        if (PyLong_Check(pad_obj)) {
            pad_as_percentage = false;
            pad = (double)PyLong_AsLongLong(pad_obj);
            if (PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert pad to int.");
                return NULL;
            }
        } else if (PyFloat_Check(pad_obj)) {
            pad_as_percentage = false;
            pad = PyFloat_AsDouble(pad_obj);
            if (PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert pad to double.");
                return NULL;
            }
        } else if (PyUnicode_Check(pad_obj)) {
            Py_ssize_t len = 0;
            const char* src = PyUnicode_AsUTF8AndSize(pad_obj, &len);
            if (!src) {
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert pad to string.");
                return NULL;
            }
            char* end = NULL;
            pad = strtod(src, &end);
            pad_as_percentage = *end == '%';
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument pad must be a number or str.");
            return NULL;
        }
    }

    StyleMap style = {0};
    if (style_obj != Py_None && update_style(style_obj, style, "style") < 0) return NULL;

    StyleMap label_style = {0};
    if (label_style_obj != Py_None && update_style(label_style_obj, label_style, "fontstyle") < 0) {
        style.clear();
        return NULL;
    }

    const char* filename = PyBytes_AS_STRING(pybytes);

    if (sort_obj == Py_None) {
        self->cell->write_svg(filename, scaling, style, label_style, background, pad,
                              pad_as_percentage, NULL);
    } else {
        if (!PyCallable_Check(sort_obj)) {
            PyErr_SetString(PyExc_TypeError, "Argument sort_function must be callable.");
            Py_DECREF(pybytes);
            style.clear();
            label_style.clear();
            return NULL;
        }
        polygon_comparison_pyfunc = sort_obj;
        polygon_comparison_pylist = PyList_New(0);
        self->cell->write_svg(filename, scaling, style, label_style, background, pad,
                              pad_as_percentage, polygon_comparison);
        Py_DECREF(polygon_comparison_pylist);
    }

    Py_DECREF(pybytes);

    style.clear();
    label_style.clear();

    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* cell_object_remove(CellObject* self, PyObject* args) {
    uint64_t len = PyTuple_GET_SIZE(args);
    for (uint64_t i = 0; i < len; i++) {
        PyObject* arg = PyTuple_GET_ITEM(args, i);
        if (PolygonObject_Check(arg)) {
            Polygon* polygon = ((PolygonObject*)arg)->polygon;
            Array<Polygon*>* array = &self->cell->polygon_array;
            if (array->remove_item(polygon)) {
                Py_DECREF((PyObject*)polygon->owner);
            }
        } else if (ReferenceObject_Check(arg)) {
            Reference* reference = ((ReferenceObject*)arg)->reference;
            Array<Reference*>* array = &self->cell->reference_array;
            if (array->remove_item(reference)) {
                Py_DECREF((PyObject*)reference->owner);
            }
        } else if (FlexPathObject_Check(arg)) {
            FlexPath* flexpath = ((FlexPathObject*)arg)->flexpath;
            Array<FlexPath*>* array = &self->cell->flexpath_array;
            if (array->remove_item(flexpath)) {
                Py_DECREF((PyObject*)flexpath->owner);
            }
        } else if (RobustPathObject_Check(arg)) {
            RobustPath* robustpath = ((RobustPathObject*)arg)->robustpath;
            Array<RobustPath*>* array = &self->cell->robustpath_array;
            if (array->remove_item(robustpath)) {
                Py_DECREF((PyObject*)robustpath->owner);
            }
        } else if (LabelObject_Check(arg)) {
            Label* label = ((LabelObject*)arg)->label;
            Array<Label*>* array = &self->cell->label_array;
            if (array->remove_item(label)) {
                Py_DECREF((PyObject*)label->owner);
            }
        } else {
            PyErr_SetString(PyExc_TypeError,
                            "Arguments must be Polygon, FlexPath, RobustPath, Label or Reference.");
            return NULL;
        }
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static bool filter_check(int8_t operation, bool a, bool b) {
    switch (operation) {
        case 0:
            return a && b;
        case 1:
            return a || b;
        case 2:
            return a != b;
        case 3:
            return !(a && b);
        case 4:
            return !(a || b);
        case 5:
            return a == b;
        default:
            return false;
    }
}

static PyObject* cell_object_filter(CellObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_layers = NULL;
    PyObject* py_types = NULL;
    char* operation = NULL;
    int polygons = 1;
    int paths = 1;
    int labels = 1;
    const char* keywords[] = {"layers", "types", "operation", "polygons", "paths", "labels", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOs|ppp:filter", (char**)keywords, &py_layers,
                                     &py_types, &operation, &polygons, &paths, &labels))
        return NULL;

    if (PySequence_Check(py_layers) == 0 || PySequence_Check(py_types) == 0) {
        PyErr_SetString(PyExc_TypeError, "Arguments layers and types must be sequences.");
        return NULL;
    }

    Array<uint32_t> layers = {0};
    Array<uint32_t> types = {0};
    if (parse_uint_sequence(py_layers, layers, "layers") < 0 ||
        parse_uint_sequence(py_types, types, "types") < 0) {
        layers.clear();
        types.clear();
        return NULL;
    }

    int8_t op = -1;
    if (strcmp(operation, "and") == 0) {
        op = 0;
    } else if (strcmp(operation, "or") == 0) {
        op = 1;
    } else if (strcmp(operation, "xor") == 0) {
        op = 2;
    } else if (strcmp(operation, "nand") == 0) {
        op = 3;
    } else if (strcmp(operation, "nor") == 0) {
        op = 4;
    } else if (strcmp(operation, "nxor") == 0) {
        op = 5;
    } else {
        PyErr_SetString(PyExc_ValueError,
                        "Operation must be one of 'and', 'or', 'xor', 'nand', 'or', 'nxor'.");
        layers.clear();
        types.clear();
        return NULL;
    }

    Cell* cell = self->cell;

    if (polygons > 0) {
        uint64_t i = 0;
        while (i < cell->polygon_array.count) {
            Polygon* poly = cell->polygon_array[i];
            if (filter_check(op, layers.contains(poly->layer), types.contains(poly->datatype))) {
                cell->polygon_array.remove_unordered(i);
                Py_DECREF(poly->owner);
            } else {
                ++i;
            }
        }
    }

    if (paths > 0) {
        uint64_t i = 0;
        while (i < cell->flexpath_array.count) {
            FlexPath* path = cell->flexpath_array[i];
            uint64_t remove = 0;
            uint64_t j = 0;
            while (j < path->num_elements) {
                FlexPathElement* el = path->elements + j++;
                if (filter_check(op, layers.contains(el->layer), types.contains(el->datatype)))
                    remove++;
            }
            if (remove == path->num_elements) {
                cell->flexpath_array.remove_unordered(i);
                Py_DECREF(path->owner);
            } else {
                if (remove > 0) {
                    j = 0;
                    while (j < path->num_elements) {
                        FlexPathElement* el = path->elements + j;
                        if (filter_check(op, layers.contains(el->layer),
                                         types.contains(el->datatype))) {
                            el->half_width_and_offset.clear();
                            path->elements[j] = path->elements[--path->num_elements];
                        } else {
                            ++j;
                        }
                    }
                }
                ++i;
            }
        }

        i = 0;
        while (i < cell->robustpath_array.count) {
            RobustPath* path = cell->robustpath_array[i];
            uint64_t remove = 0;
            uint64_t j = 0;
            while (j < path->num_elements) {
                RobustPathElement* el = path->elements + j++;
                if (filter_check(op, layers.contains(el->layer), types.contains(el->datatype)))
                    remove++;
            }
            if (remove == path->num_elements) {
                cell->robustpath_array.remove_unordered(i);
                Py_DECREF(path->owner);
            } else {
                if (remove > 0) {
                    j = 0;
                    while (j < path->num_elements) {
                        RobustPathElement* el = path->elements + j;
                        if (filter_check(op, layers.contains(el->layer),
                                         types.contains(el->datatype))) {
                            el->width_array.clear();
                            el->offset_array.clear();
                            path->elements[j] = path->elements[--path->num_elements];
                        } else {
                            ++j;
                        }
                    }
                }
                ++i;
            }
        }
    }

    if (labels > 0) {
        uint64_t i = 0;
        while (i < cell->label_array.count) {
            Label* label = cell->label_array[i];
            if (filter_check(op, layers.contains(label->layer), types.contains(label->texttype))) {
                cell->label_array.remove_unordered(i);
                Py_DECREF(label->owner);
            } else {
                ++i;
            }
        }
    }

    layers.clear();
    types.clear();

    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* cell_object_dependencies(CellObject* self, PyObject* args) {
    int recursive;
    if (!PyArg_ParseTuple(args, "p:dependencies", &recursive)) return NULL;
    Map<Cell*> cell_map = {0};
    Map<RawCell*> rawcell_map = {0};
    self->cell->get_dependencies(recursive > 0, cell_map);
    self->cell->get_raw_dependencies(recursive > 0, rawcell_map);
    PyObject* result = PyList_New(cell_map.count + rawcell_map.count);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        cell_map.clear();
        rawcell_map.clear();
        return NULL;
    }
    uint64_t i = 0;
    for (MapItem<Cell*>* item = cell_map.next(NULL); item != NULL; item = cell_map.next(item)) {
        PyObject* cell_obj = (PyObject*)item->value->owner;
        Py_INCREF(cell_obj);
        PyList_SET_ITEM(result, i++, cell_obj);
    }
    cell_map.clear();
    for (MapItem<RawCell*>* item = rawcell_map.next(NULL); item != NULL;
         item = rawcell_map.next(item)) {
        PyObject* rawcell_obj = (PyObject*)item->value->owner;
        Py_INCREF(rawcell_obj);
        PyList_SET_ITEM(result, i++, rawcell_obj);
    }
    rawcell_map.clear();
    return result;
}

static PyObject* cell_object_set_property(CellObject* self, PyObject* args) {
    if (!parse_property(self->cell->properties, args)) return NULL;
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* cell_object_get_property(CellObject* self, PyObject* args) {
    return build_property(self->cell->properties, args);
}

static PyObject* cell_object_delete_property(CellObject* self, PyObject* args) {
    char* name;
    if (!PyArg_ParseTuple(args, "s:delete_property", &name)) return NULL;
    remove_property(self->cell->properties, name, false);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyMethodDef cell_object_methods[] = {
    {"add", (PyCFunction)cell_object_add, METH_VARARGS, cell_object_add_doc},
    {"area", (PyCFunction)cell_object_area, METH_VARARGS, cell_object_area_doc},
    {"bounding_box", (PyCFunction)cell_object_bounding_box, METH_NOARGS,
     cell_object_bounding_box_doc},
    {"convex_hull", (PyCFunction)cell_object_convex_hull, METH_NOARGS, cell_object_convex_hull_doc},
    {"flatten", (PyCFunction)cell_object_flatten, METH_VARARGS | METH_KEYWORDS,
     cell_object_flatten_doc},
    {"copy", (PyCFunction)cell_object_copy, METH_VARARGS | METH_KEYWORDS, cell_object_copy_doc},
    {"write_svg", (PyCFunction)cell_object_write_svg, METH_VARARGS | METH_KEYWORDS,
     cell_object_write_svg_doc},
    {"remove", (PyCFunction)cell_object_remove, METH_VARARGS, cell_object_remove_doc},
    {"filter", (PyCFunction)cell_object_filter, METH_VARARGS | METH_KEYWORDS,
     cell_object_filter_doc},
    {"dependencies", (PyCFunction)cell_object_dependencies, METH_VARARGS,
     cell_object_dependencies_doc},
    {"set_property", (PyCFunction)cell_object_set_property, METH_VARARGS, object_set_property_doc},
    {"get_property", (PyCFunction)cell_object_get_property, METH_VARARGS, object_get_property_doc},
    {"delete_property", (PyCFunction)cell_object_delete_property, METH_VARARGS,
     object_delete_property_doc},
    {NULL}};

PyObject* cell_object_get_name(CellObject* self, void*) {
    PyObject* result = PyUnicode_FromString(self->cell->name);
    if (!result) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert value to string.");
        return NULL;
    }
    return result;
}

int cell_object_set_name(CellObject* self, PyObject* arg, void*) {
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Name must be a string.");
        return -1;
    }

    Py_ssize_t len = 0;
    const char* src = PyUnicode_AsUTF8AndSize(arg, &len);
    if (!src) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert value to string.");
        return -1;
    }

    Cell* cell = self->cell;
    if (cell->name) free_allocation(cell->name);
    cell->name = (char*)allocate(++len);
    memcpy(cell->name, src, len);
    return 0;
}

PyObject* cell_object_get_polygons(CellObject* self, void*) {
    Array<Polygon*>* array = &self->cell->polygon_array;
    PyObject* result = PyList_New(array->count);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        return NULL;
    }
    Polygon** poly = array->items;
    for (uint64_t i = 0; i < array->count; i++) {
        PyObject* poly_obj = (PyObject*)(*poly++)->owner;
        Py_INCREF(poly_obj);
        PyList_SET_ITEM(result, i, poly_obj);
    }
    return result;
}

PyObject* cell_object_get_references(CellObject* self, void*) {
    Array<Reference*>* array = &self->cell->reference_array;
    PyObject* result = PyList_New(array->count);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        return NULL;
    }
    Reference** ref = array->items;
    for (uint64_t i = 0; i < array->count; i++) {
        PyObject* ref_obj = (PyObject*)(*ref++)->owner;
        Py_INCREF(ref_obj);
        PyList_SET_ITEM(result, i, ref_obj);
    }
    return result;
}

PyObject* cell_object_get_paths(CellObject* self, void*) {
    Array<FlexPath*>* flexpath_array = &self->cell->flexpath_array;
    Array<RobustPath*>* robustpath_array = &self->cell->robustpath_array;
    uint64_t fp_size = flexpath_array->count;
    uint64_t rp_size = robustpath_array->count;

    PyObject* result = PyList_New(fp_size + rp_size);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        return NULL;
    }
    FlexPath** flexpath = flexpath_array->items;
    for (uint64_t i = 0; i < fp_size; i++) {
        PyObject* flexpath_obj = (PyObject*)(*flexpath++)->owner;
        Py_INCREF(flexpath_obj);
        PyList_SET_ITEM(result, i, flexpath_obj);
    }
    RobustPath** robustpath = robustpath_array->items;
    for (uint64_t i = 0; i < rp_size; i++) {
        PyObject* robustpath_obj = (PyObject*)(*robustpath++)->owner;
        Py_INCREF(robustpath_obj);
        PyList_SET_ITEM(result, fp_size + i, robustpath_obj);
    }
    return result;
}

PyObject* cell_object_get_labels(CellObject* self, void*) {
    Array<Label*>* array = &self->cell->label_array;
    PyObject* result = PyList_New(array->count);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        return NULL;
    }
    Label** label = array->items;
    for (uint64_t i = 0; i < array->count; i++) {
        PyObject* label_obj = (PyObject*)(*label++)->owner;
        Py_INCREF(label_obj);
        PyList_SET_ITEM(result, i, label_obj);
    }
    return result;
}

static PyObject* cell_object_get_properties(CellObject* self, void*) {
    return build_properties(self->cell->properties);
}

int cell_object_set_properties(CellObject* self, PyObject* arg, void*) {
    return parse_properties(self->cell->properties, arg);
}

static PyGetSetDef cell_object_getset[] = {
    {"name", (getter)cell_object_get_name, (setter)cell_object_set_name, cell_object_name_doc,
     NULL},
    {"polygons", (getter)cell_object_get_polygons, NULL, cell_object_polygons_doc, NULL},
    {"references", (getter)cell_object_get_references, NULL, cell_object_references_doc, NULL},
    {"paths", (getter)cell_object_get_paths, NULL, cell_object_paths_doc, NULL},
    {"labels", (getter)cell_object_get_labels, NULL, cell_object_labels_doc, NULL},
    {"properties", (getter)cell_object_get_properties, (setter)cell_object_set_properties,
     object_properties_doc, NULL},
    {NULL}};
