/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* cell_object_str(CellObject* self) {
    char buffer[GDSTK_PRINT_BUFFER_COUNT];
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
        free_allocation(cell);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
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
    cell->name = copy_string(name, &len);
    cell->owner = self;
    if (len <= 1) {
        free_allocation(cell->name);
        free_allocation(cell);
        self->cell = NULL;
        PyErr_SetString(PyExc_ValueError, "Empty cell name.");
        return -1;
    }
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
    Array<Polygon*> array = {};
    self->cell->get_polygons(true, true, -1, false, 0, array);
    if (by_spec) {
        result = PyDict_New();
        if (!result) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create dictionary.");
            return NULL;
        }
        Polygon** p_item = array.items;
        for (uint64_t k = 0; k < array.count; k++, p_item++) {
            Polygon* poly = *p_item;
            PyObject* area = PyFloat_FromDouble(poly->area());
            if (!area) {
                PyErr_SetString(PyExc_RuntimeError, "Could not convert area to float.");
                Py_DECREF(result);
                for (uint64_t i = 0; i < array.count; i++) {
                    array[i]->clear();
                    free_allocation(array[i]);
                }
                array.clear();
                return NULL;
            }
            PyObject* key = Py_BuildValue("(hh)", get_layer(poly->tag), get_type(poly->tag));
            if (!key) {
                PyErr_SetString(PyExc_RuntimeError, "Unable to build key.");
                Py_DECREF(area);
                Py_DECREF(result);
                for (uint64_t i = 0; i < array.count; i++) {
                    array[i]->clear();
                    free_allocation(array[i]);
                }
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
                    for (uint64_t i = 0; i < array.count; i++) {
                        array[i]->clear();
                        free_allocation(array[i]);
                    }
                    array.clear();
                    return NULL;
                }
                if (PyDict_SetItem(result, key, sum) < 0) {
                    PyErr_SetString(PyExc_RuntimeError, "Unable to insert value.");
                    Py_DECREF(key);
                    Py_DECREF(area);
                    Py_DECREF(result);
                    for (uint64_t i = 0; i < array.count; i++) {
                        array[i]->clear();
                        free_allocation(array[i]);
                    }
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
                    for (uint64_t i = 0; i < array.count; i++) {
                        array[i]->clear();
                        free_allocation(array[i]);
                    }
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
    for (uint64_t i = 0; i < array.count; i++) {
        array[i]->clear();
        free_allocation(array[i]);
    }
    array.clear();
    return result;
}

static PyObject* cell_object_bounding_box(CellObject* self, PyObject*) {
    Vec2 min, max;
    self->cell->bounding_box(min, max);
    if (min.x > max.x) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return Py_BuildValue("((dd)(dd))", min.x, min.y, max.x, max.y);
}

static PyObject* cell_object_convex_hull(CellObject* self, PyObject*) {
    Array<Vec2> points = {};
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

static PyObject* cell_object_get_polygons(CellObject* self, PyObject* args, PyObject* kwds) {
    int apply_repetitions = 1;
    int include_paths = 1;
    PyObject* py_depth = Py_None;
    PyObject* py_layer = Py_None;
    PyObject* py_datatype = Py_None;
    const char* keywords[] = {
        "apply_repetitions", "include_paths", "depth", "layer", "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ppOOO:get_polygons", (char**)keywords,
                                     &apply_repetitions, &include_paths, &py_depth, &py_layer,
                                     &py_datatype))
        return NULL;

    int64_t depth = -1;
    if (py_depth != Py_None) {
        depth = PyLong_AsLongLong(py_depth);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert depth to integer.");
            return NULL;
        }
    }

    if ((py_layer == Py_None) != (py_datatype == Py_None)) {
        PyErr_SetString(PyExc_ValueError,
                        "Filtering is only enabled if both layer and datatype are set.");
        return NULL;
    }

    uint32_t layer = 0;
    uint32_t datatype = 0;
    bool filter = (py_layer != Py_None) && (py_datatype != Py_None);
    if (filter) {
        layer = PyLong_AsUnsignedLong(py_layer);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert layer to unsigned integer.");
            return NULL;
        }
        datatype = PyLong_AsUnsignedLong(py_datatype);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert datatype to unsigned integer.");
            return NULL;
        }
    }

    Array<Polygon*> array = {};
    self->cell->get_polygons(apply_repetitions > 0, include_paths > 0, depth, filter,
                             make_tag(layer, datatype), array);

    PyObject* result = PyList_New(array.count);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        for (uint64_t i = 0; i < array.count; i++) {
            array[i]->clear();
            free_allocation(array[i]);
        }
        array.clear();
        return NULL;
    }

    for (uint64_t i = 0; i < array.count; i++) {
        Polygon* poly = array[i];
        PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
        obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
        obj->polygon = poly;
        poly->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }

    array.clear();
    return result;
}

static PyObject* cell_object_get_paths(CellObject* self, PyObject* args, PyObject* kwds) {
    int apply_repetitions = 1;
    PyObject* py_depth = Py_None;
    PyObject* py_layer = Py_None;
    PyObject* py_datatype = Py_None;
    const char* keywords[] = {"apply_repetitions", "depth", "layer", "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|pOOO:get_polygons", (char**)keywords,
                                     &apply_repetitions, &py_depth, &py_layer, &py_datatype))
        return NULL;

    int64_t depth = -1;
    if (py_depth != Py_None) {
        depth = PyLong_AsLongLong(py_depth);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert depth to integer.");
            return NULL;
        }
    }

    uint32_t layer = 0;
    uint32_t datatype = 0;
    bool filter = (py_layer != Py_None) && (py_datatype != Py_None);
    if (filter) {
        layer = PyLong_AsUnsignedLong(py_layer);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert layer to unsigned integer.");
            return NULL;
        }
        datatype = PyLong_AsUnsignedLong(py_datatype);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert datatype to unsigned integer.");
            return NULL;
        }
    }

    Array<FlexPath*> fp_array = {};
    self->cell->get_flexpaths(apply_repetitions > 0, depth, filter, make_tag(layer, datatype),
                              fp_array);

    Array<RobustPath*> rp_array = {};
    self->cell->get_robustpaths(apply_repetitions > 0, depth, filter, make_tag(layer, datatype),
                                rp_array);

    PyObject* result = PyList_New(fp_array.count + rp_array.count);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        for (uint64_t i = 0; i < fp_array.count; i++) {
            fp_array[i]->clear();
            free_allocation(fp_array[i]);
        }
        fp_array.clear();
        for (uint64_t i = 0; i < rp_array.count; i++) {
            rp_array[i]->clear();
            free_allocation(rp_array[i]);
        }
        rp_array.clear();
        return NULL;
    }

    for (uint64_t i = 0; i < fp_array.count; i++) {
        FlexPath* path = fp_array[i];
        FlexPathObject* obj = PyObject_New(FlexPathObject, &flexpath_object_type);
        obj = (FlexPathObject*)PyObject_Init((PyObject*)obj, &flexpath_object_type);
        obj->flexpath = path;
        path->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }
    for (uint64_t i = 0; i < rp_array.count; i++) {
        RobustPath* path = rp_array[i];
        RobustPathObject* obj = PyObject_New(RobustPathObject, &robustpath_object_type);
        obj = (RobustPathObject*)PyObject_Init((PyObject*)obj, &robustpath_object_type);
        obj->robustpath = path;
        path->owner = obj;
        PyList_SET_ITEM(result, i + fp_array.count, (PyObject*)obj);
    }

    fp_array.clear();
    rp_array.clear();
    return result;
}

static PyObject* cell_object_get_labels(CellObject* self, PyObject* args, PyObject* kwds) {
    int apply_repetitions = 1;
    PyObject* py_depth = Py_None;
    PyObject* py_layer = Py_None;
    PyObject* py_texttype = Py_None;
    const char* keywords[] = {"apply_repetitions", "depth", "layer", "texttype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|pOOO:get_polygons", (char**)keywords,
                                     &apply_repetitions, &py_depth, &py_layer, &py_texttype))
        return NULL;

    int64_t depth = -1;
    if (py_depth != Py_None) {
        depth = PyLong_AsLongLong(py_depth);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert depth to integer.");
            return NULL;
        }
    }

    uint32_t layer = 0;
    uint32_t texttype = 0;
    bool filter = (py_layer != Py_None) && (py_texttype != Py_None);
    if (filter) {
        layer = PyLong_AsUnsignedLong(py_layer);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert layer to unsigned integer.");
            return NULL;
        }
        texttype = PyLong_AsUnsignedLong(py_texttype);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert texttype to unsigned integer.");
            return NULL;
        }
    }

    Array<Label*> array = {};
    self->cell->get_labels(apply_repetitions > 0, depth, filter, make_tag(layer, texttype), array);

    PyObject* result = PyList_New(array.count);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        for (uint64_t i = 0; i < array.count; i++) {
            array[i]->clear();
            free_allocation(array[i]);
        }
        array.clear();
        return NULL;
    }

    for (uint64_t i = 0; i < array.count; i++) {
        Label* label = array[i];
        LabelObject* obj = PyObject_New(LabelObject, &label_object_type);
        obj = (LabelObject*)PyObject_Init((PyObject*)obj, &label_object_type);
        obj->label = label;
        label->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }

    array.clear();
    return result;
}

static PyObject* cell_object_flatten(CellObject* self, PyObject* args, PyObject* kwds) {
    int apply_repetitions = 1;
    const char* keywords[] = {"apply_repetitions", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|p:flatten", (char**)keywords,
                                     &apply_repetitions))
        return NULL;

    Cell* cell = self->cell;
    uint64_t last_polygon = cell->polygon_array.count;
    uint64_t last_flexpath = cell->flexpath_array.count;
    uint64_t last_robustpath = cell->robustpath_array.count;
    uint64_t last_label = cell->label_array.count;

    Array<Reference*> reference_array = {};
    cell->flatten(apply_repetitions > 0, reference_array);
    Reference** ref = reference_array.items;
    for (uint64_t i = reference_array.count; i > 0; i--, ref++) Py_XDECREF((*ref)->owner);
    reference_array.clear();

    Polygon** p_item = cell->polygon_array.items + last_polygon;
    for (uint64_t i = cell->polygon_array.count; i > last_polygon; i--, p_item++) {
        Polygon* poly = *p_item;
        if (!poly->owner) {
            PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
            obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
            obj->polygon = poly;
            obj->polygon->owner = obj;
        } else {
            // This case should never happen, we're just future-proofing
            Py_INCREF(poly->owner);
        }
    }

    FlexPath** fp_item = cell->flexpath_array.items + last_flexpath;
    for (uint64_t i = cell->flexpath_array.count; i > last_flexpath; i--, fp_item++) {
        FlexPath* flexpath = *fp_item;
        if (!flexpath->owner) {
            FlexPathObject* obj = PyObject_New(FlexPathObject, &flexpath_object_type);
            obj = (FlexPathObject*)PyObject_Init((PyObject*)obj, &flexpath_object_type);
            obj->flexpath = flexpath;
            obj->flexpath->owner = obj;
        } else {
            // This case should never happen, we're just future-proofing
            Py_INCREF(flexpath->owner);
        }
    }

    RobustPath** rp_item = cell->robustpath_array.items + last_robustpath;
    for (uint64_t i = cell->robustpath_array.count; i > last_robustpath; i--, rp_item++) {
        RobustPath* robustpath = *rp_item;
        if (!robustpath->owner) {
            RobustPathObject* obj = PyObject_New(RobustPathObject, &robustpath_object_type);
            obj = (RobustPathObject*)PyObject_Init((PyObject*)obj, &robustpath_object_type);
            obj->robustpath = robustpath;
            obj->robustpath->owner = obj;
        } else {
            // This case should never happen, we're just future-proofing
            Py_INCREF(robustpath->owner);
        }
    }

    Label** l_item = cell->label_array.items + last_label;
    for (uint64_t i = cell->label_array.count; i > last_label; i--, l_item++) {
        Label* label = *l_item;
        if (!label->owner) {
            LabelObject* obj = PyObject_New(LabelObject, &label_object_type);
            obj = (LabelObject*)PyObject_Init((PyObject*)obj, &label_object_type);
            obj->label = label;
            obj->label->owner = obj;
        } else {
            // This case should never happen, we're just future-proofing
            Py_INCREF(label->owner);
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

    if (name[0] == 0) {
        PyErr_SetString(PyExc_ValueError, "Empty cell name.");
        return NULL;
    }

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
    unsigned int precision = 6;
    PyObject* pybytes = NULL;
    PyObject* style_obj = Py_None;
    PyObject* label_style_obj = Py_None;
    PyObject* pad_obj = NULL;
    PyObject* sort_obj = Py_None;
    const char* background = "#222222";
    const char* keywords[] = {"outfile",     "scaling",       "precision",
                              "shape_style", "label_style",   "background",
                              "pad",         "sort_function", NULL};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O&|dIOOzOO:write_svg", (char**)keywords, PyUnicode_FSConverter, &pybytes,
            &scaling, &precision, &style_obj, &label_style_obj, &background, &pad_obj, &sort_obj))
        return NULL;

    double pad = 5;
    bool pad_as_percentage = true;
    if (pad_obj) {
        if (PyLong_Check(pad_obj)) {
            pad_as_percentage = false;
            pad = (double)PyLong_AsLongLong(pad_obj);
            if (PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert pad to integer.");
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

    StyleMap shape_style = {};
    if (style_obj != Py_None && update_style(style_obj, shape_style, "shape_style") < 0)
        return NULL;

    StyleMap label_style = {};
    if (label_style_obj != Py_None &&
        update_style(label_style_obj, label_style, "label_style") < 0) {
        shape_style.clear();
        return NULL;
    }

    const char* filename = PyBytes_AS_STRING(pybytes);

    ErrorCode error_code;
    if (sort_obj == Py_None) {
        error_code = self->cell->write_svg(filename, scaling, precision, &shape_style, &label_style,
                                           background, pad, pad_as_percentage, NULL);
    } else {
        if (!PyCallable_Check(sort_obj)) {
            PyErr_SetString(PyExc_TypeError, "Argument sort_function must be callable.");
            Py_DECREF(pybytes);
            shape_style.clear();
            label_style.clear();
            return NULL;
        }
        polygon_comparison_pyfunc = sort_obj;
        polygon_comparison_pylist = PyList_New(0);
        error_code = self->cell->write_svg(filename, scaling, precision, &shape_style, &label_style,
                                           background, pad, pad_as_percentage, polygon_comparison);
        Py_DECREF(polygon_comparison_pylist);
        polygon_comparison_pylist = NULL;
        polygon_comparison_pyfunc = NULL;
    }

    Py_DECREF(pybytes);

    shape_style.clear();
    label_style.clear();

    if (return_error(error_code)) return NULL;

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

static PyObject* cell_object_filter(CellObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_filter = NULL;
    int remove = 1;
    int polygons = 1;
    int paths = 1;
    int labels = 1;
    const char* keywords[] = {"spec", "remove", "polygons", "paths", "labels", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|pppp:filter", (char**)keywords, &py_filter,
                                     &remove, &polygons, &paths, &labels))
        return NULL;

    Set<Tag> tag_set = {};
    if (py_filter != Py_None) {
        if (parse_tag_sequence(py_filter, tag_set, "spec") < 0) {
            tag_set.clear();
            return NULL;
        }
    }

    Cell* cell = self->cell;

    if (polygons > 0) {
        uint64_t i = 0;
        while (i < cell->polygon_array.count) {
            Polygon* poly = cell->polygon_array[i];
            if (tag_set.has_value(poly->tag) == (remove > 0)) {
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
            uint64_t remove_count = 0;
            uint64_t j = 0;
            while (j < path->num_elements) {
                FlexPathElement* el = path->elements + j++;
                if (tag_set.has_value(el->tag) == (remove > 0)) {
                    remove_count++;
                }
            }
            if (remove_count == path->num_elements) {
                cell->flexpath_array.remove_unordered(i);
                Py_DECREF(path->owner);
            } else {
                if (remove_count > 0) {
                    j = 0;
                    while (j < path->num_elements) {
                        FlexPathElement* el = path->elements + j;
                        if (tag_set.has_value(el->tag) == (remove > 0)) {
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
            uint64_t remove_count = 0;
            uint64_t j = 0;
            while (j < path->num_elements) {
                RobustPathElement* el = path->elements + j++;
                if (tag_set.has_value(el->tag) == (remove > 0)) {
                    remove_count++;
                }
            }
            if (remove_count == path->num_elements) {
                cell->robustpath_array.remove_unordered(i);
                Py_DECREF(path->owner);
            } else {
                if (remove_count > 0) {
                    j = 0;
                    while (j < path->num_elements) {
                        RobustPathElement* el = path->elements + j;
                        if (tag_set.has_value(el->tag) == (remove > 0)) {
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
            if (tag_set.has_value(label->tag) == (remove > 0)) {
                cell->label_array.remove_unordered(i);
                Py_DECREF(label->owner);
            } else {
                ++i;
            }
        }
    }

    tag_set.clear();

    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* cell_object_remap(CellObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_map = NULL;
    const char* keywords[] = {"layer_type_map", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:remap", (char**)keywords, &py_map)) return NULL;

    if (!PyMapping_Check(py_map)) {
        PyErr_SetString(
            PyExc_TypeError,
            "Argument layer_type_map must be a mapping of (layer, type) tuples to (layer, type) tuples.");
        return NULL;
    }

    PyObject* py_items = PyMapping_Items(py_map);
    if (!py_items) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to get map items.");
        return NULL;
    }

    TagMap map = {};
    const int64_t count = PyList_Size(py_items);
    for (int64_t i = 0; i < count; i++) {
        PyObject* py_item = PyList_GET_ITEM(py_items, i);
        PyObject* py_key = PyTuple_GET_ITEM(py_item, 0);
        PyObject* py_value = PyTuple_GET_ITEM(py_item, 1);
        Tag key;
        if (!parse_tag(py_key, key)) {
            PyErr_SetString(PyExc_TypeError, "Keys must be (layer, type) tuples.");
            Py_DECREF(py_items);
            map.clear();
            return NULL;
        }
        Tag value;
        if (!parse_tag(py_value, value)) {
            PyErr_SetString(PyExc_TypeError, "Values must be (layer, type) tuples.");
            Py_DECREF(py_items);
            map.clear();
            return NULL;
        }
        map.set(key, value);
    }

    self->cell->remap_tags(map);
    map.clear();
    Py_DECREF(py_items);

    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* cell_object_dependencies(CellObject* self, PyObject* args, PyObject* kwds) {
    int recursive = 1;
    const char* keywords[] = {"recursive", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|p:dependencies", (char**)keywords, &recursive))
        return NULL;

    Map<Cell*> cell_map = {};
    Map<RawCell*> rawcell_map = {};
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
    {"get_polygons", (PyCFunction)cell_object_get_polygons, METH_VARARGS | METH_KEYWORDS,
     cell_object_get_polygons_doc},
    {"get_paths", (PyCFunction)cell_object_get_paths, METH_VARARGS | METH_KEYWORDS,
     cell_object_get_paths_doc},
    {"get_labels", (PyCFunction)cell_object_get_labels, METH_VARARGS | METH_KEYWORDS,
     cell_object_get_labels_doc},
    {"flatten", (PyCFunction)cell_object_flatten, METH_VARARGS | METH_KEYWORDS,
     cell_object_flatten_doc},
    {"copy", (PyCFunction)cell_object_copy, METH_VARARGS | METH_KEYWORDS, cell_object_copy_doc},
    {"write_svg", (PyCFunction)cell_object_write_svg, METH_VARARGS | METH_KEYWORDS,
     cell_object_write_svg_doc},
    {"remove", (PyCFunction)cell_object_remove, METH_VARARGS, cell_object_remove_doc},
    {"filter", (PyCFunction)cell_object_filter, METH_VARARGS | METH_KEYWORDS,
     cell_object_filter_doc},
    {"remap", (PyCFunction)cell_object_remap, METH_VARARGS | METH_KEYWORDS, cell_object_remap_doc},
    {"dependencies", (PyCFunction)cell_object_dependencies, METH_VARARGS | METH_KEYWORDS,
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
    if (!src) return -1;
    if (len <= 0) {
        PyErr_SetString(PyExc_ValueError, "Empty cell name.");
        return -1;
    }

    Cell* cell = self->cell;
    if (cell->name) free_allocation(cell->name);
    cell->name = (char*)allocate(++len);
    memcpy(cell->name, src, len);
    return 0;
}

PyObject* cell_object_get_polygons_attr(CellObject* self, void*) {
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

PyObject* cell_object_get_paths_attr(CellObject* self, void*) {
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

PyObject* cell_object_get_labels_attr(CellObject* self, void*) {
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
    {"polygons", (getter)cell_object_get_polygons_attr, NULL, cell_object_polygons_doc, NULL},
    {"references", (getter)cell_object_get_references, NULL, cell_object_references_doc, NULL},
    {"paths", (getter)cell_object_get_paths_attr, NULL, cell_object_paths_doc, NULL},
    {"labels", (getter)cell_object_get_labels_attr, NULL, cell_object_labels_doc, NULL},
    {"properties", (getter)cell_object_get_properties, (setter)cell_object_set_properties,
     object_properties_doc, NULL},
    {NULL}};
