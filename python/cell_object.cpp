/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* cell_object_str(CellObject* self) {
    char buffer[256];
    snprintf(buffer, COUNT(buffer),
             "Cell '%s' with %" PRId64 " polygons, %" PRId64 " flexpaths, %" PRId64
             " robustpaths, %" PRId64 " references, and %" PRId64 " labels",
             self->cell->name, self->cell->polygon_array.size, self->cell->flexpath_array.size,
             self->cell->robustpath_array.size, self->cell->reference_array.size,
             self->cell->label_array.size);
    return PyUnicode_FromString(buffer);
}

static void cell_object_dealloc(CellObject* self) {
    Cell* cell = self->cell;
    if (cell) {
        for (int64_t i = 0; i < cell->polygon_array.size; i++)
            Py_DECREF(cell->polygon_array[i]->owner);
        for (int64_t i = 0; i < cell->reference_array.size; i++)
            Py_DECREF(cell->reference_array[i]->owner);
        for (int64_t i = 0; i < cell->flexpath_array.size; i++)
            Py_DECREF(cell->flexpath_array[i]->owner);
        for (int64_t i = 0; i < cell->robustpath_array.size; i++)
            Py_DECREF(cell->robustpath_array[i]->owner);
        for (int64_t i = 0; i < cell->label_array.size; i++) Py_DECREF(cell->label_array[i]->owner);
        cell->clear();
        free(cell);
    }
    PyObject_Del(self);
}

static int cell_object_init(CellObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"name", NULL};
    char* name = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:Cell", (char**)keywords, &name)) return -1;
    int64_t len = strlen(name) + 1;
    Cell* cell = self->cell;
    if (cell) {
        for (int64_t i = 0; i < cell->polygon_array.size; i++)
            Py_XDECREF(cell->polygon_array[i]->owner);
        for (int64_t i = 0; i < cell->reference_array.size; i++)
            Py_XDECREF(cell->reference_array[i]->owner);
        for (int64_t i = 0; i < cell->flexpath_array.size; i++)
            Py_XDECREF(cell->flexpath_array[i]->owner);
        for (int64_t i = 0; i < cell->robustpath_array.size; i++)
            Py_XDECREF(cell->robustpath_array[i]->owner);
        for (int64_t i = 0; i < cell->label_array.size; i++)
            Py_XDECREF(cell->label_array[i]->owner);
        cell->clear();
    } else {
        self->cell = (Cell*)calloc(1, sizeof(Cell));
        cell = self->cell;
    }
    cell->name = (char*)malloc(sizeof(char) * len);
    memcpy(cell->name, name, len);
    cell->owner = self;
    return 0;
}

static PyObject* cell_object_add(CellObject* self, PyObject* args) {
    Py_ssize_t len = PyTuple_GET_SIZE(args);
    Cell* cell = self->cell;
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject* arg = PyTuple_GET_ITEM(args, i);
        Py_INCREF(arg);
        if (PolygonObject_Check(arg))
            cell->polygon_array.append(((PolygonObject*)arg)->polygon);
        else if (ReferenceObject_Check(arg))
            cell->reference_array.append(((ReferenceObject*)arg)->reference);
        else if (FlexPathObject_Check(arg))
            cell->flexpath_array.append(((FlexPathObject*)arg)->flexpath);
        else if (RobustPathObject_Check(arg))
            cell->robustpath_array.append(((RobustPathObject*)arg)->robustpath);
        else if (LabelObject_Check(arg))
            cell->label_array.append(((LabelObject*)arg)->label);
        else {
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
    Array<Polygon*> array = self->cell->get_polygons(true, -1);
    if (by_spec) {
        result = PyDict_New();
        if (!result) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create dictionary.");
            return NULL;
        }
        Polygon** poly = array.items;
        for (int64_t i = 0; i < array.size; i++, poly++) {
            PyObject* area = PyFloat_FromDouble((*poly)->area());
            if (!area) {
                PyErr_SetString(PyExc_RuntimeError, "Could not convert area to float.");
                Py_DECREF(result);
                array.clear();
                return NULL;
            }
            PyObject* key = Py_BuildValue("(hh)", (*poly)->layer, (*poly)->datatype);
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
        for (int64_t i = 0; i < array.size; i++, poly++) area += (*poly)->area();
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

static PyObject* cell_object_flatten(CellObject* self, PyObject* args) {
    Cell* cell = self->cell;

    Array<Reference*> reference_array = self->cell->flatten();
    Reference** ref = reference_array.items;
    for (int64_t i = reference_array.size - 1; i >= 0; i--, ref++) Py_XDECREF((*ref)->owner);
    reference_array.clear();

    Polygon** poly = cell->polygon_array.items;
    for (int64_t i = cell->polygon_array.size - 1; i >= 0; i--, poly++)
        if (!(*poly)->owner) {
            PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
            obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
            obj->polygon = *poly;
            obj->polygon->owner = obj;
        }

    FlexPath** flexpath = cell->flexpath_array.items;
    for (int64_t i = cell->flexpath_array.size - 1; i >= 0; i--, flexpath++)
        if (!(*flexpath)->owner) {
            FlexPathObject* obj = PyObject_New(FlexPathObject, &flexpath_object_type);
            obj = (FlexPathObject*)PyObject_Init((PyObject*)obj, &flexpath_object_type);
            obj->flexpath = *flexpath;
            obj->flexpath->owner = obj;
        }

    RobustPath** robustpath = cell->robustpath_array.items;
    for (int64_t i = cell->robustpath_array.size - 1; i >= 0; i--, robustpath++)
        if (!(*robustpath)->owner) {
            RobustPathObject* obj = PyObject_New(RobustPathObject, &robustpath_object_type);
            obj = (RobustPathObject*)PyObject_Init((PyObject*)obj, &robustpath_object_type);
            obj->robustpath = *robustpath;
            obj->robustpath->owner = obj;
        }

    Label** label = cell->label_array.items;
    for (int64_t i = cell->label_array.size - 1; i >= 0; i--, label++)
        if (!(*label)->owner) {
            LabelObject* obj = PyObject_New(LabelObject, &label_object_type);
            obj = (LabelObject*)PyObject_Init((PyObject*)obj, &label_object_type);
            obj->label = *label;
            obj->label->owner = obj;
        }

    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* cell_object_copy(CellObject* self, PyObject* args, PyObject* kwds) {
    const Vec2 origin = {0, 0};
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
    result->cell = (Cell*)malloc(sizeof(Cell));
    Cell* cell = result->cell;
    cell->owner = result;
    cell->copy_from(*self->cell, name, deep_copy > 0);

    Array<Polygon*>* polygon_array = &cell->polygon_array;
    if (deep_copy) {
        for (int64_t i = 0; i < polygon_array->size; i++) {
            PolygonObject* new_obj = PyObject_New(PolygonObject, &polygon_object_type);
            new_obj = (PolygonObject*)PyObject_Init((PyObject*)new_obj, &polygon_object_type);
            Polygon* polygon = (*polygon_array)[i];
            polygon->owner = new_obj;
            new_obj->polygon = polygon;
            if (transform)
                polygon->transform(magnification, translation, x_reflection > 0, rotation, origin);
        }
    } else {
        for (int64_t i = 0; i < polygon_array->size; i++) Py_INCREF((*polygon_array)[i]->owner);
    }

    Array<Reference*>* reference_array = &cell->reference_array;
    if (deep_copy) {
        for (int64_t i = 0; i < reference_array->size; i++) {
            ReferenceObject* new_obj = PyObject_New(ReferenceObject, &reference_object_type);
            new_obj = (ReferenceObject*)PyObject_Init((PyObject*)new_obj, &reference_object_type);
            Reference* reference = (*reference_array)[i];
            reference->owner = new_obj;
            new_obj->reference = reference;
            if (reference->type == ReferenceType::Cell)
                Py_INCREF(reference->cell->owner);
            else if (reference->type == ReferenceType::RawCell)
                Py_INCREF(reference->rawcell->owner);
            if (transform)
                reference->transform(magnification, translation, x_reflection > 0, rotation,
                                     origin);
        }
    } else {
        for (int64_t i = 0; i < reference_array->size; i++) Py_INCREF((*reference_array)[i]->owner);
    }

    Array<FlexPath*>* flexpath_array = &cell->flexpath_array;
    if (deep_copy) {
        for (int64_t i = 0; i < flexpath_array->size; i++) {
            FlexPathObject* new_obj = PyObject_New(FlexPathObject, &flexpath_object_type);
            new_obj = (FlexPathObject*)PyObject_Init((PyObject*)new_obj, &flexpath_object_type);
            FlexPath* path = (*flexpath_array)[i];
            path->owner = new_obj;
            new_obj->flexpath = path;
            if (transform)
                path->transform(magnification, translation, x_reflection > 0, rotation, origin);
        }
    } else {
        for (int64_t i = 0; i < flexpath_array->size; i++) Py_INCREF((*flexpath_array)[i]->owner);
    }

    Array<RobustPath*>* robustpath_array = &cell->robustpath_array;
    if (deep_copy) {
        for (int64_t i = 0; i < robustpath_array->size; i++) {
            RobustPathObject* new_obj = PyObject_New(RobustPathObject, &robustpath_object_type);
            new_obj = (RobustPathObject*)PyObject_Init((PyObject*)new_obj, &robustpath_object_type);
            RobustPath* path = (*robustpath_array)[i];
            path->owner = new_obj;
            new_obj->robustpath = path;
            if (transform)
                path->transform(magnification, translation, x_reflection > 0, rotation, origin);
        }
    } else {
        for (int64_t i = 0; i < robustpath_array->size; i++)
            Py_INCREF((*robustpath_array)[i]->owner);
    }

    Array<Label*>* label_array = &cell->label_array;
    if (deep_copy) {
        for (int64_t i = 0; i < label_array->size; i++) {
            LabelObject* new_obj = PyObject_New(LabelObject, &label_object_type);
            new_obj = (LabelObject*)PyObject_Init((PyObject*)new_obj, &label_object_type);
            Label* label = (*label_array)[i];
            label->owner = new_obj;
            new_obj->label = label;
            if (transform)
                label->transform(magnification, translation, x_reflection > 0, rotation, origin);
        }
    } else {
        for (int64_t i = 0; i < label_array->size; i++) Py_INCREF((*label_array)[i]->owner);
    }

    return (PyObject*)result;
}

static PyObject* cell_object_write_svg(CellObject* self, PyObject* args, PyObject* kwds) {
    double scaling = 10;
    PyObject* pybytes = NULL;
    PyObject* style_obj = NULL;
    PyObject* label_style_obj = NULL;
    PyObject* pad_obj = NULL;
    const char* background = "#2222222";
    const char* keywords[] = {"outfile",    "scaling", "style", "fontstyle",
                              "background", "pad",     NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|dOOzO:write_svg", (char**)keywords,
                                     PyUnicode_FSConverter, &pybytes, &scaling, &style_obj,
                                     &label_style_obj, &background, &pad_obj))
        return NULL;

    double pad = 5;
    bool pad_as_percentage = true;
    if (pad_obj) {
        if (PyLong_Check(pad_obj)) {
            pad_as_percentage = false;
            pad = PyLong_AsLong(pad_obj);
            if (PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError, "Unable to convert pad to long.");
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
    if (style_obj && update_style(style_obj, style, "style") < 0) return NULL;

    StyleMap label_style = {0};
    if (label_style_obj && update_style(label_style_obj, label_style, "fontstyle") < 0) {
        style.clear();
        return NULL;
    }

    FILE* out = fopen(PyBytes_AS_STRING(pybytes), "w");
    if (!out) {
        PyErr_SetString(PyExc_TypeError, "Unable to open file for writing.");
        style.clear();
        label_style.clear();
        return NULL;
    }
    Py_DECREF(pybytes);
    self->cell->write_svg(out, scaling, style, label_style, background, pad, pad_as_percentage);
    fclose(out);

    style.clear();
    label_style.clear();

    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* cell_object_remove(CellObject* self, PyObject* args) {
    Py_ssize_t len = PyTuple_GET_SIZE(args);
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject* arg = PyTuple_GET_ITEM(args, i);
        if (PolygonObject_Check(arg)) {
            Polygon* polygon = ((PolygonObject*)arg)->polygon;
            Array<Polygon*>* array = &self->cell->polygon_array;
            int64_t i = array->index(polygon);
            if (i >= 0) {
                array->remove(i);
                Py_DECREF((PyObject*)polygon->owner);
            }
        } else if (ReferenceObject_Check(arg)) {
            Reference* reference = ((ReferenceObject*)arg)->reference;
            Array<Reference*>* array = &self->cell->reference_array;
            int64_t i = array->index(reference);
            if (i >= 0) {
                array->remove(i);
                Py_DECREF((PyObject*)reference->owner);
            }
        } else if (FlexPathObject_Check(arg)) {
            FlexPath* flexpath = ((FlexPathObject*)arg)->flexpath;
            Array<FlexPath*>* array = &self->cell->flexpath_array;
            int64_t i = array->index(flexpath);
            if (i >= 0) {
                array->remove(i);
                Py_DECREF((PyObject*)flexpath->owner);
            }
        } else if (RobustPathObject_Check(arg)) {
            RobustPath* robustpath = ((RobustPathObject*)arg)->robustpath;
            Array<RobustPath*>* array = &self->cell->robustpath_array;
            int64_t i = array->index(robustpath);
            if (i >= 0) {
                array->remove(i);
                Py_DECREF((PyObject*)robustpath->owner);
            }
        } else if (LabelObject_Check(arg)) {
            Label* label = ((LabelObject*)arg)->label;
            Array<Label*>* array = &self->cell->label_array;
            int64_t i = array->index(label);
            if (i >= 0) {
                array->remove(i);
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

static PyObject* cell_object_dependencies(CellObject* self, PyObject* args) {
    int recursive;
    if (!PyArg_ParseTuple(args, "p:dependencies", &recursive)) return NULL;
    Array<Cell*> cell_array = {0};
    Array<RawCell*> rawcell_array = {0};
    self->cell->get_dependencies(recursive > 0, cell_array);
    self->cell->get_raw_dependencies(recursive > 0, rawcell_array);
    PyObject* result = PyList_New(cell_array.size + rawcell_array.size);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        cell_array.clear();
        rawcell_array.clear();
        return NULL;
    }
    Cell** cell = cell_array.items;
    for (int64_t i = 0; i < cell_array.size; i++, cell++) {
        PyObject* cell_obj = (PyObject*)(*cell)->owner;
        Py_INCREF(cell_obj);
        PyList_SET_ITEM(result, i, cell_obj);
    }
    cell_array.clear();
    RawCell** rawcell = rawcell_array.items;
    for (int64_t i = 0; i < rawcell_array.size; i++, rawcell++) {
        PyObject* rawcell_obj = (PyObject*)(*rawcell)->owner;
        Py_INCREF(rawcell_obj);
        PyList_SET_ITEM(result, i, rawcell_obj);
    }
    rawcell_array.clear();
    return result;
}

static PyMethodDef cell_object_methods[] = {
    {"add", (PyCFunction)cell_object_add, METH_VARARGS, cell_object_add_doc},
    {"area", (PyCFunction)cell_object_area, METH_VARARGS, cell_object_area_doc},
    {"bounding_box", (PyCFunction)cell_object_bounding_box, METH_NOARGS,
     cell_object_bounding_box_doc},
    {"flatten", (PyCFunction)cell_object_flatten, METH_VARARGS | METH_KEYWORDS,
     cell_object_flatten_doc},
    {"copy", (PyCFunction)cell_object_copy, METH_VARARGS | METH_KEYWORDS, cell_object_copy_doc},
    {"write_svg", (PyCFunction)cell_object_write_svg, METH_VARARGS | METH_KEYWORDS,
     cell_object_write_svg_doc},
    {"remove", (PyCFunction)cell_object_remove, METH_VARARGS, cell_object_remove_doc},
    {"dependencies", (PyCFunction)cell_object_dependencies, METH_VARARGS,
     cell_object_dependencies_doc},
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
    if (cell->name) free(cell->name);
    cell->name = (char*)malloc(sizeof(char) * (++len));
    memcpy(cell->name, src, len);
    return 0;
}

PyObject* cell_object_get_polygons(CellObject* self, void*) {
    Array<Polygon*>* array = &self->cell->polygon_array;
    PyObject* result = PyList_New(array->size);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        return NULL;
    }
    Polygon** poly = array->items;
    for (int64_t i = 0; i < array->size; i++, poly++) {
        PyObject* poly_obj = (PyObject*)(*poly)->owner;
        Py_INCREF(poly_obj);
        PyList_SET_ITEM(result, i, poly_obj);
    }
    return result;
}

PyObject* cell_object_get_references(CellObject* self, void*) {
    Array<Reference*>* array = &self->cell->reference_array;
    PyObject* result = PyList_New(array->size);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        return NULL;
    }
    Reference** ref = array->items;
    for (int64_t i = 0; i < array->size; i++, ref++) {
        PyObject* ref_obj = (PyObject*)(*ref)->owner;
        Py_INCREF(ref_obj);
        PyList_SET_ITEM(result, i, ref_obj);
    }
    return result;
}

PyObject* cell_object_get_paths(CellObject* self, void*) {
    Array<FlexPath*>* flexpath_array = &self->cell->flexpath_array;
    Array<RobustPath*>* robustpath_array = &self->cell->robustpath_array;
    int64_t fp_size = flexpath_array->size;
    int64_t rp_size = robustpath_array->size;

    PyObject* result = PyList_New(fp_size + rp_size);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        return NULL;
    }
    FlexPath** flexpath = flexpath_array->items;
    for (int64_t i = 0; i < fp_size; i++, flexpath++) {
        PyObject* flexpath_obj = (PyObject*)(*flexpath)->owner;
        Py_INCREF(flexpath_obj);
        PyList_SET_ITEM(result, i, flexpath_obj);
    }
    RobustPath** robustpath = robustpath_array->items;
    for (int64_t i = 0; i < rp_size; i++, robustpath++) {
        PyObject* robustpath_obj = (PyObject*)(*robustpath)->owner;
        Py_INCREF(robustpath_obj);
        PyList_SET_ITEM(result, fp_size + i, robustpath_obj);
    }
    return result;
}

PyObject* cell_object_get_labels(CellObject* self, void*) {
    Array<Label*>* array = &self->cell->label_array;
    PyObject* result = PyList_New(array->size);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        return NULL;
    }
    Label** label = array->items;
    for (int64_t i = 0; i < array->size; i++, label++) {
        PyObject* label_obj = (PyObject*)(*label)->owner;
        Py_INCREF(label_obj);
        PyList_SET_ITEM(result, i, label_obj);
    }
    return result;
}

static PyGetSetDef cell_object_getset[] = {
    {"name", (getter)cell_object_get_name, (setter)cell_object_set_name, cell_object_name_doc,
     NULL},
    {"polygons", (getter)cell_object_get_polygons, NULL, cell_object_polygons_doc, NULL},
    {"references", (getter)cell_object_get_references, NULL, cell_object_references_doc, NULL},
    {"paths", (getter)cell_object_get_paths, NULL, cell_object_paths_doc, NULL},
    {"labels", (getter)cell_object_get_labels, NULL, cell_object_labels_doc, NULL},
    {NULL}};
