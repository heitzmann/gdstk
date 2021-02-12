/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* library_object_str(LibraryObject* self) {
    char buffer[256];
    snprintf(buffer, COUNT(buffer), "Library '%s' with %" PRIu64 " cells and %" PRIu64 " raw cells",
             self->library->name, self->library->cell_array.count,
             self->library->rawcell_array.count);
    return PyUnicode_FromString(buffer);
}

static void library_object_dealloc(LibraryObject* self) {
    Library* library = self->library;
    if (library) {
        for (uint64_t i = 0; i < library->cell_array.count; i++)
            Py_DECREF(library->cell_array[i]->owner);
        for (uint64_t i = 0; i < library->rawcell_array.count; i++)
            Py_DECREF(library->rawcell_array[i]->owner);
        library->clear();
        free_allocation(library);
    }
    PyObject_Del(self);
}

static int library_object_init(LibraryObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"name", "unit", "precision", NULL};
    const char* default_name = "library";
    char* name = NULL;
    double unit = 1e-6;
    double precision = 1e-9;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|sdd:Library", (char**)keywords, &name, &unit,
                                     &precision))
        return -1;
    Library* library = self->library;
    if (library) {
        for (uint64_t i = 0; i < library->cell_array.count; i++)
            Py_DECREF(library->cell_array[i]->owner);
        for (uint64_t i = 0; i < library->rawcell_array.count; i++)
            Py_DECREF(library->rawcell_array[i]->owner);
        library->clear();
    } else {
        self->library = (Library*)allocate_clear(sizeof(Library));
        library = self->library;
    }

    if (!name) name = (char*)default_name;
    uint64_t len;
    library->name = copy_string(name, len);

    library->unit = unit;
    library->precision = precision;
    library->owner = self;
    return 0;
}

static PyObject* library_object_add(LibraryObject* self, PyObject* args) {
    uint64_t len = PyTuple_GET_SIZE(args);
    Library* library = self->library;
    for (uint64_t i = 0; i < len; i++) {
        PyObject* arg = PyTuple_GET_ITEM(args, i);
        Py_INCREF(arg);
        if (CellObject_Check(arg)) {
            library->cell_array.append(((CellObject*)arg)->cell);
        } else if (RawCellObject_Check(arg)) {
            library->rawcell_array.append(((RawCellObject*)arg)->rawcell);
        } else if (PyIter_Check(arg)) {
            PyObject* item = PyIter_Next(arg);
            while (item) {
                if (CellObject_Check(item)) {
                    library->cell_array.append(((CellObject*)item)->cell);
                } else if (RawCellObject_Check(item)) {
                    library->rawcell_array.append(((RawCellObject*)item)->rawcell);
                } else {
                    PyErr_SetString(PyExc_TypeError, "Arguments must be of type Cell or RawCell.");
                    Py_DECREF(item);
                    Py_DECREF(arg);
                    return NULL;
                }
                item = PyIter_Next(arg);
            }
            Py_DECREF(arg);
        } else {
            PyErr_SetString(PyExc_TypeError, "Arguments must be of type Cell or RawCell.");
            Py_DECREF(arg);
            return NULL;
        }
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* library_object_new_cell(LibraryObject* self, PyObject* args) {
    const char* name = NULL;
    if (!PyArg_ParseTuple(args, "s:new_cell", &name)) return NULL;
    CellObject* result = PyObject_New(CellObject, &cell_object_type);
    result = (CellObject*)PyObject_Init((PyObject*)result, &cell_object_type);
    result->cell = (Cell*)allocate_clear(sizeof(Cell));
    Cell* cell = result->cell;
    cell->owner = result;
    uint64_t len;
    cell->name = copy_string(name, len);
    self->library->cell_array.append(cell);
    Py_INCREF(result);
    return (PyObject*)result;
}

static PyObject* library_object_remove(LibraryObject* self, PyObject* args) {
    uint64_t len = PyTuple_GET_SIZE(args);
    for (uint64_t i = 0; i < len; i++) {
        PyObject* arg = PyTuple_GET_ITEM(args, i);
        if (CellObject_Check(arg)) {
            Cell* cell = ((CellObject*)arg)->cell;
            Array<Cell*>* array = &self->library->cell_array;
            if (array->remove_item(cell)) {
                Py_DECREF((PyObject*)cell->owner);
            }
        } else if (RawCellObject_Check(arg)) {
            RawCell* rawcell = ((RawCellObject*)arg)->rawcell;
            Array<RawCell*>* array = &self->library->rawcell_array;
            if (array->remove_item(rawcell)) {
                Py_DECREF((PyObject*)rawcell->owner);
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

static PyObject* library_object_top_level(LibraryObject* self, PyObject* args) {
    Library* library = self->library;
    Array<Cell*> top_cells = {0};
    Array<RawCell*> top_rawcells = {0};
    library->top_level(top_cells, top_rawcells);

    const uint64_t i0 = top_cells.count;
    const uint64_t i1 = top_rawcells.count;

    PyObject* result = PyList_New(i0 + i1);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create list.");
        top_cells.clear();
        top_rawcells.clear();
        return NULL;
    }

    Cell** c_item = top_cells.items;
    for (uint64_t i = 0; i < i0; i++) {
        PyObject* obj = (PyObject*)(*c_item++)->owner;
        Py_INCREF(obj);
        PyList_SET_ITEM(result, i, obj);
    }

    RawCell** r_item = top_rawcells.items;
    for (uint64_t i = 0; i < i1; i++) {
        PyObject* obj = (PyObject*)(*r_item++)->owner;
        Py_INCREF(obj);
        PyList_SET_ITEM(result, i0 + i, obj);
    }

    top_cells.clear();
    top_rawcells.clear();
    return result;
}

static PyObject* library_object_write_gds(LibraryObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"outfile", "max_points", NULL};
    PyObject* pybytes = NULL;
    uint64_t max_points = 199;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|K:write_gds", (char**)keywords,
                                     PyUnicode_FSConverter, &pybytes, &max_points))
        return NULL;

    const char* filename = PyBytes_AS_STRING(pybytes);
    self->library->write_gds(filename, max_points, NULL);
    Py_DECREF(pybytes);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* library_object_write_oas(LibraryObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {
        "outfile",          "compression_level",   "detect_rectangles", "detect_trapezoids",
        "circle_tolerance", "standard_properties", "validation",        NULL};
    PyObject* pybytes = NULL;
    uint8_t compression_level = 6;
    int detect_rectangles = 1;
    int detect_trapezoids = 1;
    double circle_tolerance = 0;
    int standard_properties = 0;
    char* validation = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|bppdpz:write_oas", (char**)keywords,
                                     PyUnicode_FSConverter, &pybytes, &compression_level,
                                     &detect_rectangles, &detect_trapezoids, &circle_tolerance,
                                     &standard_properties, &validation))
        return NULL;

    uint8_t config_flags = 0;
    if (detect_rectangles == 1) config_flags |= OASIS_CONFIG_DETECT_RECTANGLES;
    if (detect_trapezoids == 1) config_flags |= OASIS_CONFIG_DETECT_TRAPEZOIDS;
    if (standard_properties == 1) config_flags |= OASIS_CONFIG_STANDARD_PROPERTIES;
    if (validation != NULL) {
        if (strcmp(validation, "crc32") == 0) {
            config_flags |= OASIS_CONFIG_INCLUDE_CRC32;
        } else if (strcmp(validation, "checksum32") == 0) {
            config_flags |= OASIS_CONFIG_INCLUDE_CHECKSUM32;
        } else {
            PyErr_SetString(PyExc_ValueError,
                            "Argument validation must be \"crc32\", \"checksum32\", or None.");
            return NULL;
        }
    }

    const char* filename = PyBytes_AS_STRING(pybytes);
    self->library->write_oas(filename, circle_tolerance, compression_level, config_flags);
    Py_DECREF(pybytes);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* library_object_set_property(LibraryObject* self, PyObject* args) {
    if (!parse_property(self->library->properties, args)) return NULL;
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* library_object_get_property(LibraryObject* self, PyObject* args) {
    return build_property(self->library->properties, args);
}

static PyObject* library_object_delete_property(LibraryObject* self, PyObject* args) {
    char* name;
    if (!PyArg_ParseTuple(args, "s:delete_property", &name)) return NULL;
    remove_property(self->library->properties, name, false);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyMethodDef library_object_methods[] = {
    {"add", (PyCFunction)library_object_add, METH_VARARGS, library_object_add_doc},
    {"remove", (PyCFunction)library_object_remove, METH_VARARGS, library_object_remove_doc},
    {"new_cell", (PyCFunction)library_object_new_cell, METH_VARARGS, library_object_new_cell_doc},
    {"top_level", (PyCFunction)library_object_top_level, METH_NOARGS, library_object_top_level_doc},
    {"write_gds", (PyCFunction)library_object_write_gds, METH_VARARGS | METH_KEYWORDS,
     library_object_write_gds_doc},
    {"write_oas", (PyCFunction)library_object_write_oas, METH_VARARGS | METH_KEYWORDS,
     library_object_write_oas_doc},
    {"set_property", (PyCFunction)library_object_set_property, METH_VARARGS,
     object_set_property_doc},
    {"get_property", (PyCFunction)library_object_get_property, METH_VARARGS,
     object_get_property_doc},
    {"delete_property", (PyCFunction)library_object_delete_property, METH_VARARGS,
     object_delete_property_doc},
    {NULL}};

PyObject* library_object_get_name(LibraryObject* self, void*) {
    PyObject* result = PyUnicode_FromString(self->library->name);
    if (!result) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert value to string.");
        return NULL;
    }
    return result;
}

int library_object_set_name(LibraryObject* self, PyObject* arg, void*) {
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Name must be a string.");
        return -1;
    }

    Py_ssize_t len = 0;
    const char* src = PyUnicode_AsUTF8AndSize(arg, &len);
    if (!src) return -1;

    Library* library = self->library;
    if (library->name) free_allocation(library->name);
    library->name = (char*)allocate(++len);
    memcpy(library->name, src, len);
    return 0;
}

PyObject* library_object_get_cells(LibraryObject* self, void*) {
    Array<Cell*>* cell_array = &self->library->cell_array;
    Array<RawCell*>* rawcell_array = &self->library->rawcell_array;
    const uint64_t count = cell_array->count + rawcell_array->count;
    PyObject* result = PyList_New(count);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create list.");
        return NULL;
    }
    uint64_t i = 0;
    for (Cell** cell = cell_array->items; i < cell_array->count; i++, cell++) {
        PyObject* cell_obj = (PyObject*)(*cell)->owner;
        Py_INCREF(cell_obj);
        PyList_SET_ITEM(result, i, cell_obj);
    }
    for (RawCell** rawcell = rawcell_array->items; i < count; i++, rawcell++) {
        PyObject* rawcell_obj = (PyObject*)(*rawcell)->owner;
        Py_INCREF(rawcell_obj);
        PyList_SET_ITEM(result, i, rawcell_obj);
    }
    return result;
}

static PyObject* library_object_get_properties(LibraryObject* self, void*) {
    return build_properties(self->library->properties);
}

int library_object_set_properties(LibraryObject* self, PyObject* arg, void*) {
    return parse_properties(self->library->properties, arg);
}

static PyGetSetDef library_object_getset[] = {
    {"name", (getter)library_object_get_name, (setter)library_object_set_name,
     library_object_name_doc, NULL},
    {"cells", (getter)library_object_get_cells, NULL, library_object_cells_doc, NULL},
    {"properties", (getter)library_object_get_properties, (setter)library_object_set_properties,
     object_properties_doc, NULL},
    {NULL}};
