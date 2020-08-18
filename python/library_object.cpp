/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* library_object_str(LibraryObject* self) {
    char buffer[256];
    snprintf(buffer, COUNT(buffer), "Library '%s' with %" PRId64 " cells and %" PRId64 " raw cells",
             self->library->name, self->library->cell_array.size,
             self->library->rawcell_array.size);
    return PyUnicode_FromString(buffer);
}

static void library_object_dealloc(LibraryObject* self) {
    Library* library = self->library;
    if (library) {
        for (int64_t i = 0; i < library->cell_array.size; i++)
            Py_DECREF(library->cell_array[i]->owner);
        for (int64_t i = 0; i < library->rawcell_array.size; i++)
            Py_DECREF(library->rawcell_array[i]->owner);
        library->clear();
        free(library);
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
        for (int64_t i = 0; i < library->cell_array.size; i++)
            Py_DECREF(library->cell_array[i]->owner);
        for (int64_t i = 0; i < library->rawcell_array.size; i++)
            Py_DECREF(library->rawcell_array[i]->owner);
        library->clear();
    } else {
        self->library = (Library*)calloc(1, sizeof(Library));
        library = self->library;
    }

    if (!name) name = (char*)default_name;
    int64_t len = strlen(name) + 1;
    library->name = (char*)malloc(sizeof(char) * len);
    memcpy(library->name, name, len);

    library->unit = unit;
    library->precision = precision;
    library->owner = self;
    return 0;
}

static PyObject* library_object_add(LibraryObject* self, PyObject* args) {
    Py_ssize_t len = PyTuple_GET_SIZE(args);
    Library* library = self->library;
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject* arg = PyTuple_GET_ITEM(args, i);
        if (CellObject_Check(arg)) {
            Py_INCREF(arg);
            library->cell_array.append(((CellObject*)arg)->cell);
        } else if (RawCellObject_Check(arg)) {
            Py_INCREF(arg);
            library->rawcell_array.append(((RawCellObject*)arg)->rawcell);
        } else {
            PyErr_SetString(PyExc_TypeError, "Arguments must be of type Cell or RawCell.");
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
    result->cell = (Cell*)calloc(1, sizeof(Cell));
    Cell* cell = result->cell;
    cell->owner = result;
    int64_t len = strlen(name) + 1;
    cell->name = (char*)malloc(sizeof(char) * len);
    memcpy(cell->name, name, len);
    self->library->cell_array.append(cell);
    Py_INCREF(result);
    return (PyObject*)result;
}

static PyObject* library_object_remove(LibraryObject* self, PyObject* args) {
    Py_ssize_t len = PyTuple_GET_SIZE(args);
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject* arg = PyTuple_GET_ITEM(args, i);
        if (CellObject_Check(arg)) {
            Cell* cell = ((CellObject*)arg)->cell;
            Array<Cell*>* array = &self->library->cell_array;
            int64_t i = array->index(cell);
            if (i >= 0) {
                array->remove(i);
                Py_DECREF((PyObject*)cell->owner);
            }
        } else if (RawCellObject_Check(arg)) {
            RawCell* rawcell = ((RawCellObject*)arg)->rawcell;
            Array<RawCell*>* array = &self->library->rawcell_array;
            int64_t i = array->index(rawcell);
            if (i >= 0) {
                array->remove(i);
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

    const int64_t i0 = top_cells.size;
    const int64_t i1 = top_rawcells.size;

    PyObject* result = PyList_New(i0 + i1);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create list.");
        top_cells.clear();
        top_rawcells.clear();
        return NULL;
    }

    Cell** cell = top_cells.items;
    for (int64_t i = 0; i < i0; i++, cell++) {
        Py_INCREF((*cell)->owner);
        PyList_SET_ITEM(result, i, (PyObject*)(*cell)->owner);
    }

    RawCell** rawcell = top_rawcells.items;
    for (int64_t i = 0; i < i1; i++, rawcell++) {
        Py_INCREF((*rawcell)->owner);
        PyList_SET_ITEM(result, i0 + i, (PyObject*)(*rawcell)->owner);
    }

    top_cells.clear();
    top_rawcells.clear();
    return result;
}

static PyObject* library_object_write_gds(LibraryObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"outfile", "max_points", NULL};
    PyObject* pybytes = NULL;
    Py_ssize_t max_points = 199;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|n:write_gds", (char**)keywords,
                                     PyUnicode_FSConverter, &pybytes, &max_points))
        return NULL;

    FILE* out = fopen(PyBytes_AS_STRING(pybytes), "wb");
    if (!out) {
        PyErr_SetString(PyExc_TypeError, "Could not open file for writing.");
        return NULL;
    }
    self->library->write_gds(out, max_points, NULL);
    fclose(out);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef library_object_methods[] = {
    {"add", (PyCFunction)library_object_add, METH_VARARGS, library_object_add_doc},
    {"remove", (PyCFunction)library_object_remove, METH_VARARGS, library_object_remove_doc},
    {"new_cell", (PyCFunction)library_object_new_cell, METH_VARARGS, library_object_new_cell_doc},
    {"top_level", (PyCFunction)library_object_top_level, METH_NOARGS, library_object_top_level_doc},
    {"write_gds", (PyCFunction)library_object_write_gds, METH_VARARGS | METH_KEYWORDS,
     library_object_write_gds_doc},
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
    if (library->name) free(library->name);
    library->name = (char*)malloc(sizeof(char) * (++len));
    memcpy(library->name, src, len);
    return 0;
}

PyObject* library_object_get_cells(LibraryObject* self, void*) {
    Array<Cell*>* cell_array = &self->library->cell_array;
    Array<RawCell*>* rawcell_array = &self->library->rawcell_array;
    const int64_t size = cell_array->size + rawcell_array->size;
    PyObject* result = PyList_New(size);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create list.");
        return NULL;
    }
    int64_t i = 0;
    for (Cell** cell = cell_array->items; i < cell_array->size; i++, cell++) {
        PyObject* cell_obj = (PyObject*)(*cell)->owner;
        Py_INCREF(cell_obj);
        PyList_SET_ITEM(result, i, cell_obj);
    }
    for (RawCell** rawcell = rawcell_array->items; i < size; i++, rawcell++) {
        PyObject* rawcell_obj = (PyObject*)(*rawcell)->owner;
        Py_INCREF(rawcell_obj);
        PyList_SET_ITEM(result, i, rawcell_obj);
    }
    return result;
}

static PyGetSetDef library_object_getset[] = {
    {"name", (getter)library_object_get_name, (setter)library_object_set_name,
     library_object_name_doc, NULL},
    {"cells", (getter)library_object_get_cells, NULL, library_object_cells_doc, NULL},
    {NULL}};
