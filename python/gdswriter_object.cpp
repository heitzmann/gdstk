/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* gdswriter_object_str(GdsWriterObject* self) {
    char buffer[256];
    snprintf(buffer, COUNT(buffer),
             "GdsWriter with unit %lg, precision %lg, %" PRIu64 " maximal points per polygon",
             self->gdswriter->unit, self->gdswriter->precision, self->gdswriter->max_points);
    return PyUnicode_FromString(buffer);
}

static void gdswriter_object_dealloc(GdsWriterObject* self) {
    free_allocation(self->gdswriter);
    PyObject_Del(self);
}

static int gdswriter_object_init(GdsWriterObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"outfile", "name", "unit", "precision", "max_points", NULL};
    const char* default_name = "library";
    PyObject* pybytes = NULL;
    uint64_t max_points = 199;
    char* name = NULL;
    double unit = 1e-6;
    double precision = 1e-9;
    time_t now = time(NULL);
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|sddK:GdsWriter", (char**)keywords,
                                     PyUnicode_FSConverter, &pybytes, &name, &unit, &precision,
                                     &max_points))
        return -1;
    if (!self->gdswriter) self->gdswriter = (GdsWriter*)allocate_clear(sizeof(GdsWriter));

    FILE* out = fopen(PyBytes_AS_STRING(pybytes), "wb");
    if (!out) {
        PyErr_SetString(PyExc_TypeError, "Could not open file for writing.");
        return -1;
    }
    GdsWriter* gdswriter = self->gdswriter;
    gdswriter->out = out;
    gdswriter->unit = unit;
    gdswriter->precision = precision;
    gdswriter->max_points = max_points;
    gdswriter->timestamp = *localtime(&now);
    gdswriter->owner = self;
    if (!name)
        gdswriter->write_gds(default_name);
    else
        gdswriter->write_gds(name);
    return 0;
}

static PyObject* gdswriter_object_write(GdsWriterObject* self, PyObject* args) {
    uint64_t len = PyTuple_GET_SIZE(args);
    GdsWriter* gdswriter = self->gdswriter;
    for (uint64_t i = 0; i < len; i++) {
        PyObject* arg = PyTuple_GET_ITEM(args, i);
        if (CellObject_Check(arg))
            gdswriter->write_cell(*((CellObject*)arg)->cell);
        else if (RawCellObject_Check(arg))
            gdswriter->write_rawcell(*((RawCellObject*)arg)->rawcell);
        else {
            PyErr_SetString(PyExc_TypeError, "Arguments must be Cell or RawCell.");
            return NULL;
        }
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* gdswriter_object_close(GdsWriterObject* self, PyObject* args) {
    self->gdswriter->close();
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef gdswriter_object_methods[] = {
    {"write", (PyCFunction)gdswriter_object_write, METH_VARARGS, gdswriter_object_write_doc},
    {"close", (PyCFunction)gdswriter_object_close, METH_NOARGS, gdswriter_object_close_doc},
    {NULL}};
