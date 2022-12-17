/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* gdswriter_object_str(GdsWriterObject* self) {
    char buffer[GDSTK_PRINT_BUFFER_COUNT];
    snprintf(buffer, COUNT(buffer),
             "GdsWriter with unit %lg, precision %lg, %" PRIu64 " maximal points per polygon",
             self->gdswriter->unit, self->gdswriter->precision, self->gdswriter->max_points);
    return PyUnicode_FromString(buffer);
}

static void gdswriter_object_dealloc(GdsWriterObject* self) {
    free_allocation(self->gdswriter);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int gdswriter_object_init(GdsWriterObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"outfile",    "name",      "unit", "precision",
                              "max_points", "timestamp", NULL};
    const char* default_name = "library";
    PyObject* pybytes = NULL;
    PyObject* pytimestamp = Py_None;
    tm timestamp = {};
    uint64_t max_points = 199;
    char* name = NULL;
    double unit = 1e-6;
    double precision = 1e-9;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|sddKO:GdsWriter", (char**)keywords,
                                     PyUnicode_FSConverter, &pybytes, &name, &unit, &precision,
                                     &max_points, &pytimestamp))
        return -1;

    if (unit <= 0) {
        PyErr_SetString(PyExc_ValueError, "Unit must be positive.");
        Py_DECREF(pybytes);
        return -1;
    }

    if (precision <= 0) {
        PyErr_SetString(PyExc_ValueError, "Precision must be positive.");
        Py_DECREF(pybytes);
        return -1;
    }

    if (pytimestamp != Py_None) {
        if (!PyDateTime_Check(pytimestamp)) {
            PyErr_SetString(PyExc_TypeError, "Timestamp must be a datetime object.");
            Py_DECREF(pybytes);
            return -1;
        }
        timestamp.tm_year = PyDateTime_GET_YEAR(pytimestamp) - 1900;
        timestamp.tm_mon = PyDateTime_GET_MONTH(pytimestamp) - 1;
        timestamp.tm_mday = PyDateTime_GET_DAY(pytimestamp);
        timestamp.tm_hour = PyDateTime_DATE_GET_HOUR(pytimestamp);
        timestamp.tm_min = PyDateTime_DATE_GET_MINUTE(pytimestamp);
        timestamp.tm_sec = PyDateTime_DATE_GET_SECOND(pytimestamp);
    } else {
        get_now(timestamp);
    }

    if (!name) name = (char*)default_name;

    if (!self->gdswriter) self->gdswriter = (GdsWriter*)allocate_clear(sizeof(GdsWriter));

    *self->gdswriter = gdswriter_init(PyBytes_AS_STRING(pybytes), name, unit, precision, max_points,
                                      &timestamp, NULL);
    self->gdswriter->owner = self;
    Py_DECREF(pybytes);

    if (!self->gdswriter->out) {
        PyErr_SetString(PyExc_TypeError, "Could not open file for writing.");
        return -1;
    }
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

static PyObject* gdswriter_object_close(GdsWriterObject* self, PyObject*) {
    self->gdswriter->close();
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef gdswriter_object_methods[] = {
    {"write", (PyCFunction)gdswriter_object_write, METH_VARARGS, gdswriter_object_write_doc},
    {"close", (PyCFunction)gdswriter_object_close, METH_NOARGS, gdswriter_object_close_doc},
    {NULL}};
