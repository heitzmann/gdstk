/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* label_object_str(LabelObject* self) {
    char buffer[256];
    snprintf(buffer, COUNT(buffer), "Label '%s' at layer %hd, texttype %hd", self->label->text,
             self->label->layer, self->label->texttype);
    return PyUnicode_FromString(buffer);
}

static void label_object_dealloc(LabelObject* self) {
    if (self->label) {
        self->label->clear();
        free(self->label);
    }
    PyObject_Del(self);
}

static int label_object_init(LabelObject* self, PyObject* args, PyObject* kwds) {
    const char* s;
    PyObject* py_origin;
    PyObject* py_anchor = NULL;
    double rotation = 0;
    double magnification = 1;
    int x_reflection = 0;
    short layer = 0;
    short texttype = 0;
    const char* keywords[] = {"text",         "origin", "anchor",   "rotation", "magnification",
                              "x_reflection", "layer",  "texttype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO|Oddphh:Label", (char**)keywords, &s,
                                     &py_origin, &py_anchor, &rotation, &magnification,
                                     &x_reflection, &layer, &texttype))
        return -1;

    if (self->label)
        self->label->clear();
    else
        self->label = (Label*)calloc(1, sizeof(Label));

    Label* label = self->label;
    label->layer = layer;
    label->texttype = texttype;
    if (parse_point(py_origin, label->origin, "origin") != 0) return -1;
    if (py_anchor == NULL || py_anchor == Py_None) {
        label->anchor = Anchor::O;
    } else {
        if (!PyUnicode_Check(py_anchor)) {
            PyErr_SetString(
                PyExc_TypeError,
                "Argument anchor must be one of 'n', 's', 'e', 'w', 'o', 'ne', 'nw', 'se', 'sw'.");
            return -1;
        }
        if (PyUnicode_CompareWithASCIIString(py_anchor, "o") == 0)
            label->anchor = Anchor::O;
        else if (PyUnicode_CompareWithASCIIString(py_anchor, "n") == 0)
            label->anchor = Anchor::N;
        else if (PyUnicode_CompareWithASCIIString(py_anchor, "s") == 0)
            label->anchor = Anchor::S;
        else if (PyUnicode_CompareWithASCIIString(py_anchor, "w") == 0)
            label->anchor = Anchor::W;
        else if (PyUnicode_CompareWithASCIIString(py_anchor, "e") == 0)
            label->anchor = Anchor::E;
        else if (PyUnicode_CompareWithASCIIString(py_anchor, "nw") == 0)
            label->anchor = Anchor::NW;
        else if (PyUnicode_CompareWithASCIIString(py_anchor, "ne") == 0)
            label->anchor = Anchor::NE;
        else if (PyUnicode_CompareWithASCIIString(py_anchor, "sw") == 0)
            label->anchor = Anchor::SW;
        else if (PyUnicode_CompareWithASCIIString(py_anchor, "se") == 0)
            label->anchor = Anchor::SE;
        else {
            PyErr_SetString(
                PyExc_RuntimeError,
                "Argument anchor must be one of 'n', 's', 'e', 'w', 'o', 'ne', 'nw', 'se', 'sw'.");
            return -1;
        }
    }
    label->rotation = rotation;
    label->magnification = magnification;
    label->x_reflection = x_reflection > 0;
    label->text = (char*)malloc((strlen(s) + 1) * sizeof(char));
    strcpy(label->text, s);
    properties_clear(label->properties);
    label->properties = NULL;
    label->owner = self;
    return 0;
}

static PyObject* label_object_copy(LabelObject* self, PyObject* args) {
    LabelObject* result = PyObject_New(LabelObject, &label_object_type);
    result = (LabelObject*)PyObject_Init((PyObject*)result, &label_object_type);
    result->label = (Label*)calloc(1, sizeof(Label));
    result->label->copy_from(*self->label);
    result->label->owner = result;
    return (PyObject*)result;
}

static PyObject* label_object_set_property(LabelObject* self, PyObject* args) {
    int16_t attr;
    char* value;
    if (!PyArg_ParseTuple(args, "hs:set_property", &attr, &value)) return NULL;
    set_property(self->label->properties, attr, value);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* label_object_get_property(LabelObject* self, PyObject* args) {
    Property* property = self->label->properties;

    if (PyTuple_Size(args) == 0 || PyTuple_GetItem(args, 0) == Py_None) {
        PyObject* result = PyDict_New();
        for (; property; property = property->next) {
            PyObject* key = PyLong_FromLong(property->key);
            if (!key) {
                PyErr_SetString(PyExc_TypeError, "Unable to convert key to int.");
                Py_DECREF(result);
                return NULL;
            }
            PyObject* val = PyUnicode_FromString(property->value);
            if (!val) {
                PyErr_SetString(PyExc_TypeError, "Unable to convert value to string.");
                Py_DECREF(key);
                Py_DECREF(result);
                return NULL;
            }
            PyDict_SetItem(result, key, val);
            Py_DECREF(key);
            Py_DECREF(val);
        }
        return result;
    }

    int16_t attr;
    if (!PyArg_ParseTuple(args, "h:get_property", &attr)) return NULL;
    const char* value = get_property(property, attr);
    if (!value) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return PyUnicode_FromString(value);
}

static PyObject* label_object_delete_property(LabelObject* self, PyObject* args) {
    int16_t attr;
    if (!PyArg_ParseTuple(args, "h:delete_property", &attr)) return NULL;
    delete_property(self->label->properties, attr);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyMethodDef label_object_methods[] = {
    {"copy", (PyCFunction)label_object_copy, METH_NOARGS, label_object_copy_doc},
    {"set_property", (PyCFunction)label_object_set_property, METH_VARARGS,
     label_object_set_property_doc},
    {"get_property", (PyCFunction)label_object_get_property, METH_VARARGS,
     label_object_get_property_doc},
    {"delete_property", (PyCFunction)label_object_delete_property, METH_VARARGS,
     label_object_delete_property_doc},
    {NULL}};

PyObject* label_object_get_text(LabelObject* self, void*) {
    PyObject* result = PyUnicode_FromString(self->label->text);
    if (!result) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert value to string.");
        return NULL;
    }
    return result;
}

int label_object_set_text(LabelObject* self, PyObject* arg, void*) {
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Text must be a string.");
        return -1;
    }

    Py_ssize_t len = 0;
    const char* src = PyUnicode_AsUTF8AndSize(arg, &len);
    if (!src) return -1;

    Label* label = self->label;
    if (label->text) free(label->text);
    label->text = (char*)malloc(sizeof(char) * (++len));
    memcpy(label->text, src, len);
    return 0;
}

static PyObject* label_object_get_origin(LabelObject* self, void*) {
    return Py_BuildValue("(dd)", self->label->origin.x, self->label->origin.y);
}

int label_object_set_origin(LabelObject* self, PyObject* arg, void*) {
    if (parse_point(arg, self->label->origin, "origin") != 0) return -1;
    return 0;
}

static PyObject* label_object_get_anchor(LabelObject* self, void*) {
    PyObject* result = NULL;
    switch (self->label->anchor) {
        case Anchor::NW:
            result = PyUnicode_FromString("nw");
            break;
        case Anchor::N:
            result = PyUnicode_FromString("n");
            break;
        case Anchor::NE:
            result = PyUnicode_FromString("ne");
            break;
        case Anchor::W:
            result = PyUnicode_FromString("w");
            break;
        case Anchor::O:
            result = PyUnicode_FromString("o");
            break;
        case Anchor::E:
            result = PyUnicode_FromString("e");
            break;
        case Anchor::SW:
            result = PyUnicode_FromString("sw");
            break;
        case Anchor::S:
            result = PyUnicode_FromString("s");
            break;
        case Anchor::SE:
            result = PyUnicode_FromString("se");
            break;
    }
    if (!result) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert value to string.");
        return NULL;
    }
    return result;
}

int label_object_set_anchor(LabelObject* self, PyObject* arg, void*) {
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError,
                        "Anchor must be one of 'n', 's', 'e', 'w', 'o', 'ne', 'nw', 'se', 'sw'.");
        return -1;
    }
    if (PyUnicode_CompareWithASCIIString(arg, "o") == 0)
        self->label->anchor = Anchor::O;
    else if (PyUnicode_CompareWithASCIIString(arg, "n") == 0)
        self->label->anchor = Anchor::N;
    else if (PyUnicode_CompareWithASCIIString(arg, "s") == 0)
        self->label->anchor = Anchor::S;
    else if (PyUnicode_CompareWithASCIIString(arg, "w") == 0)
        self->label->anchor = Anchor::W;
    else if (PyUnicode_CompareWithASCIIString(arg, "e") == 0)
        self->label->anchor = Anchor::E;
    else if (PyUnicode_CompareWithASCIIString(arg, "nw") == 0)
        self->label->anchor = Anchor::NW;
    else if (PyUnicode_CompareWithASCIIString(arg, "ne") == 0)
        self->label->anchor = Anchor::NE;
    else if (PyUnicode_CompareWithASCIIString(arg, "sw") == 0)
        self->label->anchor = Anchor::SW;
    else if (PyUnicode_CompareWithASCIIString(arg, "se") == 0)
        self->label->anchor = Anchor::SE;
    else {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Argument anchor must be one of 'n', 's', 'e', 'w', 'o', 'ne', 'nw', 'se', 'sw'.");
        return -1;
    }
    return 0;
}

static PyObject* label_object_get_rotation(LabelObject* self, void*) {
    PyObject* result = PyFloat_FromDouble(self->label->rotation);
    if (!result) PyErr_SetString(PyExc_RuntimeError, "Unable to create float.");
    return result;
}

int label_object_set_rotation(LabelObject* self, PyObject* arg, void*) {
    self->label->rotation = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to convert value to float.");
        return -1;
    }
    return 0;
}

static PyObject* label_object_get_magnification(LabelObject* self, void*) {
    PyObject* result = PyFloat_FromDouble(self->label->magnification);
    if (!result) PyErr_SetString(PyExc_RuntimeError, "Unable to create float.");
    return result;
}

int label_object_set_magnification(LabelObject* self, PyObject* arg, void*) {
    self->label->magnification = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to convert value to float.");
        return -1;
    }
    return 0;
}

static PyObject* label_object_get_x_reflection(LabelObject* self, void*) {
    if (self->label->x_reflection) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

int label_object_set_x_reflection(LabelObject* self, PyObject* arg, void*) {
    int test = PyObject_IsTrue(arg);
    if (test < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to determine truth value.");
        return -1;
    } else
        self->label->x_reflection = test > 0;
    return 0;
}

static PyObject* label_object_get_layer(LabelObject* self, void*) {
    return PyLong_FromLong(self->label->layer);
}

static int label_object_set_layer(LabelObject* self, PyObject* arg, void*) {
    self->label->layer = PyLong_AsLong(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert layer to int.");
        return -1;
    }
    return 0;
}

static PyObject* label_object_get_texttype(LabelObject* self, void*) {
    return PyLong_FromLong(self->label->texttype);
}

static int label_object_set_texttype(LabelObject* self, PyObject* arg, void*) {
    self->label->texttype = PyLong_AsLong(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert texttype to int.");
        return -1;
    }
    return 0;
}

static PyGetSetDef label_object_getset[] = {
    {"text", (getter)label_object_get_text, (setter)label_object_set_text, label_object_text_doc,
     NULL},
    {"origin", (getter)label_object_get_origin, (setter)label_object_set_origin,
     label_object_origin_doc, NULL},
    {"anchor", (getter)label_object_get_anchor, (setter)label_object_set_anchor,
     label_object_anchor_doc, NULL},
    {"rotation", (getter)label_object_get_rotation, (setter)label_object_set_rotation,
     label_object_rotation_doc},
    {"magnification", (getter)label_object_get_magnification,
     (setter)label_object_set_magnification, label_object_magnification_doc},
    {"x_reflection", (getter)label_object_get_x_reflection, (setter)label_object_set_x_reflection,
     label_object_x_reflection_doc},
    {"layer", (getter)label_object_get_layer, (setter)label_object_set_layer,
     label_object_layer_doc, NULL},
    {"texttype", (getter)label_object_get_texttype, (setter)label_object_set_texttype,
     label_object_texttype_doc, NULL},
    {NULL}};
