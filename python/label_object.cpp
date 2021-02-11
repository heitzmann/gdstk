/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* label_object_str(LabelObject* self) {
    char buffer[256];
    snprintf(buffer, COUNT(buffer), "Label '%s' at layer %" PRIu32 ", texttype %" PRIu32 "",
             self->label->text, self->label->layer, self->label->texttype);
    return PyUnicode_FromString(buffer);
}

static void label_object_dealloc(LabelObject* self) {
    if (self->label) {
        self->label->clear();
        free_allocation(self->label);
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
    unsigned long layer = 0;
    unsigned long texttype = 0;
    uint64_t len;
    const char* keywords[] = {"text",         "origin", "anchor",   "rotation", "magnification",
                              "x_reflection", "layer",  "texttype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO|Oddpkk:Label", (char**)keywords, &s,
                                     &py_origin, &py_anchor, &rotation, &magnification,
                                     &x_reflection, &layer, &texttype))
        return -1;

    if (self->label)
        self->label->clear();
    else
        self->label = (Label*)allocate_clear(sizeof(Label));

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
    label->text = copy_string(s, len);
    label->owner = self;
    return 0;
}

static PyObject* label_object_copy(LabelObject* self, PyObject* args) {
    LabelObject* result = PyObject_New(LabelObject, &label_object_type);
    result = (LabelObject*)PyObject_Init((PyObject*)result, &label_object_type);
    result->label = (Label*)allocate_clear(sizeof(Label));
    result->label->copy_from(*self->label);
    result->label->owner = result;
    return (PyObject*)result;
}

static PyObject* label_object_apply_repetition(LabelObject* self, PyObject* args) {
    Array<Label*> array = {0};
    self->label->apply_repetition(array);
    PyObject* result = PyList_New(array.count);
    for (uint64_t i = 0; i < array.count; i++) {
        LabelObject* obj = PyObject_New(LabelObject, &label_object_type);
        obj = (LabelObject*)PyObject_Init((PyObject*)obj, &label_object_type);
        obj->label = array[i];
        array[i]->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }
    array.clear();
    return result;
}

static PyObject* label_object_set_property(LabelObject* self, PyObject* args) {
    if (!parse_property(self->label->properties, args)) return NULL;
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* label_object_get_property(LabelObject* self, PyObject* args) {
    return build_property(self->label->properties, args);
}

static PyObject* label_object_delete_property(LabelObject* self, PyObject* args) {
    char* name;
    if (!PyArg_ParseTuple(args, "s:delete_property", &name)) return NULL;
    remove_property(self->label->properties, name, false);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* label_object_set_gds_property(LabelObject* self, PyObject* args) {
    uint16_t attribute;
    char* value;
    if (!PyArg_ParseTuple(args, "Hs:set_gds_property", &attribute, &value)) return NULL;
    set_gds_property(self->label->properties, attribute, value);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* label_object_get_gds_property(LabelObject* self, PyObject* args) {
    uint16_t attribute;
    if (!PyArg_ParseTuple(args, "H:get_gds_property", &attribute)) return NULL;
    const PropertyValue* value = get_gds_property(self->label->properties, attribute);
    if (!value) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return PyUnicode_FromString((char*)value->bytes);
}

static PyObject* label_object_delete_gds_property(LabelObject* self, PyObject* args) {
    uint16_t attribute;
    if (!PyArg_ParseTuple(args, "H:delete_gds_property", &attribute)) return NULL;
    remove_gds_property(self->label->properties, attribute);
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyMethodDef label_object_methods[] = {
    {"copy", (PyCFunction)label_object_copy, METH_NOARGS, label_object_copy_doc},
    {"apply_repetition", (PyCFunction)label_object_apply_repetition, METH_NOARGS,
     label_object_apply_repetition_doc},
    {"set_property", (PyCFunction)label_object_set_property, METH_VARARGS, object_set_property_doc},
    {"get_property", (PyCFunction)label_object_get_property, METH_VARARGS, object_get_property_doc},
    {"delete_property", (PyCFunction)label_object_delete_property, METH_VARARGS,
     object_delete_property_doc},
    {"set_gds_property", (PyCFunction)label_object_set_gds_property, METH_VARARGS,
     object_set_gds_property_doc},
    {"get_gds_property", (PyCFunction)label_object_get_gds_property, METH_VARARGS,
     object_get_gds_property_doc},
    {"delete_gds_property", (PyCFunction)label_object_delete_gds_property, METH_VARARGS,
     object_delete_gds_property_doc},
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
    if (label->text) free_allocation(label->text);
    label->text = (char*)allocate(++len);
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
    return PyLong_FromUnsignedLongLong(self->label->layer);
}

static int label_object_set_layer(LabelObject* self, PyObject* arg, void*) {
    self->label->layer = (uint32_t)PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert layer to int.");
        return -1;
    }
    return 0;
}

static PyObject* label_object_get_texttype(LabelObject* self, void*) {
    return PyLong_FromUnsignedLongLong(self->label->texttype);
}

static int label_object_set_texttype(LabelObject* self, PyObject* arg, void*) {
    self->label->texttype = (uint32_t)PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert texttype to int.");
        return -1;
    }
    return 0;
}

static PyObject* label_object_get_properties(LabelObject* self, void*) {
    return build_properties(self->label->properties);
}

int label_object_set_properties(LabelObject* self, PyObject* arg, void*) {
    return parse_properties(self->label->properties, arg);
}

static PyObject* label_object_get_repetition(LabelObject* self, void*) {
    RepetitionObject* obj = PyObject_New(RepetitionObject, &repetition_object_type);
    obj = (RepetitionObject*)PyObject_Init((PyObject*)obj, &repetition_object_type);
    obj->repetition.copy_from(self->label->repetition);
    return (PyObject*)obj;
}

int label_object_set_repetition(LabelObject* self, PyObject* arg, void*) {
    if (arg == Py_None) {
        self->label->repetition.clear();
        return 0;
    } else if (!RepetitionObject_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Value must be a Repetition object.");
        return -1;
    }
    RepetitionObject* repetition_obj = (RepetitionObject*)arg;
    self->label->repetition.clear();
    self->label->repetition.copy_from(repetition_obj->repetition);
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
    {"properties", (getter)label_object_get_properties, (setter)label_object_set_properties,
     object_properties_doc, NULL},
    {"repetition", (getter)label_object_get_repetition, (setter)label_object_set_repetition,
     object_repetition_doc, NULL},
    {NULL}};
