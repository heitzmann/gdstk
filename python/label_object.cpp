/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* label_object_str(LabelObject* self) {
    char buffer[GDSTK_PRINT_BUFFER_COUNT];
    snprintf(buffer, COUNT(buffer), "Label '%s' at layer %" PRIu32 ", texttype %" PRIu32 "",
             self->label->text, get_layer(self->label->tag), get_type(self->label->tag));
    return PyUnicode_FromString(buffer);
}

static void label_object_dealloc(LabelObject* self) {
    if (self->label) {
        self->label->clear();
        free_allocation(self->label);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int label_object_init(LabelObject* self, PyObject* args, PyObject* kwds) {
    const char* text;
    PyObject* py_origin;
    PyObject* py_anchor = NULL;
    double rotation = 0;
    double magnification = 1;
    int x_reflection = 0;
    unsigned long layer = 0;
    unsigned long texttype = 0;
    const char* keywords[] = {"text",         "origin", "anchor",   "rotation", "magnification",
                              "x_reflection", "layer",  "texttype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO|Oddpkk:Label", (char**)keywords, &text,
                                     &py_origin, &py_anchor, &rotation, &magnification,
                                     &x_reflection, &layer, &texttype))
        return -1;

    if (self->label)
        self->label->clear();
    else
        self->label = (Label*)allocate_clear(sizeof(Label));

    Label* label = self->label;
    label->tag = make_tag(layer, texttype);
    if (parse_point(py_origin, label->origin, "origin") != 0) return -1;
    if (py_anchor == NULL) {
        label->anchor = Anchor::O;
    } else {
        if (!PyUnicode_Check(py_anchor)) {
            PyErr_SetString(
                PyExc_TypeError,
                "Argument anchor must be one of 'n', 's', 'e', 'w', 'o', 'ne', 'nw', 'se', 'sw'.");
            return -1;
        }
        Py_ssize_t len = 0;
        const char* anchor = PyUnicode_AsUTF8AndSize(py_anchor, &len);
        if (len == 1) {
            switch (anchor[0]) {
                case 'o':
                    label->anchor = Anchor::O;
                    break;
                case 'n':
                    label->anchor = Anchor::N;
                    break;
                case 's':
                    label->anchor = Anchor::S;
                    break;
                case 'w':
                    label->anchor = Anchor::W;
                    break;
                case 'e':
                    label->anchor = Anchor::E;
                    break;
                default:
                    goto anchor_error;
            }
        } else if (len == 2) {
            switch (anchor[0]) {
                case 'n': {
                    switch (anchor[1]) {
                        case 'w':
                            label->anchor = Anchor::NW;
                            break;
                        case 'e':
                            label->anchor = Anchor::NE;
                            break;
                        default:
                            goto anchor_error;
                    }
                } break;
                case 's': {
                    switch (anchor[1]) {
                        case 'w':
                            label->anchor = Anchor::SW;
                            break;
                        case 'e':
                            label->anchor = Anchor::SE;
                            break;
                        default:
                            goto anchor_error;
                    }
                } break;
                default:
                    goto anchor_error;
            }
        }
    }
    label->rotation = rotation;
    label->magnification = magnification;
    label->x_reflection = x_reflection > 0;
    label->text = copy_string(text, NULL);
    label->owner = self;
    return 0;

anchor_error:
    PyErr_SetString(
        PyExc_RuntimeError,
        "Argument anchor must be one of 'n', 's', 'e', 'w', 'o', 'ne', 'nw', 'se', 'sw'.");
    return -1;
}

static PyObject* label_object_copy(LabelObject* self, PyObject*) {
    LabelObject* result = PyObject_New(LabelObject, &label_object_type);
    result = (LabelObject*)PyObject_Init((PyObject*)result, &label_object_type);
    result->label = (Label*)allocate_clear(sizeof(Label));
    result->label->copy_from(*self->label);
    result->label->owner = result;
    return (PyObject*)result;
}

static PyObject* label_object_deepcopy(LabelObject* self, PyObject* arg) {
    return label_object_copy(self, NULL);
}

static PyObject* label_object_apply_repetition(LabelObject* self, PyObject*) {
    Array<Label*> array = {};
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
    Py_ssize_t count;
    if (!PyArg_ParseTuple(args, "Hs#:set_gds_property", &attribute, &value, &count)) return NULL;
    if (count >= 0) set_gds_property(self->label->properties, attribute, value, (uint64_t)count);
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
    PyObject* result = PyUnicode_FromStringAndSize((char*)value->bytes, (Py_ssize_t)value->count);
    if (PyErr_Occurred()) {
        Py_XDECREF(result);
        PyErr_Clear();
        result = PyBytes_FromStringAndSize((char*)value->bytes, (Py_ssize_t)value->count);
    }
    return result;
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
    {"__deepcopy__", (PyCFunction)label_object_deepcopy, METH_VARARGS | METH_KEYWORDS,
     label_object_deepcopy_doc},
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
    label->text = (char*)reallocate(label->text, ++len);
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
    Py_ssize_t len = 0;
    const char* anchor = PyUnicode_AsUTF8AndSize(arg, &len);
    if (len == 1) {
        switch (anchor[0]) {
            case 'o':
                self->label->anchor = Anchor::O;
                break;
            case 'n':
                self->label->anchor = Anchor::N;
                break;
            case 's':
                self->label->anchor = Anchor::S;
                break;
            case 'w':
                self->label->anchor = Anchor::W;
                break;
            case 'e':
                self->label->anchor = Anchor::E;
                break;
            default:
                goto error_out;
        }
    } else if (len == 2) {
        switch (anchor[0]) {
            case 'n': {
                switch (anchor[1]) {
                    case 'w':
                        self->label->anchor = Anchor::NW;
                        break;
                    case 'e':
                        self->label->anchor = Anchor::NE;
                        break;
                    default:
                        goto error_out;
                }
            } break;
            case 's': {
                switch (anchor[1]) {
                    case 'w':
                        self->label->anchor = Anchor::SW;
                        break;
                    case 'e':
                        self->label->anchor = Anchor::SE;
                        break;
                    default:
                        goto error_out;
                }
            } break;
            default:
                goto error_out;
        }
    }
    return 0;

error_out:
    PyErr_SetString(PyExc_RuntimeError,
                    "Anchor must be one of 'n', 's', 'e', 'w', 'o', 'ne', 'nw', 'se', 'sw'.");
    return -1;
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
    return PyLong_FromUnsignedLongLong(get_layer(self->label->tag));
}

static int label_object_set_layer(LabelObject* self, PyObject* arg, void*) {
    set_layer(self->label->tag, (uint32_t)PyLong_AsUnsignedLongLong(arg));
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert layer to int.");
        return -1;
    }
    return 0;
}

static PyObject* label_object_get_texttype(LabelObject* self, void*) {
    return PyLong_FromUnsignedLongLong(get_type(self->label->tag));
}

static int label_object_set_texttype(LabelObject* self, PyObject* arg, void*) {
    set_type(self->label->tag, (uint32_t)PyLong_AsUnsignedLongLong(arg));
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
