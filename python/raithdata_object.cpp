static PyObject* raithdata_object_str(RaithDataObject* self) {
    RaithData& raith_data = self->raith_data;
    char buffer[GDSTK_PRINT_BUFFER_COUNT];
    snprintf(
        buffer, COUNT(buffer),
        "RaithData(base_cell_name='%s', dwelltime_selection=%" PRIu8
        ", pitch_parallel_to_path=%lg, pitch_perpendicular_to_path=%lg, pitch_scale=%lg, periods=%" PRIu32
        ", grating_type=%" PRIu32 ", dots_per_cycle=%" PRIu32 ")",
        raith_data.base_cell_name ? raith_data.base_cell_name : "", raith_data.dwelltime_selection,
        raith_data.pitch_parallel_to_path, raith_data.pitch_perpendicular_to_path,
        raith_data.pitch_scale, raith_data.periods, raith_data.grating_type, raith_data.dots_per_cycle);
    return PyUnicode_FromString(buffer);
}

static void raithdata_object_dealloc(RaithDataObject* self) {
    self->raith_data.clear();
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int raithdata_object_init(RaithDataObject* self, PyObject* args, PyObject* kwds) {
    const char* keywords[] = {"base_cell_name",
                              "dwelltime_selection",
                              "pitch_parallel_to_path",
                              "pitch_perpendicular_to_path",
                              "pitch_scale",
                              "periods",
                              "grating_type",
                              "dots_per_cycle",
                              NULL};
    char const* base_cell_name = NULL;
    unsigned int dwelltime_selection = 0;
    double pitch_parallel_to_path = 0;
    double pitch_perpendicular_to_path = 0;
    double pitch_scale = 0;
    int periods = 0;
    int grating_type = 0;
    int dots_per_cycle = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|Idddiii:RaithData", (char**)keywords,
                                     &base_cell_name, &dwelltime_selection, &pitch_parallel_to_path,
                                     &pitch_perpendicular_to_path, &pitch_scale, &periods,
                                     &grating_type, &dots_per_cycle))
        return -1;

    RaithData& raith_data = self->raith_data;
    raith_data.clear();

    raith_data.base_cell_name = copy_string(base_cell_name, NULL);
    raith_data.pitch_parallel_to_path = pitch_parallel_to_path;
    raith_data.pitch_perpendicular_to_path = pitch_perpendicular_to_path;
    raith_data.pitch_scale = pitch_scale;
    raith_data.periods = (int32_t)periods;
    raith_data.grating_type = (int32_t)grating_type;
    raith_data.dots_per_cycle = (int32_t)dots_per_cycle;
    raith_data.dwelltime_selection = (uint8_t)dwelltime_selection;
    raith_data.owner = self;
    return 0;
}

static PyObject* raithdata_object_get_dwelltime_selection(RaithDataObject* self, void*) {
    return PyLong_FromUnsignedLong(self->raith_data.dwelltime_selection);
}

// int raithdata_object_set_dwelltime_selection(RaithDataObject* self, PyObject* value, void*) {
//     if (!PyLong_Check(value)) {
//         PyErr_SetString(PyExc_TypeError,
//                         "The dwelltime_selection attribute value must be an integer.");
//         return -1;
//     }
//     self->raith_data.dwelltime_selection = (uint8_t)PyLong_AsUnsignedLong(value);
//     return 0;
// }

static PyObject* raithdata_object_get_pitch_parallel_to_path(RaithDataObject* self, void*) {
    return PyFloat_FromDouble(self->raith_data.pitch_parallel_to_path);
}

// int raithdata_object_set_pitch_parallel_to_path(RaithDataObject* self, PyObject* value, void*) {
//     double new_value = PyFloat_AsDouble(value);
//     if (PyErr_Occurred()) {
//         PyErr_SetString(PyExc_TypeError,
//                         "The pitch_parallel_to_path attribute value must be a float.");
//         return -1;
//     }
//     self->raith_data.pitch_parallel_to_path = new_value;
//     return 0;
// }

static PyObject* raithdata_object_get_pitch_perpendicular_to_path(RaithDataObject* self, void*) {
    return PyFloat_FromDouble(self->raith_data.pitch_perpendicular_to_path);
}

// int raithdata_object_set_pitch_perpendicular_to_path(RaithDataObject* self, PyObject* value,
//                                                      void*) {
//     double new_value = PyFloat_AsDouble(value);
//     if (PyErr_Occurred()) {
//         PyErr_SetString(PyExc_TypeError,
//                         "The pitch_perpendicular_to_path attribute value must be a float.");
//         return -1;
//     }
//     self->raith_data.pitch_perpendicular_to_path = new_value;
//     return 0;
// }

static PyObject* raithdata_object_get_pitch_scale(RaithDataObject* self, void*) {
    return PyFloat_FromDouble(self->raith_data.pitch_scale);
}

// int raithdata_object_set_pitch_scale(RaithDataObject* self, PyObject* value, void*) {
//     double new_value = PyFloat_AsDouble(value);
//     if (PyErr_Occurred()) {
//         PyErr_SetString(PyExc_TypeError, "The pitch_scale attribute value must be a float.");
//         return -1;
//     }
//     self->raith_data.pitch_scale = new_value;
//     return 0;
// }

static PyObject* raithdata_object_get_periods(RaithDataObject* self, void*) {
    return PyLong_FromLong(self->raith_data.periods);
}

// int raithdata_object_set_periods(RaithDataObject* self, PyObject* value, void*) {
//     if (!PyLong_Check(value)) {
//         PyErr_SetString(PyExc_TypeError, "The periods attribute value must be an integer.");
//         return -1;
//     }
//     self->raith_data.periods = (int32_t)PyLong_AsLong(value);
//     return 0;
// }

static PyObject* raithdata_object_get_grating_type(RaithDataObject* self, void*) {
    return PyLong_FromLong(self->raith_data.grating_type);
}

// int raithdata_object_set_grating_type(RaithDataObject* self, PyObject* value, void*) {
//     if (!PyLong_Check(value)) {
//         PyErr_SetString(PyExc_TypeError, "The grating_type attribute value must be an integer.");
//         return -1;
//     }
//     self->raith_data.grating_type = (int32_t)PyLong_AsLong(value);
//     return 0;
// }

static PyObject* raithdata_object_get_dots_per_cycle(RaithDataObject* self, void*) {
    return PyLong_FromLong(self->raith_data.dots_per_cycle);
}

// int raithdata_object_set_dots_per_cycle(RaithDataObject* self, PyObject* value, void*) {
//     if (!PyLong_Check(value)) {
//         PyErr_SetString(PyExc_TypeError, "The dots_per_cycle attribute value must be an
//         integer."); return -1;
//     }
//     self->raith_data.dots_per_cycle = (int32_t)PyLong_AsLong(value);
//     return 0;
// }

static PyObject* raithdata_object_get_base_cell_name(RaithDataObject* self, void*) {
    PyObject* result = self->raith_data.base_cell_name
                           ? PyUnicode_FromString(self->raith_data.base_cell_name)
                           : Py_None;
    if (!result) {
        PyErr_SetString(PyExc_TypeError, "Unable to convert value to string.");
        return NULL;
    }
    return result;
}

// int raithdata_object_set_base_cell_name(RaithDataObject* self, PyObject* arg, void*) {
//     RaithData& raith_data = self->raith_data;
//
//     if (arg == Py_None) {
//         if (raith_data.base_cell_name) free_allocation(raith_data.base_cell_name);
//         raith_data.base_cell_name = NULL;
//         return 0;
//     }
//
//     if (!PyUnicode_Check(arg)) {
//         PyErr_SetString(PyExc_TypeError, "Name must be a string.");
//         return -1;
//     }
//     Py_ssize_t len = 0;
//     const char* src = PyUnicode_AsUTF8AndSize(arg, &len);
//     if (!src) return -1;
//     if (len <= 0) {
//         PyErr_SetString(PyExc_ValueError, "Empty cell name.");
//         return -1;
//     }
//
//     if (raith_data.base_cell_name) free_allocation(raith_data.base_cell_name);
//     raith_data.base_cell_name = copy_string(src, NULL);
//     return 0;
// }

static PyGetSetDef raithdata_object_getset[] = {
    {"dwelltime_selection", (getter)raithdata_object_get_dwelltime_selection, NULL,
     raithdata_object_dwelltime_selection_doc, NULL},
    {"pitch_parallel_to_path", (getter)raithdata_object_get_pitch_parallel_to_path, NULL,
     raithdata_object_pitch_parallel_to_path_doc, NULL},
    {"pitch_perpendicular_to_path", (getter)raithdata_object_get_pitch_perpendicular_to_path, NULL,
     raithdata_object_pitch_parallel_to_path_doc, NULL},
    {"pitch_scale", (getter)raithdata_object_get_pitch_scale, NULL,
     raithdata_object_pitch_scale_doc, NULL},
    {"periods", (getter)raithdata_object_get_periods, NULL, raithdata_object_periods_doc, NULL},
    {"grating_type", (getter)raithdata_object_get_grating_type, NULL,
     raithdata_object_grating_type_doc, NULL},
    {"dots_per_cycle", (getter)raithdata_object_get_dots_per_cycle, NULL,
     raithdata_object_dots_per_cycle_doc, NULL},
    {"base_cell_name", (getter)raithdata_object_get_base_cell_name, NULL,
     raithdata_object_base_cell_name_doc, NULL},
    {NULL}};
