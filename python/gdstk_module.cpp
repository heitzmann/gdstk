/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#ifdef NDEBUG
#define DEBUG_PYTHON(o) ((void)0)
#else
#define DEBUG_PYTHON(o)                            \
    do {                                           \
        PyObject_Print((PyObject*)(o), stderr, 0); \
        putc('\n', stderr);                        \
        fflush(stderr);                            \
    } while (false)
#endif

#define CellObject_Check(o) PyObject_TypeCheck((o), &cell_object_type)
#define RaithDataObject_Check(o) PyObject_TypeCheck((o), &raithdata_object_type)
#define FlexPathObject_Check(o) PyObject_TypeCheck((o), &flexpath_object_type)
#define LabelObject_Check(o) PyObject_TypeCheck((o), &label_object_type)
#define LibraryObject_Check(o) PyObject_TypeCheck((o), &library_object_type)
#define GdsWriterObject_Check(o) PyObject_TypeCheck((o), &gdswriter_object_type)
#define PolygonObject_Check(o) PyObject_TypeCheck((o), &polygon_object_type)
#define RawCellObject_Check(o) PyObject_TypeCheck((o), &rawcell_object_type)
#define ReferenceObject_Check(o) PyObject_TypeCheck((o), &reference_object_type)
#define RepetitionObject_Check(o) PyObject_TypeCheck((o), &repetition_object_type)
#define RobustPathObject_Check(o) PyObject_TypeCheck((o), &robustpath_object_type)

#include <Python.h>
#include <datetime.h>
#include <inttypes.h>
#include <numpy/arrayobject.h>
#include <structmember.h>

#include <gdstk/gdstk.hpp>

#include "docstrings.cpp"

using namespace gdstk;

static int return_error(ErrorCode error_code) {
    switch (error_code) {
        case ErrorCode::NoError:
            return 0;
        // Warnings
        case ErrorCode::EmptyPath:
            if (PyErr_WarnEx(PyExc_RuntimeWarning, "Empty path.", 1) != 0) return -1;
            return 0;
        case ErrorCode::MissingReference:
            if (PyErr_WarnEx(PyExc_RuntimeWarning, "Missing reference.", 1) != 0) return -1;
            return 0;
        case ErrorCode::InvalidRepetition:
            if (PyErr_WarnEx(PyExc_RuntimeWarning, "Invalid repetition.", 1) != 0) return -1;
            return 0;
        case ErrorCode::BooleanError:
            if (PyErr_WarnEx(PyExc_RuntimeWarning, "Error in boolean operation.", 1) != 0)
                return -1;
            return 0;
        case ErrorCode::IntersectionNotFound:
            if (PyErr_WarnEx(PyExc_RuntimeWarning, "Intersection not found in path construction.",
                             1) != 0)
                return -1;
            return 0;
        case ErrorCode::UnofficialSpecification:
            if (PyErr_WarnEx(PyExc_RuntimeWarning,
                             "Saved file uses unofficially supported extensions.", 1) != 0)
                return -1;
            return 0;
        case ErrorCode::Overflow:
            if (PyErr_WarnEx(PyExc_RuntimeWarning, "Overflow detected.", 1) != 0) return -1;
            return 0;
        case ErrorCode::UnsupportedRecord:
            if (PyErr_WarnEx(PyExc_RuntimeWarning, "Unsupported record in file.", 1) != 0)
                return -1;
            return 0;
        // Errors
        case ErrorCode::InputFileError:
            PyErr_SetString(PyExc_OSError, "Error reading input file.");
            return -1;
        case ErrorCode::InputFileOpenError:
            PyErr_SetString(PyExc_OSError, "Error opening input file.");
            return -1;
        case ErrorCode::OutputFileOpenError:
            PyErr_SetString(PyExc_OSError, "Error opening output file.");
            return -1;
        case ErrorCode::FileError:
            PyErr_SetString(PyExc_OSError, "Error handling file.");
            return -1;
        case ErrorCode::InvalidFile:
            PyErr_SetString(PyExc_RuntimeError, "Invalid or corrupted file.");
            return -1;
        case ErrorCode::InsufficientMemory:
            PyErr_SetString(PyExc_MemoryError, "Insufficient memory.");
            return -1;
        case ErrorCode::ChecksumError:
            PyErr_SetString(PyExc_RuntimeError, "Checksum error.");
            return -1;
        case ErrorCode::ZlibError:
            PyErr_SetString(PyExc_RuntimeError, "Error in zlib library.");
            return -1;
    }
    return 0;
};

struct CurveObject {
    PyObject_HEAD;
    Curve* curve;
};

struct PolygonObject {
    PyObject_HEAD;
    Polygon* polygon;
};

struct ReferenceObject {
    PyObject_HEAD;
    Reference* reference;
};

struct FlexPathObject {
    PyObject_HEAD;
    FlexPath* flexpath;
};

struct RobustPathObject {
    PyObject_HEAD;
    RobustPath* robustpath;
};

struct LabelObject {
    PyObject_HEAD;
    Label* label;
};

struct CellObject {
    PyObject_HEAD;
    Cell* cell;
};

struct RawCellObject {
    PyObject_HEAD;
    RawCell* rawcell;
};

struct LibraryObject {
    PyObject_HEAD;
    Library* library;
};

struct GdsWriterObject {
    PyObject_HEAD;
    GdsWriter* gdswriter;
};

struct RepetitionObject {
    PyObject_HEAD;
    Repetition repetition;
};

struct RaithDataObject {
    PyObject_HEAD;
    RaithData raith_data;
};

static PyTypeObject curve_object_type = {PyVarObject_HEAD_INIT(NULL, 0) "gdstk.Curve",
                                         sizeof(CurveObject),
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                         curve_object_type_doc,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         PyType_GenericNew,
                                         0,
                                         0};

static PyTypeObject polygon_object_type = {PyVarObject_HEAD_INIT(NULL, 0) "gdstk.Polygon",
                                           sizeof(PolygonObject),
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                           polygon_object_type_doc,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           PyType_GenericNew,
                                           0,
                                           0};

static PyTypeObject reference_object_type = {PyVarObject_HEAD_INIT(NULL, 0) "gdstk.Reference",
                                             sizeof(ReferenceObject),
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                             reference_object_type_doc,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             PyType_GenericNew,
                                             0,
                                             0};

static PyTypeObject flexpath_object_type = {PyVarObject_HEAD_INIT(NULL, 0) "gdstk.FlexPath",
                                            sizeof(FlexPathObject),
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                            flexpath_object_type_doc,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            PyType_GenericNew,
                                            0,
                                            0};

static PyTypeObject robustpath_object_type = {PyVarObject_HEAD_INIT(NULL, 0) "gdstk.RobustPath",
                                              sizeof(RobustPathObject),
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                              robustpath_object_type_doc,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              PyType_GenericNew,
                                              0,
                                              0};

static PyTypeObject label_object_type = {PyVarObject_HEAD_INIT(NULL, 0) "gdstk.Label",
                                         sizeof(LabelObject),
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                         label_object_type_doc,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         PyType_GenericNew,
                                         0,
                                         0};

static PyTypeObject repetition_object_type = {PyVarObject_HEAD_INIT(NULL, 0) "gdstk.Repetition",
                                              sizeof(RepetitionObject),
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                              repetition_object_type_doc,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              PyType_GenericNew,
                                              0,
                                              0};

static PyTypeObject raithdata_object_type = {PyVarObject_HEAD_INIT(NULL, 0) "gdstk.RaithData",
                                             sizeof(RaithDataObject),
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                             raithdata_object_type_doc,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             PyType_GenericNew,
                                             0,
                                             0};

static PyTypeObject cell_object_type = {PyVarObject_HEAD_INIT(NULL, 0) "gdstk.Cell",
                                        sizeof(CellObject),
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                        cell_object_type_doc,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        PyType_GenericNew,
                                        0,
                                        0};

static PyTypeObject rawcell_object_type = {PyVarObject_HEAD_INIT(NULL, 0) "gdstk.RawCell",
                                           sizeof(RawCellObject),
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                           rawcell_object_type_doc,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           PyType_GenericNew,
                                           0,
                                           0};

static PyTypeObject library_object_type = {PyVarObject_HEAD_INIT(NULL, 0) "gdstk.Library",
                                           sizeof(LibraryObject),
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                           library_object_type_doc,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           PyType_GenericNew,
                                           0,
                                           0};

static PyTypeObject gdswriter_object_type = {PyVarObject_HEAD_INIT(NULL, 0) "gdstk.GdsWriter",
                                             sizeof(GdsWriterObject),
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                             gdswriter_object_type_doc,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             PyType_GenericNew,
                                             0,
                                             0};

#include "parsing.cpp"

// These two globals are required because we don't want to pollute the C++ API
// and add unnecessary steps before passing a comparator function to sort in
// cell.to_svg.
PyObject* polygon_comparison_pyfunc;
// This list will be populated with PolygonObjects that need to be created for
// sorting (because the original polygons might not have a corresponding python
// owner).  It should be initialized to an empty list before the sort call and
// cleared afterwards.
PyObject* polygon_comparison_pylist;
bool polygon_comparison(Polygon* const& p1, Polygon* const& p2) {
    PyObject* p1_obj;
    PyObject* p2_obj;

    if (p1->owner == NULL) {
        p1_obj = (PyObject*)PyObject_New(PolygonObject, &polygon_object_type);
        p1_obj = PyObject_Init(p1_obj, &polygon_object_type);
        ((PolygonObject*)p1_obj)->polygon = p1;
        p1->owner = p1_obj;
        PyList_Append(polygon_comparison_pylist, p1_obj);
    } else {
        p1_obj = (PyObject*)p1->owner;
        Py_INCREF(p1_obj);
    }

    if (p2->owner == NULL) {
        p2_obj = (PyObject*)PyObject_New(PolygonObject, &polygon_object_type);
        p2_obj = PyObject_Init(p2_obj, &polygon_object_type);
        ((PolygonObject*)p2_obj)->polygon = p2;
        p2->owner = p2_obj;
        PyList_Append(polygon_comparison_pylist, p2_obj);
    } else {
        p2_obj = (PyObject*)p2->owner;
        Py_INCREF(p2_obj);
    }

    PyObject* args = PyTuple_New(2);
    PyTuple_SET_ITEM(args, 0, p1_obj);
    PyTuple_SET_ITEM(args, 1, p2_obj);
    PyObject* py_result = PyObject_CallObject(polygon_comparison_pyfunc, args);
    Py_DECREF(args);

    bool result = PyObject_IsTrue(py_result) > 0;

    Py_XDECREF(py_result);
    return result;
}

double eval_parametric_double(double u, PyObject* function) {
    double result = 0;
    PyObject* py_u = PyFloat_FromDouble(u);
    if (!py_u) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Unable to create float for parametric function evaluation.");
        return result;
    }
    PyObject* args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, py_u);
    PyObject* py_result = PyObject_CallObject(function, args);
    Py_DECREF(args);
    result = PyFloat_AsDouble(py_result);
    if (PyErr_Occurred())
        PyErr_Format(PyExc_RuntimeError, "Unable to convert parametric result (%S) to double.",
                     py_result);
    Py_XDECREF(py_result);
    return result;
}

Vec2 eval_parametric_vec2(double u, PyObject* function) {
    Vec2 result = {0, 0};
    PyObject* py_u = PyFloat_FromDouble(u);
    if (!py_u) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Unable to create float for parametric function evaluation.");
        return result;
    }
    PyObject* args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, py_u);
    PyObject* py_result = PyObject_CallObject(function, args);
    Py_DECREF(args);
    if (parse_point(py_result, result, "") < 0)
        PyErr_Format(PyExc_RuntimeError,
                     "Unable to convert parametric result (%S) to coordinate pair.", py_result);
    Py_XDECREF(py_result);
    return result;
}

Array<Vec2> custom_end_function(const Vec2 first_point, const Vec2 first_direction,
                                const Vec2 second_point, const Vec2 second_direction,
                                PyObject* function) {
    Array<Vec2> array = {};
    PyObject* result = PyObject_CallFunction(
        function, "(dd)(dd)(dd)(dd)", first_point.x, first_point.y, first_direction.x,
        first_direction.y, second_point.x, second_point.y, second_direction.x, second_direction.y);
    if (result != NULL) {
        if (parse_point_sequence(result, array, "") < 0) {
            PyErr_Format(PyExc_RuntimeError, "Unable to parse return value (%S) from end function.",
                         result);
        }
        Py_DECREF(result);
    }
    return array;
}

static Array<Vec2> custom_join_function(const Vec2 first_point, const Vec2 first_direction,
                                        const Vec2 second_point, const Vec2 second_direction,
                                        const Vec2 center, double width, PyObject* function) {
    Array<Vec2> array = {};
    PyObject* result =
        PyObject_CallFunction(function, "(dd)(dd)(dd)(dd)(dd)d", first_point.x, first_point.y,
                              first_direction.x, first_direction.y, second_point.x, second_point.y,
                              second_direction.x, second_direction.y, center.x, center.y, width);
    if (result != NULL) {
        if (parse_point_sequence(result, array, "") < 0) {
            PyErr_Format(PyExc_RuntimeError,
                         "Unable to parse return value (%S) from join function.", result);
        }
        Py_DECREF(result);
    }
    return array;
}

static Array<Vec2> custom_bend_function(double radius, double initial_angle, double final_angle,
                                        const Vec2 center, PyObject* function) {
    Array<Vec2> array = {};
    PyObject* result = PyObject_CallFunction(function, "ddd(dd)", radius, initial_angle,
                                             final_angle, center.x, center.y);
    if (result != NULL) {
        if (parse_point_sequence(result, array, "") < 0) {
            PyErr_Format(PyExc_RuntimeError,
                         "Unable to parse return value (%S) from bend function.", result);
        }
        Py_DECREF(result);
    }
    return array;
}

#include "cell_object.cpp"
#include "curve_object.cpp"
#include "flexpath_object.cpp"
#include "gdswriter_object.cpp"
#include "label_object.cpp"
#include "library_object.cpp"
#include "polygon_object.cpp"
#include "raithdata_object.cpp"
#include "rawcell_object.cpp"
#include "reference_object.cpp"
#include "repetition_object.cpp"
#include "robustpath_object.cpp"

static PyObject* rectangle_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_corner1;
    PyObject* py_corner2;
    Vec2 corner1;
    Vec2 corner2;
    unsigned long layer = 0;
    unsigned long datatype = 0;
    const char* keywords[] = {"corner1", "corner2", "layer", "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|kk:rectangle", (char**)keywords, &py_corner1,
                                     &py_corner2, &layer, &datatype))
        return NULL;
    if (parse_point(py_corner1, corner1, "corner1") != 0 ||
        parse_point(py_corner2, corner2, "corner2") != 0)
        return NULL;
    PolygonObject* result = PyObject_New(PolygonObject, &polygon_object_type);
    result = (PolygonObject*)PyObject_Init((PyObject*)result, &polygon_object_type);
    result->polygon = (Polygon*)allocate_clear(sizeof(Polygon));
    *result->polygon = rectangle(corner1, corner2, make_tag(layer, datatype));
    result->polygon->owner = result;
    return (PyObject*)result;
}

static PyObject* cross_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_center;
    Vec2 center;
    double full_size;
    double arm_width;
    unsigned long layer = 0;
    unsigned long datatype = 0;
    const char* keywords[] = {"center", "full_size", "arm_width", "layer", "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Odd|kk:cross", (char**)keywords, &py_center,
                                     &full_size, &arm_width, &layer, &datatype))
        return NULL;
    if (parse_point(py_center, center, "center") != 0) return NULL;
    PolygonObject* result = PyObject_New(PolygonObject, &polygon_object_type);
    result = (PolygonObject*)PyObject_Init((PyObject*)result, &polygon_object_type);
    result->polygon = (Polygon*)allocate_clear(sizeof(Polygon));
    *result->polygon = cross(center, full_size, arm_width, make_tag(layer, datatype));
    result->polygon->owner = result;
    return (PyObject*)result;
}

static PyObject* regular_polygon_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_center;
    Vec2 center;
    double side_length;
    long sides;
    double rotation = 0;
    unsigned long layer = 0;
    unsigned long datatype = 0;
    const char* keywords[] = {"center", "side_length", "sides", "rotation",
                              "layer",  "datatype",    NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Odl|dkk:regular_polygon", (char**)keywords,
                                     &py_center, &side_length, &sides, &rotation, &layer,
                                     &datatype))
        return NULL;
    if (parse_point(py_center, center, "center") != 0) return NULL;
    if (side_length <= 0) {
        PyErr_SetString(PyExc_ValueError, "Argument side_length must be positive.");
        return NULL;
    }
    if (sides <= 2) {
        PyErr_SetString(PyExc_ValueError, "Argument sides must be greater than 2.");
        return NULL;
    }
    PolygonObject* result = PyObject_New(PolygonObject, &polygon_object_type);
    result = (PolygonObject*)PyObject_Init((PyObject*)result, &polygon_object_type);
    result->polygon = (Polygon*)allocate_clear(sizeof(Polygon));
    *result->polygon =
        regular_polygon(center, side_length, sides, rotation, make_tag(layer, datatype));
    result->polygon->owner = result;
    return (PyObject*)result;
}

static PyObject* ellipse_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_center;
    PyObject* py_radius;
    PyObject* py_inner_radius = Py_None;
    Vec2 center;
    Vec2 radius;
    Vec2 inner_radius = {-1, -1};
    double initial_angle = 0;
    double final_angle = 0;
    double tolerance = 0.01;
    unsigned long layer = 0;
    unsigned long datatype = 0;
    const char* keywords[] = {"center",        "radius",      "inner_radius",
                              "initial_angle", "final_angle", "tolerance",
                              "layer",         "datatype",    NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|Odddkk:ellipse", (char**)keywords, &py_center,
                                     &py_radius, &py_inner_radius, &initial_angle, &final_angle,
                                     &tolerance, &layer, &datatype))
        return NULL;
    if (parse_point(py_center, center, "center") != 0) return NULL;
    if (parse_point(py_radius, radius, "radius") != 0) {
        PyErr_Clear();
        radius.x = radius.y = PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert radius to float.");
            return NULL;
        }
    }
    if (py_inner_radius != Py_None &&
        parse_point(py_inner_radius, inner_radius, "inner_radius") != 0) {
        PyErr_Clear();
        inner_radius.x = inner_radius.y = PyFloat_AsDouble(py_inner_radius);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert inner_radius to float.");
            return NULL;
        }
    }
    if (radius.x <= 0 || radius.y <= 0) {
        PyErr_SetString(PyExc_ValueError, "Ellipse radius must be positive.");
        return NULL;
    }
    if (tolerance <= 0) {
        PyErr_SetString(PyExc_ValueError, "Tolerance must be positive.");
        return NULL;
    }
    PolygonObject* result = PyObject_New(PolygonObject, &polygon_object_type);
    result = (PolygonObject*)PyObject_Init((PyObject*)result, &polygon_object_type);
    result->polygon = (Polygon*)allocate_clear(sizeof(Polygon));
    *result->polygon = ellipse(center, radius.x, radius.y, inner_radius.x, inner_radius.y,
                               initial_angle, final_angle, tolerance, make_tag(layer, datatype));
    result->polygon->owner = result;
    return (PyObject*)result;
}

static PyObject* racetrack_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_center;
    Vec2 center;
    double straight_length;
    double radius;
    double inner_radius = 0;
    double tolerance = 0.01;
    int vertical = 0;
    unsigned long layer = 0;
    unsigned long datatype = 0;
    const char* keywords[] = {"center",       "straight_length", "radius",
                              "inner_radius", "vertical",        "tolerance",
                              "layer",        "datatype",        NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Odd|dpdkk:racetrack", (char**)keywords,
                                     &py_center, &straight_length, &radius, &inner_radius,
                                     &vertical, &tolerance, &layer, &datatype))
        return NULL;
    if (parse_point(py_center, center, "center") != 0) return NULL;
    if (radius <= 0) {
        PyErr_SetString(PyExc_ValueError, "Radius must be positive.");
        return NULL;
    }
    if (tolerance <= 0) {
        PyErr_SetString(PyExc_ValueError, "Tolerance must be positive.");
        return NULL;
    }
    if (straight_length < 0) {
        PyErr_SetString(PyExc_ValueError, "Argument straight_length cannot be negative.");
        return NULL;
    }
    PolygonObject* result = PyObject_New(PolygonObject, &polygon_object_type);
    result = (PolygonObject*)PyObject_Init((PyObject*)result, &polygon_object_type);
    result->polygon = (Polygon*)allocate_clear(sizeof(Polygon));
    *result->polygon = racetrack(center, straight_length, radius, inner_radius, vertical > 0,
                                 tolerance, make_tag(layer, datatype));
    result->polygon->owner = result;
    return (PyObject*)result;
}

static PyObject* text_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    const char* s;
    double size;
    PyObject* py_position;
    Vec2 position;
    int vertical = 0;
    unsigned long layer = 0;
    unsigned long datatype = 0;
    const char* keywords[] = {"text", "size", "position", "vertical", "layer", "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sdO|pkk:text", (char**)keywords, &s, &size,
                                     &py_position, &vertical, &layer, &datatype))
        return NULL;
    if (parse_point(py_position, position, "position") != 0) return NULL;
    Array<Polygon*> array = {};
    text(s, size, position, vertical > 0, make_tag(layer, datatype), array);

    PyObject* result = PyList_New(array.count);
    for (uint64_t i = 0; i < array.count; i++) {
        PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
        obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
        obj->polygon = array[i];
        array[i]->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }
    free_allocation(array.items);
    return result;
}

static PyObject* contour_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_data;
    double level = 0;
    double length_scale = 1;
    double precision = 0.01;
    unsigned long layer = 0;
    unsigned long datatype = 0;
    const char* keywords[] = {"data",     "level", "length_scale", "precision", "layer",
                              "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|dddkk:contour", (char**)keywords, &py_data,
                                     &level, &length_scale, &precision, &layer, &datatype))
        return NULL;

    PyArrayObject* data_array =
        (PyArrayObject*)PyArray_FROM_OTF(py_data, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (data_array == NULL) return NULL;

    if (PyArray_NDIM(data_array) != 2) {
        PyErr_SetString(PyExc_TypeError, "Data array must have 2 dimensions.");
        Py_DECREF(data_array);
        return NULL;
    }

    npy_intp* dims = PyArray_DIMS(data_array);
    uint64_t rows = dims[0];
    uint64_t cols = dims[1];

    double* data = (double*)PyArray_DATA(data_array);

    Array<Polygon*> result_array = {};
    ErrorCode error_code = contour(data, rows, cols, level, length_scale / precision, result_array);
    Py_DECREF(data_array);

    if (return_error(error_code)) {
        for (uint64_t i = 0; i < result_array.count; i++) {
            result_array[i]->clear();
            free_allocation(result_array[i]);
        }
        result_array.clear();
        return NULL;
    }

    Tag tag = make_tag(layer, datatype);
    const Vec2 scale = {length_scale, length_scale};
    const Vec2 center = {0, 0};
    PyObject* result = PyList_New(result_array.count);
    for (uint64_t i = 0; i < result_array.count; i++) {
        Polygon* poly = result_array[i];
        poly->scale(scale, center);
        PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
        obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
        obj->polygon = poly;
        poly->tag = tag;
        poly->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }

    result_array.clear();
    return result;
}

static PyObject* offset_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_polygons;
    double distance;
    const char* join = NULL;
    double tolerance = 2;
    double precision = 0.001;
    int use_union = 0;
    unsigned long layer = 0;
    unsigned long datatype = 0;
    const char* keywords[] = {"polygons",  "distance", "join",     "tolerance", "precision",
                              "use_union", "layer",    "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Od|sddpkk:offset", (char**)keywords, &py_polygons,
                                     &distance, &join, &tolerance, &precision, &use_union, &layer,
                                     &datatype))
        return NULL;

    if (tolerance <= 0) {
        PyErr_SetString(PyExc_ValueError, "Tolerance must be positive.");
        return NULL;
    }

    if (precision <= 0) {
        PyErr_SetString(PyExc_ValueError, "Precision must be positive.");
        return NULL;
    }

    OffsetJoin offset_join = OffsetJoin::Miter;
    if (join) {
        if (strcmp(join, "miter") == 0)
            offset_join = OffsetJoin::Miter;
        else if (strcmp(join, "bevel") == 0)
            offset_join = OffsetJoin::Bevel;
        else if (strcmp(join, "round") == 0)
            offset_join = OffsetJoin::Round;
        else {
            PyErr_SetString(PyExc_RuntimeError,
                            "Argument join must be one of 'miter', 'bevel', or 'round'.");
            return NULL;
        }
    }

    Array<Polygon*> polygon_array = {};
    if (parse_polygons(py_polygons, polygon_array, "polygons") < 0) return NULL;

    Array<Polygon*> result_array = {};
    ErrorCode error_code = offset(polygon_array, distance, offset_join, tolerance, 1 / precision,
                                  use_union > 0, result_array);

    if (return_error(error_code)) {
        for (uint64_t j = 0; j < polygon_array.count; j++) {
            polygon_array[j]->clear();
            free_allocation(polygon_array[j]);
        }
        polygon_array.clear();
        for (uint64_t j = 0; j < result_array.count; j++) {
            result_array[j]->clear();
            free_allocation(result_array[j]);
        }
        result_array.clear();
        return NULL;
    }

    Tag tag = make_tag(layer, datatype);
    PyObject* result = PyList_New(result_array.count);
    for (uint64_t i = 0; i < result_array.count; i++) {
        Polygon* poly = result_array[i];
        PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
        obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
        obj->polygon = poly;
        poly->tag = tag;
        poly->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }

    for (uint64_t j = 0; j < polygon_array.count; j++) {
        polygon_array[j]->clear();
        free_allocation(polygon_array[j]);
    }
    polygon_array.clear();
    result_array.clear();

    return result;
}

static PyObject* boolean_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_polygons1;
    PyObject* py_polygons2;
    const char* operation = NULL;
    double precision = 0.001;
    unsigned long layer = 0;
    unsigned long datatype = 0;
    const char* keywords[] = {"operand1", "operand2", "operation", "precision",
                              "layer",    "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOs|dkk:boolean", (char**)keywords, &py_polygons1,
                                     &py_polygons2, &operation, &precision, &layer, &datatype))
        return NULL;

    if (precision <= 0) {
        PyErr_SetString(PyExc_ValueError, "Precision must be positive.");
        return NULL;
    }

    Operation oper;
    if (strcmp(operation, "or") == 0)
        oper = Operation::Or;
    else if (strcmp(operation, "and") == 0)
        oper = Operation::And;
    else if (strcmp(operation, "xor") == 0)
        oper = Operation::Xor;
    else if (strcmp(operation, "not") == 0)
        oper = Operation::Not;
    else {
        PyErr_SetString(PyExc_RuntimeError,
                        "Argument operation must be one of 'or', 'and', 'xor', or 'not'.");
        return NULL;
    }

    Array<Polygon*> polygon_array1 = {};
    Array<Polygon*> polygon_array2 = {};
    if (parse_polygons(py_polygons1, polygon_array1, "operand1") < 0) return NULL;
    if (parse_polygons(py_polygons2, polygon_array2, "operand2") < 0) {
        for (uint64_t j = 0; j < polygon_array1.count; j++) {
            polygon_array1[j]->clear();
            free_allocation(polygon_array1[j]);
        }
        polygon_array1.clear();
        return NULL;
    }

    Array<Polygon*> result_array = {};
    ErrorCode error_code =
        boolean(polygon_array1, polygon_array2, oper, 1 / precision, result_array);

    if (return_error(error_code)) {
        for (uint64_t j = 0; j < polygon_array1.count; j++) {
            polygon_array1[j]->clear();
            free_allocation(polygon_array1[j]);
        }
        polygon_array1.clear();
        for (uint64_t j = 0; j < polygon_array2.count; j++) {
            polygon_array2[j]->clear();
            free_allocation(polygon_array2[j]);
        }
        polygon_array2.clear();
        for (uint64_t j = 0; j < result_array.count; j++) {
            result_array[j]->clear();
            free_allocation(result_array[j]);
        }
        result_array.clear();
        return NULL;
    }

    Tag tag = make_tag(layer, datatype);
    PyObject* result = PyList_New(result_array.count);
    for (uint64_t i = 0; i < result_array.count; i++) {
        Polygon* poly = result_array[i];
        PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
        obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
        obj->polygon = poly;
        poly->tag = tag;
        poly->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }

    for (uint64_t j = 0; j < polygon_array1.count; j++) {
        polygon_array1[j]->clear();
        free_allocation(polygon_array1[j]);
    }
    for (uint64_t j = 0; j < polygon_array2.count; j++) {
        polygon_array2[j]->clear();
        free_allocation(polygon_array2[j]);
    }
    polygon_array1.clear();
    polygon_array2.clear();
    result_array.clear();

    return result;
}

static PyObject* slice_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_polygons;
    PyObject* py_position;
    const char* axis;
    double precision = 0.001;
    const char* keywords[] = {"polygons", "position", "axis", "precision", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOs|d:slice", (char**)keywords, &py_polygons,
                                     &py_position, &axis, &precision))
        return NULL;

    if (precision <= 0) {
        PyErr_SetString(PyExc_ValueError, "Precision must be positive.");
        return NULL;
    }

    bool x_axis;
    if (strcmp(axis, "x") == 0)
        x_axis = true;
    else if (strcmp(axis, "y") == 0)
        x_axis = false;
    else {
        PyErr_SetString(PyExc_RuntimeError, "Argument axis must be 'x' or 'y'.");
        return NULL;
    }

    double single_position;
    Array<double> positions = {};
    if (PySequence_Check(py_position)) {
        if (parse_double_sequence(py_position, positions, "position") < 0) return NULL;
        sort(positions);
    } else {
        single_position = PyFloat_AsDouble(py_position);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert position to float.");
            return NULL;
        }
        positions.items = &single_position;
        positions.count = 1;
    }

    Array<Polygon*> polygon_array = {};
    if (parse_polygons(py_polygons, polygon_array, "polygons") < 0) {
        if (positions.items != &single_position) positions.clear();
        return NULL;
    }

    PyObject* result = PyList_New(positions.count + 1);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        if (positions.items != &single_position) positions.clear();
        return NULL;
    }

    Array<PyObject*> parts = {};
    parts.ensure_slots(positions.count + 1);
    for (uint64_t s = 0; s <= positions.count; s++) {
        parts[s] = PyList_New(0);
        PyList_SET_ITEM(result, s, parts[s]);
    }

    for (uint64_t i = 0; i < polygon_array.count; i++) {
        Tag tag = polygon_array[i]->tag;
        Array<Polygon*>* slices =
            (Array<Polygon*>*)allocate_clear((positions.count + 1) * sizeof(Array<Polygon*>));
        // NOTE: slice should never result in an error
        slice(*polygon_array[i], positions, x_axis, 1 / precision, slices);
        Array<Polygon*>* slice_array = slices;
        for (uint64_t s = 0; s <= positions.count; s++, slice_array++) {
            for (uint64_t j = 0; j < slice_array->count; j++) {
                PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
                obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
                obj->polygon = slice_array->items[j];
                obj->polygon->tag = tag;
                obj->polygon->owner = obj;
                if (PyList_Append(parts[s], (PyObject*)obj) < 0) {
                    Py_DECREF(obj);
                    if (positions.items != &single_position) positions.clear();
                    PyErr_SetString(PyExc_RuntimeError, "Unable to append polygon to return list.");
                    return NULL;
                }
            }
            slice_array->clear();
        }
        polygon_array[i]->clear();
        free_allocation(polygon_array[i]);
        free_allocation(slices);
    }
    polygon_array.clear();
    if (positions.items != &single_position) positions.clear();

    return result;
}

static PyObject* inside_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_points;
    PyObject* py_polygons;
    const char* keywords[] = {"points", "polygons", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO:inside", (char**)keywords, &py_points,
                                     &py_polygons))
        return NULL;

    Array<Vec2> points = {};
    if (parse_point_sequence(py_points, points, "points") < 0) {
        points.clear();
        return NULL;
    }

    Array<Polygon*> polygons = {};
    if (parse_polygons(py_polygons, polygons, "polygons") < 0) {
        points.clear();
        return NULL;
    }

    bool* values = (bool*)allocate(points.count * sizeof(bool));
    inside(points, polygons, values);

    PyObject* result = PyTuple_New(points.count);
    for (uint64_t i = 0; i < points.count; i++) {
        PyObject* res = values[i] ? Py_True : Py_False;
        Py_INCREF(res);
        PyTuple_SET_ITEM(result, i, res);
    }

    free_allocation(values);
    for (uint64_t j = 0; j < polygons.count; j++) {
        polygons[j]->clear();
        free_allocation(polygons[j]);
    }
    polygons.clear();
    points.clear();

    return result;
}

static PyObject* all_inside_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_points;
    PyObject* py_polygons;
    const char* keywords[] = {"points", "polygons", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO:all_inside", (char**)keywords, &py_points,
                                     &py_polygons))
        return NULL;

    Array<Vec2> points = {};
    if (parse_point_sequence(py_points, points, "points") < 0) {
        points.clear();
        return NULL;
    }

    Array<Polygon*> polygons = {};
    if (parse_polygons(py_polygons, polygons, "polygons") < 0) {
        points.clear();
        return NULL;
    }

    PyObject* result = all_inside(points, polygons) ? Py_True : Py_False;

    for (uint64_t j = 0; j < polygons.count; j++) {
        polygons[j]->clear();
        free_allocation(polygons[j]);
    }
    polygons.clear();
    points.clear();

    Py_INCREF(result);
    return result;
}

static PyObject* any_inside_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_points;
    PyObject* py_polygons;
    const char* keywords[] = {"points", "polygons", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO:any_inside", (char**)keywords, &py_points,
                                     &py_polygons))
        return NULL;

    Array<Vec2> points = {};
    if (parse_point_sequence(py_points, points, "points") < 0) {
        points.clear();
        return NULL;
    }

    Array<Polygon*> polygons = {};
    if (parse_polygons(py_polygons, polygons, "polygons") < 0) {
        points.clear();
        return NULL;
    }

    PyObject* result = any_inside(points, polygons) ? Py_True : Py_False;

    for (uint64_t j = 0; j < polygons.count; j++) {
        polygons[j]->clear();
        free_allocation(polygons[j]);
    }
    polygons.clear();
    points.clear();

    Py_INCREF(result);
    return result;
}

static PyObject* create_library_objects(Library* library) {
    LibraryObject* result = PyObject_New(LibraryObject, &library_object_type);
    result = (LibraryObject*)PyObject_Init((PyObject*)result, &library_object_type);
    result->library = library;
    library->owner = result;

    Cell** cell = library->cell_array.items;
    for (uint64_t i = 0; i < library->cell_array.count; i++, cell++) {
        CellObject* cell_obj = PyObject_New(CellObject, &cell_object_type);
        cell_obj = (CellObject*)PyObject_Init((PyObject*)cell_obj, &cell_object_type);
        cell_obj->cell = *cell;
        cell_obj->cell->owner = cell_obj;

        Polygon** polygon = (*cell)->polygon_array.items;
        for (uint64_t j = 0; j < (*cell)->polygon_array.count; j++, polygon++) {
            PolygonObject* polygon_obj = PyObject_New(PolygonObject, &polygon_object_type);
            polygon_obj =
                (PolygonObject*)PyObject_Init((PyObject*)polygon_obj, &polygon_object_type);
            polygon_obj->polygon = *polygon;
            polygon_obj->polygon->owner = polygon_obj;
        }

        FlexPath** flexpath = (*cell)->flexpath_array.items;
        for (uint64_t j = 0; j < (*cell)->flexpath_array.count; j++, flexpath++) {
            FlexPathObject* flexpath_obj = PyObject_New(FlexPathObject, &flexpath_object_type);
            flexpath_obj =
                (FlexPathObject*)PyObject_Init((PyObject*)flexpath_obj, &flexpath_object_type);
            flexpath_obj->flexpath = *flexpath;
            flexpath_obj->flexpath->owner = flexpath_obj;
        }

        RobustPath** robustpath = (*cell)->robustpath_array.items;
        for (uint64_t j = 0; j < (*cell)->robustpath_array.count; j++, robustpath++) {
            RobustPathObject* robustpath_obj =
                PyObject_New(RobustPathObject, &robustpath_object_type);
            robustpath_obj = (RobustPathObject*)PyObject_Init((PyObject*)robustpath_obj,
                                                              &robustpath_object_type);
            robustpath_obj->robustpath = *robustpath;
            robustpath_obj->robustpath->owner = robustpath_obj;
        }

        Reference** reference = (*cell)->reference_array.items;
        for (uint64_t j = 0; j < (*cell)->reference_array.count; j++, reference++) {
            ReferenceObject* reference_obj = PyObject_New(ReferenceObject, &reference_object_type);
            reference_obj =
                (ReferenceObject*)PyObject_Init((PyObject*)reference_obj, &reference_object_type);
            reference_obj->reference = *reference;
            reference_obj->reference->owner = reference_obj;
        }

        Label** label = (*cell)->label_array.items;
        for (uint64_t j = 0; j < (*cell)->label_array.count; j++, label++) {
            LabelObject* label_obj = PyObject_New(LabelObject, &label_object_type);
            label_obj = (LabelObject*)PyObject_Init((PyObject*)label_obj, &label_object_type);
            label_obj->label = *label;
            label_obj->label->owner = label_obj;
        }
    }

    cell = library->cell_array.items;
    for (uint64_t i = 0; i < library->cell_array.count; i++, cell++) {
        Reference** reference = (*cell)->reference_array.items;
        for (uint64_t j = 0; j < (*cell)->reference_array.count; j++, reference++) {
            // Cell reference missing (ErrorCode::MissingReference); ignore
            if ((*reference)->type == ReferenceType::Name) continue;
            Py_INCREF((*reference)->cell->owner);
        }
    }

    return (PyObject*)result;
}

static PyObject* read_gds_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* pybytes = NULL;
    double unit = 0;
    double tolerance = 0;
    PyObject* pyfilter = Py_None;
    const char* keywords[] = {"infile", "unit", "tolerance", "filter", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|ddO:read_gds", (char**)keywords,
                                     PyUnicode_FSConverter, &pybytes, &unit, &tolerance, &pyfilter))
        return NULL;

    Set<Tag> shape_tags = {};
    Set<Tag>* shape_tags_ptr = NULL;
    if (pyfilter != Py_None) {
        if (parse_tag_sequence(pyfilter, shape_tags, "filter") < 0) {
            shape_tags.clear();
            Py_DECREF(pybytes);
            return NULL;
        }
        shape_tags_ptr = &shape_tags;
    }

    const char* filename = PyBytes_AS_STRING(pybytes);
    Library* library = (Library*)allocate_clear(sizeof(Library));
    ErrorCode error_code = ErrorCode::NoError;
    *library = read_gds(filename, unit, tolerance, shape_tags_ptr, &error_code);
    Py_DECREF(pybytes);

    shape_tags.clear();

    if (return_error(error_code)) {
        library->free_all();
        free_allocation(library);
        return NULL;
    }

    return create_library_objects(library);
}

static PyObject* read_oas_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* pybytes = NULL;
    double unit = 0;
    double tolerance = 0;
    const char* keywords[] = {"infile", "unit", "tolerance", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|dd:read_oas", (char**)keywords,
                                     PyUnicode_FSConverter, &pybytes, &unit, &tolerance))
        return NULL;

    const char* filename = PyBytes_AS_STRING(pybytes);
    Library* library = (Library*)allocate_clear(sizeof(Library));
    ErrorCode error_code = ErrorCode::NoError;
    *library = read_oas(filename, unit, tolerance, &error_code);
    Py_DECREF(pybytes);

    if (return_error(error_code)) {
        library->free_all();
        free_allocation(library);
        return NULL;
    }

    return create_library_objects(library);
}

static PyObject* read_rawcells_function(PyObject* mod, PyObject* args) {
    PyObject* pybytes = NULL;
    if (!PyArg_ParseTuple(args, "O&:read_rawcells", PyUnicode_FSConverter, &pybytes)) return NULL;
    const char* filename = PyBytes_AS_STRING(pybytes);
    ErrorCode error_code = ErrorCode::NoError;
    Map<RawCell*> map = read_rawcells(filename, &error_code);
    Py_DECREF(pybytes);
    if (return_error(error_code)) return NULL;

    PyObject* result = PyDict_New();
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return dictionary.");
        return NULL;
    }
    for (MapItem<RawCell*>* item = map.next(NULL); item; item = map.next(item)) {
        RawCellObject* rawcell_obj = PyObject_New(RawCellObject, &rawcell_object_type);
        rawcell_obj = (RawCellObject*)PyObject_Init((PyObject*)rawcell_obj, &rawcell_object_type);
        rawcell_obj->rawcell = item->value;
        rawcell_obj->rawcell->owner = rawcell_obj;
        if (PyDict_SetItemString(result, rawcell_obj->rawcell->name, (PyObject*)rawcell_obj) < 0) {
            Py_DECREF(rawcell_obj);
            Py_DECREF(result);
            map.clear();
            PyErr_SetString(PyExc_RuntimeError, "Unable to insert item into result dictionary.");
            return NULL;
        }
        Py_DECREF(rawcell_obj);
    }
    for (MapItem<RawCell*>* item = map.next(NULL); item; item = map.next(item)) {
        RawCell* rawcell = item->value;
        for (uint64_t i = 0; i < rawcell->dependencies.count; i++) {
            Py_INCREF(rawcell->dependencies[i]->owner);
        }
    }
    map.clear();
    return (PyObject*)result;
}

static PyObject* gds_units_function(PyObject* mod, PyObject* args) {
    PyObject* pybytes = NULL;
    if (!PyArg_ParseTuple(args, "O&:gds_units", PyUnicode_FSConverter, &pybytes)) return NULL;

    double unit = 0;
    double precision = 0;
    const char* filename = PyBytes_AS_STRING(pybytes);
    ErrorCode error_code = gds_units(filename, unit, precision);
    Py_DECREF(pybytes);
    if (return_error(error_code)) return NULL;

    return Py_BuildValue("dd", unit, precision);
}

static PyObject* gds_timestamp_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* pybytes = NULL;
    PyObject* pytimestamp = Py_None;
    tm* timestamp = NULL;
    tm _timestamp = {};
    const char* keywords[] = {"filename", "timestamp", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O:gds_timestamp", (char**)keywords,
                                     PyUnicode_FSConverter, &pybytes, &pytimestamp))
        return NULL;

    if (pytimestamp != Py_None) {
        if (!PyDateTime_Check(pytimestamp)) {
            PyErr_SetString(PyExc_TypeError, "Timestamp must be a datetime object.");
            Py_DECREF(pybytes);
            return NULL;
        }
        _timestamp.tm_year = PyDateTime_GET_YEAR(pytimestamp) - 1900;
        _timestamp.tm_mon = PyDateTime_GET_MONTH(pytimestamp) - 1;
        _timestamp.tm_mday = PyDateTime_GET_DAY(pytimestamp);
        _timestamp.tm_hour = PyDateTime_DATE_GET_HOUR(pytimestamp);
        _timestamp.tm_min = PyDateTime_DATE_GET_MINUTE(pytimestamp);
        _timestamp.tm_sec = PyDateTime_DATE_GET_SECOND(pytimestamp);
        timestamp = &_timestamp;
    }

    ErrorCode error_code = ErrorCode::NoError;
    const char* filename = PyBytes_AS_STRING(pybytes);
    const tm lib_tm = gds_timestamp(filename, timestamp, &error_code);
    if (return_error(error_code)) {
        Py_DECREF(pybytes);
        return NULL;
    }

    Py_DECREF(pybytes);
    return PyDateTime_FromDateAndTime(lib_tm.tm_year + 1900, lib_tm.tm_mon + 1, lib_tm.tm_mday,
                                      lib_tm.tm_hour, lib_tm.tm_min, lib_tm.tm_sec, 0);
}

static PyObject* gds_info_function(PyObject* mod, PyObject* args) {
    PyObject* pybytes = NULL;
    if (!PyArg_ParseTuple(args, "O&:gds_info", PyUnicode_FSConverter, &pybytes)) return NULL;

    LibraryInfo info = {};
    const char* filename = PyBytes_AS_STRING(pybytes);
    ErrorCode error_code = gds_info(filename, info);
    Py_DECREF(pybytes);
    if (return_error(error_code)) {
        info.clear();
        return NULL;
    }

    PyObject* result = PyDict_New();
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return object.");
        info.clear();
        return NULL;
    }

    PyObject* item = PyList_New(info.cell_names.count);
    if (!item) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create list cell_names.");
        Py_DECREF(result);
        info.clear();
        return NULL;
    }
    for (uint64_t i = 0; i < info.cell_names.count; i++) {
        PyObject* name = PyUnicode_FromString(info.cell_names[i]);
        if (!name) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to create cell name.");
            Py_DECREF(result);
            Py_DECREF(item);
            info.clear();
            return NULL;
        }
        PyList_SET_ITEM(item, i, name);
    }
    if (PyDict_SetItemString(result, "cell_names", item) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to add cell_names to return dictionary.");
        Py_DECREF(item);
        Py_DECREF(result);
        info.clear();
        return NULL;
    }

    item = build_tag_set(info.shape_tags);
    if (!item) {
        Py_DECREF(result);
        info.clear();
        return NULL;
    }
    if (PyDict_SetItemString(result, "layers_and_datatypes", item) < 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Unable to add layers_and_datatypes to return dictionary.");
        Py_DECREF(item);
        Py_DECREF(result);
        info.clear();
        return NULL;
    }

    item = build_tag_set(info.label_tags);
    if (!item) {
        Py_DECREF(result);
        info.clear();
        return NULL;
    }
    if (PyDict_SetItemString(result, "layers_and_texttypes", item) < 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Unable to add layers_and_texttypes to return dictionary.");
        Py_DECREF(item);
        Py_DECREF(result);
        info.clear();
        return NULL;
    }

    item = PyLong_FromUnsignedLongLong(info.num_polygons);
    if (!item) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create integer.");
        Py_DECREF(result);
        info.clear();
        return NULL;
    }
    if (PyDict_SetItemString(result, "num_polygons", item) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to add num_polygons to return dictionary.");
        Py_DECREF(item);
        Py_DECREF(result);
        info.clear();
        return NULL;
    }

    item = PyLong_FromUnsignedLongLong(info.num_paths);
    if (!item) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create integer.");
        Py_DECREF(result);
        info.clear();
        return NULL;
    }
    if (PyDict_SetItemString(result, "num_paths", item) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to add num_paths to return dictionary.");
        Py_DECREF(item);
        Py_DECREF(result);
        info.clear();
        return NULL;
    }

    item = PyLong_FromUnsignedLongLong(info.num_references);
    if (!item) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create integer.");
        Py_DECREF(result);
        info.clear();
        return NULL;
    }
    if (PyDict_SetItemString(result, "num_references", item) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to add num_references to return dictionary.");
        Py_DECREF(item);
        Py_DECREF(result);
        info.clear();
        return NULL;
    }

    item = PyLong_FromUnsignedLongLong(info.num_labels);
    if (!item) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create integer.");
        Py_DECREF(result);
        info.clear();
        return NULL;
    }
    if (PyDict_SetItemString(result, "num_labels", item) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to add num_labels to return dictionary.");
        Py_DECREF(item);
        Py_DECREF(result);
        info.clear();
        return NULL;
    }

    item = PyFloat_FromDouble(info.unit);
    if (!item) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create float.");
        Py_DECREF(result);
        info.clear();
        return NULL;
    }
    if (PyDict_SetItemString(result, "unit", item) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to add unit to return dictionary.");
        Py_DECREF(item);
        Py_DECREF(result);
        info.clear();
        return NULL;
    }

    item = PyFloat_FromDouble(info.precision);
    if (!item) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create float.");
        Py_DECREF(result);
        info.clear();
        return NULL;
    }
    if (PyDict_SetItemString(result, "precision", item) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to add precision to return dictionary.");
        Py_DECREF(item);
        Py_DECREF(result);
        info.clear();
        return NULL;
    }

    info.clear();
    return result;
}

static PyObject* oas_precision_function(PyObject* mod, PyObject* args) {
    PyObject* pybytes = NULL;
    if (!PyArg_ParseTuple(args, "O&:oas_precision", PyUnicode_FSConverter, &pybytes)) return NULL;

    double precision = 0;
    const char* filename = PyBytes_AS_STRING(pybytes);
    ErrorCode error_code = oas_precision(filename, precision);
    Py_DECREF(pybytes);
    if (return_error(error_code)) return NULL;

    return PyFloat_FromDouble(precision);
}

static PyObject* oas_validate_function(PyObject* mod, PyObject* args) {
    PyObject* pybytes = NULL;
    if (!PyArg_ParseTuple(args, "O&:oas_validate", PyUnicode_FSConverter, &pybytes)) return NULL;

    const char* filename = PyBytes_AS_STRING(pybytes);
    uint32_t signature = 0;
    ErrorCode error_code = ErrorCode::NoError;
    bool result = oas_validate(filename, &signature, &error_code);
    Py_DECREF(pybytes);
    if (error_code == ErrorCode::ChecksumError) {
        return Py_BuildValue("Ok", Py_None, signature);
    } else if (return_error(error_code)) {
        return NULL;
    }

    return Py_BuildValue("Ok", result ? Py_True : Py_False, signature);
}

extern "C" {

static PyMethodDef gdstk_methods[] = {
    {"rectangle", (PyCFunction)rectangle_function, METH_VARARGS | METH_KEYWORDS,
     rectangle_function_doc},
    {"cross", (PyCFunction)cross_function, METH_VARARGS | METH_KEYWORDS, cross_function_doc},
    {"regular_polygon", (PyCFunction)regular_polygon_function, METH_VARARGS | METH_KEYWORDS,
     regular_polygon_function_doc},
    {"ellipse", (PyCFunction)ellipse_function, METH_VARARGS | METH_KEYWORDS, ellipse_function_doc},
    {"racetrack", (PyCFunction)racetrack_function, METH_VARARGS | METH_KEYWORDS,
     racetrack_function_doc},
    {"text", (PyCFunction)text_function, METH_VARARGS | METH_KEYWORDS, text_function_doc},
    {"contour", (PyCFunction)contour_function, METH_VARARGS | METH_KEYWORDS, contour_function_doc},
    {"offset", (PyCFunction)offset_function, METH_VARARGS | METH_KEYWORDS, offset_function_doc},
    {"boolean", (PyCFunction)boolean_function, METH_VARARGS | METH_KEYWORDS, boolean_function_doc},
    {"slice", (PyCFunction)slice_function, METH_VARARGS | METH_KEYWORDS, slice_function_doc},
    {"inside", (PyCFunction)inside_function, METH_VARARGS | METH_KEYWORDS, inside_function_doc},
    {"all_inside", (PyCFunction)all_inside_function, METH_VARARGS | METH_KEYWORDS,
     all_inside_function_doc},
    {"any_inside", (PyCFunction)any_inside_function, METH_VARARGS | METH_KEYWORDS,
     any_inside_function_doc},
    {"read_gds", (PyCFunction)read_gds_function, METH_VARARGS | METH_KEYWORDS,
     read_gds_function_doc},
    {"read_oas", (PyCFunction)read_oas_function, METH_VARARGS | METH_KEYWORDS,
     read_oas_function_doc},
    {"read_rawcells", (PyCFunction)read_rawcells_function, METH_VARARGS,
     read_rawcells_function_doc},
    {"gds_units", (PyCFunction)gds_units_function, METH_VARARGS, gds_units_function_doc},
    {"gds_timestamp", (PyCFunction)gds_timestamp_function, METH_VARARGS | METH_KEYWORDS,
     gds_timestamp_function_doc},
    {"gds_info", (PyCFunction)gds_info_function, METH_VARARGS, gds_info_function_doc},
    {"oas_precision", (PyCFunction)oas_precision_function, METH_VARARGS,
     oas_precision_function_doc},
    {"oas_validate", (PyCFunction)oas_validate_function, METH_VARARGS, oas_validate_function_doc},
    {NULL, NULL, 0, NULL}};

static int gdstk_exec(PyObject* module) {
    if (PyModule_AddStringConstant(module, "__version__", GDSTK_VERSION) < 0) {
        Py_XDECREF(module);
        return -1;
    }

    curve_object_type.tp_dealloc = (destructor)curve_object_dealloc;
    curve_object_type.tp_init = (initproc)curve_object_init;
    curve_object_type.tp_methods = curve_object_methods;
    curve_object_type.tp_getset = curve_object_getset;
    curve_object_type.tp_str = (reprfunc)curve_object_str;

    polygon_object_type.tp_dealloc = (destructor)polygon_object_dealloc;
    polygon_object_type.tp_init = (initproc)polygon_object_init;
    polygon_object_type.tp_methods = polygon_object_methods;
    polygon_object_type.tp_getset = polygon_object_getset;
    polygon_object_type.tp_str = (reprfunc)polygon_object_str;

    reference_object_type.tp_dealloc = (destructor)reference_object_dealloc;
    reference_object_type.tp_init = (initproc)reference_object_init;
    reference_object_type.tp_methods = reference_object_methods;
    reference_object_type.tp_getset = reference_object_getset;
    reference_object_type.tp_str = (reprfunc)reference_object_str;

    raithdata_object_type.tp_dealloc = (destructor)raithdata_object_dealloc;
    raithdata_object_type.tp_init = (initproc)raithdata_object_init;
    // raithdata_object_type.tp_methods = raithdata_object_methods;
    raithdata_object_type.tp_getset = raithdata_object_getset;
    raithdata_object_type.tp_str = (reprfunc)raithdata_object_str;

    flexpath_object_type.tp_dealloc = (destructor)flexpath_object_dealloc;
    flexpath_object_type.tp_init = (initproc)flexpath_object_init;
    flexpath_object_type.tp_methods = flexpath_object_methods;
    flexpath_object_type.tp_getset = flexpath_object_getset;
    flexpath_object_type.tp_str = (reprfunc)flexpath_object_str;

    robustpath_object_type.tp_dealloc = (destructor)robustpath_object_dealloc;
    robustpath_object_type.tp_init = (initproc)robustpath_object_init;
    robustpath_object_type.tp_methods = robustpath_object_methods;
    robustpath_object_type.tp_getset = robustpath_object_getset;
    robustpath_object_type.tp_str = (reprfunc)robustpath_object_str;

    label_object_type.tp_dealloc = (destructor)label_object_dealloc;
    label_object_type.tp_init = (initproc)label_object_init;
    label_object_type.tp_methods = label_object_methods;
    label_object_type.tp_getset = label_object_getset;
    label_object_type.tp_str = (reprfunc)label_object_str;

    cell_object_type.tp_dealloc = (destructor)cell_object_dealloc;
    cell_object_type.tp_init = (initproc)cell_object_init;
    cell_object_type.tp_methods = cell_object_methods;
    cell_object_type.tp_getset = cell_object_getset;
    cell_object_type.tp_str = (reprfunc)cell_object_str;

    rawcell_object_type.tp_dealloc = (destructor)rawcell_object_dealloc;
    rawcell_object_type.tp_init = (initproc)rawcell_object_init;
    rawcell_object_type.tp_methods = rawcell_object_methods;
    rawcell_object_type.tp_getset = rawcell_object_getset;
    rawcell_object_type.tp_str = (reprfunc)rawcell_object_str;

    library_object_type.tp_dealloc = (destructor)library_object_dealloc;
    library_object_type.tp_init = (initproc)library_object_init;
    library_object_type.tp_methods = library_object_methods;
    library_object_type.tp_getset = library_object_getset;
    library_object_type.tp_as_mapping = &library_object_as_mapping;
    library_object_type.tp_str = (reprfunc)library_object_str;

    gdswriter_object_type.tp_dealloc = (destructor)gdswriter_object_dealloc;
    gdswriter_object_type.tp_init = (initproc)gdswriter_object_init;
    gdswriter_object_type.tp_methods = gdswriter_object_methods;
    // gdswriter_object_type.tp_getset = gdswriter_object_getset;
    gdswriter_object_type.tp_str = (reprfunc)gdswriter_object_str;

    repetition_object_type.tp_dealloc = (destructor)repetition_object_dealloc;
    repetition_object_type.tp_init = (initproc)repetition_object_init;
    repetition_object_type.tp_methods = repetition_object_methods;
    repetition_object_type.tp_getset = repetition_object_getset;
    repetition_object_type.tp_str = (reprfunc)repetition_object_str;

    char const* names[] = {"Library",    "Cell",       "Polygon", "RaithData",
                           "FlexPath",   "RobustPath", "Label",   "Reference",
                           "Repetition", "Curve",      "RawCell", "GdsWriter"};
    PyTypeObject* types[] = {
        &library_object_type,   &cell_object_type,      &polygon_object_type,
        &raithdata_object_type, &flexpath_object_type,  &robustpath_object_type,
        &label_object_type,     &reference_object_type, &repetition_object_type,
        &curve_object_type,     &rawcell_object_type,   &gdswriter_object_type};
    for (unsigned long i = 0; i < sizeof(types) / sizeof(types[0]); ++i) {
        if (PyType_Ready(types[i]) < 0) {
            Py_DECREF(module);
            return -1;
        }
        Py_INCREF(types[i]);
        if (PyModule_AddObject(module, names[i], (PyObject*)types[i]) < 0) {
            Py_DECREF(types[i]);
            Py_DECREF(module);
            return -1;
        }
    }

    return 0;
}

static PyModuleDef_Slot gdstk_slots[] = {{Py_mod_exec, (void*)gdstk_exec}, {0, NULL}};

static struct PyModuleDef gdstk_module = {PyModuleDef_HEAD_INIT,
                                          "_gdstk",
                                          gdstk_module_doc,
                                          0,
                                          gdstk_methods,
                                          gdstk_slots,
                                          NULL,
                                          NULL,
                                          NULL};

PyMODINIT_FUNC PyInit__gdstk(void) {
    PyDateTime_IMPORT;
    PyObject* module = PyModuleDef_Init(&gdstk_module);
    if (!module) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to initialize module.");
        return NULL;
    }
    import_array();
    return module;
}

}  // extern "C"
