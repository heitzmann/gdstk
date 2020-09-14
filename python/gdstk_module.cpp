/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define PYDEBUG(o)                             \
    PyObject_Print((PyObject*)(o), stdout, 0); \
    printf("\n");                              \
    fflush(stdout);

#define CellObject_Check(o) PyObject_TypeCheck((o), &cell_object_type)
#define FlexPathObject_Check(o) PyObject_TypeCheck((o), &flexpath_object_type)
#define LabelObject_Check(o) PyObject_TypeCheck((o), &label_object_type)
#define LibraryObject_Check(o) PyObject_TypeCheck((o), &library_object_type)
#define GdsWriterObject_Check(o) PyObject_TypeCheck((o), &gdswriter_object_type)
#define PolygonObject_Check(o) PyObject_TypeCheck((o), &polygon_object_type)
#define RawCellObject_Check(o) PyObject_TypeCheck((o), &rawcell_object_type)
#define ReferenceObject_Check(o) PyObject_TypeCheck((o), &reference_object_type)
#define RobustPathObject_Check(o) PyObject_TypeCheck((o), &robustpath_object_type)

#define __STDC_FORMAT_MACROS
#include <cinttypes>

#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>

#include "docstrings.cpp"
#include "gdstk.h"

using namespace gdstk;

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
                                         Py_TPFLAGS_DEFAULT,
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
                                           Py_TPFLAGS_DEFAULT,
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
                                             Py_TPFLAGS_DEFAULT,
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
                                            Py_TPFLAGS_DEFAULT,
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
                                              Py_TPFLAGS_DEFAULT,
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
                                         Py_TPFLAGS_DEFAULT,
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
                                        Py_TPFLAGS_DEFAULT,
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
                                           Py_TPFLAGS_DEFAULT,
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
                                           Py_TPFLAGS_DEFAULT,
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
                                             Py_TPFLAGS_DEFAULT,
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
    Array<Vec2> array = {0};
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
    Array<Vec2> array = {0};
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
    Array<Vec2> array = {0};
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
#include "rawcell_object.cpp"
#include "reference_object.cpp"
#include "robustpath_object.cpp"

static PyObject* rectangle_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_corner1;
    PyObject* py_corner2;
    Vec2 corner1;
    Vec2 corner2;
    short layer = 0;
    short datatype = 0;
    const char* keywords[] = {"corner1", "corner2", "layer", "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|hh:rectangle", (char**)keywords, &py_corner1,
                                     &py_corner2, &layer, &datatype))
        return NULL;
    if (parse_point(py_corner1, corner1, "corner1") != 0 ||
        parse_point(py_corner2, corner2, "corner2") != 0)
        return NULL;
    PolygonObject* result = PyObject_New(PolygonObject, &polygon_object_type);
    result = (PolygonObject*)PyObject_Init((PyObject*)result, &polygon_object_type);
    result->polygon = (Polygon*)calloc(1, sizeof(Polygon));
    *result->polygon = rectangle(corner1, corner2, layer, datatype);
    result->polygon->owner = result;
    return (PyObject*)result;
}

static PyObject* cross_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_center;
    Vec2 center;
    double full_size;
    double arm_width;
    short layer = 0;
    short datatype = 0;
    const char* keywords[] = {"center", "full_size", "arm_width", "layer", "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Odd|hh:cross", (char**)keywords, &py_center,
                                     &full_size, &arm_width, &layer, &datatype))
        return NULL;
    if (parse_point(py_center, center, "center") != 0) return NULL;
    PolygonObject* result = PyObject_New(PolygonObject, &polygon_object_type);
    result = (PolygonObject*)PyObject_Init((PyObject*)result, &polygon_object_type);
    result->polygon = (Polygon*)calloc(1, sizeof(Polygon));
    *result->polygon = cross(center, full_size, arm_width, layer, datatype);
    result->polygon->owner = result;
    return (PyObject*)result;
}

static PyObject* regular_polygon_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_center;
    Vec2 center;
    double side_length;
    long sides;
    double rotation = 0;
    short layer = 0;
    short datatype = 0;
    const char* keywords[] = {"center", "side_length", "sides", "rotation",
                              "layer",  "datatype",    NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Odl|dhh:regular_polygon", (char**)keywords,
                                     &py_center, &side_length, &sides, &rotation, &layer,
                                     &datatype))
        return NULL;
    if (parse_point(py_center, center, "center") != 0) return NULL;
    PolygonObject* result = PyObject_New(PolygonObject, &polygon_object_type);
    result = (PolygonObject*)PyObject_Init((PyObject*)result, &polygon_object_type);
    result->polygon = (Polygon*)calloc(1, sizeof(Polygon));
    *result->polygon = regular_polygon(center, side_length, sides, rotation, layer, datatype);
    result->polygon->owner = result;
    return (PyObject*)result;
}

static PyObject* ellipse_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_center;
    PyObject* py_radius;
    PyObject* py_inner_radius = NULL;
    Vec2 center;
    Vec2 radius;
    Vec2 inner_radius = {-1, -1};
    double initial_angle = 0;
    double final_angle = 0;
    double tolerance = 0.01;
    short layer = 0;
    short datatype = 0;
    const char* keywords[] = {"center",        "radius",      "inner_radius",
                              "initial_angle", "final_angle", "tolerance",
                              "layer",         "datatype",    NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|Odddhh:ellipse", (char**)keywords, &py_center,
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
    PolygonObject* result = PyObject_New(PolygonObject, &polygon_object_type);
    result = (PolygonObject*)PyObject_Init((PyObject*)result, &polygon_object_type);
    result->polygon = (Polygon*)calloc(1, sizeof(Polygon));
    *result->polygon = ellipse(center, radius.x, radius.y, inner_radius.x, inner_radius.y,
                               initial_angle, final_angle, tolerance, layer, datatype);
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
    short layer = 0;
    short datatype = 0;
    const char* keywords[] = {"center",       "straight_length", "radius",
                              "inner_radius", "vertical",        "tolerance",
                              "layer",        "datatype",        NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Odd|dpdhh:racetrack", (char**)keywords,
                                     &py_center, &straight_length, &radius, &inner_radius,
                                     &vertical, &tolerance, &layer, &datatype))
        return NULL;
    if (parse_point(py_center, center, "center") != 0) return NULL;
    PolygonObject* result = PyObject_New(PolygonObject, &polygon_object_type);
    result = (PolygonObject*)PyObject_Init((PyObject*)result, &polygon_object_type);
    result->polygon = (Polygon*)calloc(1, sizeof(Polygon));
    *result->polygon = racetrack(center, straight_length, radius, inner_radius, vertical > 0,
                                 tolerance, layer, datatype);
    result->polygon->owner = result;
    return (PyObject*)result;
}

static PyObject* text_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    const char* s;
    double size;
    PyObject* py_position;
    Vec2 position;
    int vertical = 0;
    short layer = 0;
    short datatype = 0;
    const char* keywords[] = {"text", "size", "position", "vertical", "layer", "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sdO|phh:text", (char**)keywords, &s, &size,
                                     &py_position, &vertical, &layer, &datatype))
        return NULL;
    if (parse_point(py_position, position, "position") != 0) return NULL;
    Array<Polygon*> array = text(s, size, position, vertical > 0, layer, datatype);

    PyObject* result = PyList_New(array.size);
    for (Py_ssize_t i = 0; i < array.size; i++) {
        PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
        obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
        obj->polygon = array[i];
        array[i]->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }
    free(array.items);
    return result;
}

// polygon_array should be zero-initialized
static int parse_polygons(PyObject* py_polygons, Array<Polygon*>& polygon_array, const char* name) {
    if (PolygonObject_Check(py_polygons)) {
        Polygon* polygon = (Polygon*)calloc(1, sizeof(Polygon));
        polygon->copy_from(*((PolygonObject*)py_polygons)->polygon);
        polygon_array.append(polygon);
    } else if (FlexPathObject_Check(py_polygons)) {
        polygon_array = ((FlexPathObject*)py_polygons)->flexpath->to_polygons();
    } else if (RobustPathObject_Check(py_polygons)) {
        polygon_array = ((RobustPathObject*)py_polygons)->robustpath->to_polygons();
    } else if (ReferenceObject_Check(py_polygons)) {
        polygon_array = ((ReferenceObject*)py_polygons)->reference->polygons(true, -1);
    } else if (PySequence_Check(py_polygons)) {
        for (int64_t i = PySequence_Length(py_polygons) - 1; i >= 0; i--) {
            PyObject* arg = PySequence_ITEM(py_polygons, i);
            if (arg == NULL) {
                PyErr_Format(PyExc_RuntimeError,
                             "Unable to retrieve item %" PRId64 " from sequence %s.", i, name);
                for (int64_t j = polygon_array.size - 1; j >= 0; j--) {
                    polygon_array[j]->clear();
                    free(polygon_array[j]);
                }
                polygon_array.clear();
                return -1;
            }
            if (PolygonObject_Check(arg)) {
                Polygon* polygon = (Polygon*)calloc(1, sizeof(Polygon));
                polygon->copy_from(*((PolygonObject*)arg)->polygon);
                polygon_array.append(polygon);
            } else if (FlexPathObject_Check(arg)) {
                Array<Polygon*> polys = ((FlexPathObject*)arg)->flexpath->to_polygons();
                polygon_array.extend(polys);
                polys.clear();
            } else if (RobustPathObject_Check(arg)) {
                Array<Polygon*> polys = ((RobustPathObject*)arg)->robustpath->to_polygons();
                polygon_array.extend(polys);
                polys.clear();
            } else if (ReferenceObject_Check(arg)) {
                Array<Polygon*> polys = ((ReferenceObject*)arg)->reference->polygons(true, -1);
                polygon_array.extend(polys);
                polys.clear();
            } else {
                Polygon* polygon = (Polygon*)calloc(1, sizeof(Polygon));
                if (parse_point_sequence(arg, polygon->point_array, "") < 0) {
                    PyErr_Format(PyExc_RuntimeError,
                                 "Unable to parse item %" PRId64 " from sequence %s.", i, name);
                    for (int64_t j = polygon_array.size - 1; j >= 0; j--) {
                        polygon_array[j]->clear();
                        free(polygon_array[j]);
                    }
                    polygon_array.clear();
                    return -1;
                }
                polygon_array.append(polygon);
            }
            Py_DECREF(arg);
        }
    } else {
        PyErr_Format(
            PyExc_TypeError,
            "Argument %s must be a Polygon, FlexPath, RobustPath, References. "
            "It can also be a sequence where each item is one of those or a sequence of points.",
            name);
        return -1;
    }
    return polygon_array.size;
}

static PyObject* offset_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_polygons;
    double distance;
    const char* join = NULL;
    double tolerance = 2;
    double precision = 0.001;
    int use_union = 0;
    short layer = 0;
    short datatype = 0;
    const char* keywords[] = {"polygons",  "distance", "join",     "tolerance", "precision",
                              "use_union", "layer",    "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Od|sddphh:offset", (char**)keywords, &py_polygons,
                                     &distance, &join, &tolerance, &precision, &use_union, &layer,
                                     &datatype))
        return NULL;

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

    Array<Polygon*> polygon_array = {0};
    if (parse_polygons(py_polygons, polygon_array, "polygons") < 0) return NULL;

    Array<Polygon*> result_array =
        offset(polygon_array, distance, offset_join, tolerance, 1 / precision, use_union > 0);

    PyObject* result = PyList_New(result_array.size);
    for (Py_ssize_t i = 0; i < result_array.size; i++) {
        PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
        obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
        obj->polygon = result_array[i];
        result_array[i]->layer = layer;
        result_array[i]->datatype = datatype;
        result_array[i]->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }

    for (int64_t j = polygon_array.size - 1; j >= 0; j--) {
        polygon_array[j]->clear();
        free(polygon_array[j]);
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
    short layer = 0;
    short datatype = 0;
    const char* keywords[] = {"operand1", "operand2", "operation", "precision",
                              "layer",    "datatype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOs|dhh:boolean", (char**)keywords, &py_polygons1,
                                     &py_polygons2, &operation, &precision, &layer, &datatype))
        return NULL;

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

    Array<Polygon*> polygon_array1 = {0};
    Array<Polygon*> polygon_array2 = {0};
    if (parse_polygons(py_polygons1, polygon_array1, "operand1") < 0) return NULL;
    if (parse_polygons(py_polygons2, polygon_array2, "operand2") < 0) {
        for (int64_t j = polygon_array1.size - 1; j >= 0; j--) {
            polygon_array1[j]->clear();
            free(polygon_array1[j]);
        }
        polygon_array1.clear();
        return NULL;
    }

    Array<Polygon*> result_array = boolean(polygon_array1, polygon_array2, oper, 1 / precision);

    PyObject* result = PyList_New(result_array.size);
    for (Py_ssize_t i = 0; i < result_array.size; i++) {
        PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
        obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
        obj->polygon = result_array[i];
        result_array[i]->layer = layer;
        result_array[i]->datatype = datatype;
        result_array[i]->owner = obj;
        PyList_SET_ITEM(result, i, (PyObject*)obj);
    }

    for (int64_t j = polygon_array1.size - 1; j >= 0; j--) {
        polygon_array1[j]->clear();
        free(polygon_array1[j]);
    }
    for (int64_t j = polygon_array2.size - 1; j >= 0; j--) {
        polygon_array2[j]->clear();
        free(polygon_array2[j]);
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
    Array<double> positions = {0};
    if (PySequence_Check(py_position)) {
        positions.items = parse_sequence_double(py_position, positions.size, "position");
        if (positions.items == NULL) return NULL;
    } else {
        single_position = PyFloat_AsDouble(py_position);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to convert position to float.");
            return NULL;
        }
        positions.items = &single_position;
        positions.size = 1;
    }

    Array<Polygon*> polygon_array = {0};
    if (parse_polygons(py_polygons, polygon_array, "polygons") < 0) {
        if (positions.items != &single_position) positions.clear();
        return NULL;
    }

    PyObject* result = PyList_New(positions.size + 1);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return list.");
        if (positions.items != &single_position) positions.clear();
        return NULL;
    }

    Array<PyObject*> parts = {0};
    parts.ensure_slots(positions.size + 1);
    for (int64_t s = 0; s <= positions.size; s++) {
        parts[s] = PyList_New(0);
        PyList_SET_ITEM(result, s, parts[s]);
    }

    for (Py_ssize_t i = 0; i < polygon_array.size; i++) {
        int16_t layer = polygon_array[i]->layer;
        int16_t datatype = polygon_array[i]->datatype;
        Array<Polygon*>* slices = slice(*polygon_array[i], positions, x_axis, 1 / precision);
        Array<Polygon*>* slice_array = slices;
        for (int64_t s = 0; s <= positions.size; s++, slice_array++) {
            for (int64_t j = 0; j < slice_array->size; j++) {
                PolygonObject* obj = PyObject_New(PolygonObject, &polygon_object_type);
                obj = (PolygonObject*)PyObject_Init((PyObject*)obj, &polygon_object_type);
                obj->polygon = slice_array->items[j];
                obj->polygon->layer = layer;
                obj->polygon->datatype = datatype;
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
        free(polygon_array[i]);
        free(slices);
    }
    polygon_array.clear();
    if (positions.items != &single_position) positions.clear();

    return result;
}

static PyObject* inside_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* py_points;
    PyObject* py_polygons;
    const char* short_circuit = NULL;
    double precision = 0.001;
    const char* keywords[] = {"points", "polygons", "short_circuit", "precision", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|sd:inside", (char**)keywords, &py_points,
                                     &py_polygons, &short_circuit, &precision))
        return NULL;

    ShortCircuit sc;

    if (!PySequence_Check(py_points) || PySequence_Length(py_points) == 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Argument points must be a sequence of points or groups thereof.");
        return NULL;
    }

    Array<Polygon*> points = {0};
    PyObject* item = PySequence_ITEM(py_points, 0);
    Vec2 point;
    if (parse_point(item, point, "points") == 0) {
        // no groups
        Py_DECREF(item);
        sc = ShortCircuit::None;
        points.ensure_slots(1);
        points.size = 1;
        points[0] = (Polygon*)calloc(1, sizeof(Polygon));
        if (parse_point_sequence(py_points, points[0]->point_array, "") < 0) {
            free(points[0]);
            points.clear();
            PyErr_SetString(PyExc_RuntimeError,
                            "Argument points must be a sequence of points or groups thereof.");
            return NULL;
        }
    } else {
        // groups
        Py_DECREF(item);
        PyErr_Clear();
        sc = ShortCircuit::Any;
        int64_t num_groups = PySequence_Length(py_points);
        points.ensure_slots(num_groups);
        points.size = num_groups;
        for (int64_t j = 0; j < num_groups; j++) {
            points[j] = (Polygon*)calloc(1, sizeof(Polygon));
            item = PySequence_ITEM(py_points, j);
            if (parse_point_sequence(item, points[j]->point_array, "") < 0) {
                Py_DECREF(item);
                for (; j >= 0; j--) {
                    points[j]->clear();
                    free(points[j]);
                }
                points.clear();
                PyErr_SetString(PyExc_RuntimeError,
                                "Argument points must be a sequence of points or groups thereof.");
                return NULL;
            }
            Py_DECREF(item);
        }
    }

    if (short_circuit) {
        if (strcmp(short_circuit, "none") == 0)
            sc = ShortCircuit::None;
        else if (strcmp(short_circuit, "any") == 0)
            sc = ShortCircuit::Any;
        else if (strcmp(short_circuit, "all") == 0)
            sc = ShortCircuit::All;
        else {
            PyErr_SetString(PyExc_RuntimeError,
                            "Argument short_circuit must be 'none', 'any' or 'all'.");
            for (int64_t j = points.size - 1; j >= 0; j--) {
                points[j]->clear();
                free(points[j]);
            }
            points.clear();
            return NULL;
        }
    }

    Array<Polygon*> polygon_array = {0};
    if (parse_polygons(py_polygons, polygon_array, "polygons") < 0) {
        for (int64_t j = points.size - 1; j >= 0; j--) {
            points[j]->clear();
            free(points[j]);
        }
        points.clear();
        return NULL;
    }

    int64_t result_len = 0;
    bool* result_items = inside(points, polygon_array, sc, 1 / precision, result_len);

    PyObject* result = PyTuple_New(result_len);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create return tuple.");
        return NULL;
    }
    for (int64_t i = 0; i < result_len; i++) {
        if (result_items[i]) {
            Py_INCREF(Py_True);
            PyTuple_SET_ITEM(result, i, Py_True);
        } else {
            Py_INCREF(Py_False);
            PyTuple_SET_ITEM(result, i, Py_False);
        }
    }

    for (int64_t j = polygon_array.size - 1; j >= 0; j--) {
        polygon_array[j]->clear();
        free(polygon_array[j]);
    }
    polygon_array.clear();

    for (int64_t j = points.size - 1; j >= 0; j--) {
        points[j]->clear();
        free(points[j]);
    }
    points.clear();

    free(result_items);

    return result;
}

static PyObject* read_gds_function(PyObject* mod, PyObject* args, PyObject* kwds) {
    PyObject* pybytes = NULL;
    double unit = 0;
    const char* keywords[] = {"infile", "unit", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|d:read_gds", (char**)keywords,
                                     PyUnicode_FSConverter, &pybytes, &unit))
        return NULL;
    FILE* infile = fopen(PyBytes_AS_STRING(pybytes), "rb");
    if (!infile) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to open file for reading.");
        return NULL;
    }
    Library* library = (Library*)calloc(1, sizeof(Library));
    *library = read_gds(infile, unit);
    fclose(infile);

    LibraryObject* result = PyObject_New(LibraryObject, &library_object_type);
    result = (LibraryObject*)PyObject_Init((PyObject*)result, &library_object_type);
    result->library = library;
    library->owner = result;

    Cell** cell = library->cell_array.items;
    for (int64_t i = library->cell_array.size - 1; i >= 0; i--, cell++) {
        CellObject* cell_obj = PyObject_New(CellObject, &cell_object_type);
        cell_obj = (CellObject*)PyObject_Init((PyObject*)cell_obj, &cell_object_type);
        cell_obj->cell = *cell;
        cell_obj->cell->owner = cell_obj;

        Polygon** polygon = (*cell)->polygon_array.items;
        for (int64_t j = (*cell)->polygon_array.size - 1; j >= 0; j--, polygon++) {
            PolygonObject* polygon_obj = PyObject_New(PolygonObject, &polygon_object_type);
            polygon_obj =
                (PolygonObject*)PyObject_Init((PyObject*)polygon_obj, &polygon_object_type);
            polygon_obj->polygon = *polygon;
            polygon_obj->polygon->owner = polygon_obj;
        }

        FlexPath** flexpath = (*cell)->flexpath_array.items;
        for (int64_t j = (*cell)->flexpath_array.size - 1; j >= 0; j--, flexpath++) {
            FlexPathObject* flexpath_obj = PyObject_New(FlexPathObject, &flexpath_object_type);
            flexpath_obj =
                (FlexPathObject*)PyObject_Init((PyObject*)flexpath_obj, &flexpath_object_type);
            flexpath_obj->flexpath = *flexpath;
            flexpath_obj->flexpath->owner = flexpath_obj;
        }

        RobustPath** robustpath = (*cell)->robustpath_array.items;
        for (int64_t j = (*cell)->robustpath_array.size - 1; j >= 0; j--, robustpath++) {
            RobustPathObject* robustpath_obj =
                PyObject_New(RobustPathObject, &robustpath_object_type);
            robustpath_obj = (RobustPathObject*)PyObject_Init((PyObject*)robustpath_obj,
                                                              &robustpath_object_type);
            robustpath_obj->robustpath = *robustpath;
            robustpath_obj->robustpath->owner = robustpath_obj;
        }

        Reference** reference = (*cell)->reference_array.items;
        for (int64_t j = (*cell)->reference_array.size - 1; j >= 0; j--, reference++) {
            ReferenceObject* reference_obj = PyObject_New(ReferenceObject, &reference_object_type);
            reference_obj =
                (ReferenceObject*)PyObject_Init((PyObject*)reference_obj, &reference_object_type);
            reference_obj->reference = *reference;
            reference_obj->reference->owner = reference_obj;
        }

        Label** label = (*cell)->label_array.items;
        for (int64_t j = (*cell)->label_array.size - 1; j >= 0; j--, label++) {
            LabelObject* label_obj = PyObject_New(LabelObject, &label_object_type);
            label_obj = (LabelObject*)PyObject_Init((PyObject*)label_obj, &label_object_type);
            label_obj->label = *label;
            label_obj->label->owner = label_obj;
        }
    }

    cell = library->cell_array.items;
    for (int64_t i = library->cell_array.size - 1; i >= 0; i--, cell++) {
        Reference** reference = (*cell)->reference_array.items;
        for (int64_t j = (*cell)->reference_array.size - 1; j >= 0; j--, reference++)
            Py_INCREF((*reference)->cell->owner);
    }

    return (PyObject*)result;
}

static PyObject* read_rawcells_function(PyObject* mod, PyObject* args) {
    PyObject* pybytes = NULL;
    if (!PyArg_ParseTuple(args, "O&:read_rawcells", PyUnicode_FSConverter, &pybytes)) return NULL;
    FILE* infile = fopen(PyBytes_AS_STRING(pybytes), "rb");
    if (!infile) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to open file for reading.");
        return NULL;
    }
    Map<RawCell*> map = read_rawcells(infile);
    fclose(infile);

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
    map.clear();
    return (PyObject*)result;
}

static PyObject* gds_units_function(PyObject* mod, PyObject* args) {
    PyObject* pybytes = NULL;
    if (!PyArg_ParseTuple(args, "O&:gds_units", PyUnicode_FSConverter, &pybytes)) return NULL;
    FILE* infile = fopen(PyBytes_AS_STRING(pybytes), "rb");
    if (!infile) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to open file for reading.");
        return NULL;
    }
    double unit = 0;
    double precision = 0;
    if (gds_units(infile, unit, precision) < 0) {
        fclose(infile);
        PyErr_SetString(PyExc_RuntimeError, "Unable to get units from gds file.");
        return NULL;
    }
    fclose(infile);
    return Py_BuildValue("dd", unit, precision);
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
    {"offset", (PyCFunction)offset_function, METH_VARARGS | METH_KEYWORDS, offset_function_doc},
    {"boolean", (PyCFunction)boolean_function, METH_VARARGS | METH_KEYWORDS, boolean_function_doc},
    {"slice", (PyCFunction)slice_function, METH_VARARGS | METH_KEYWORDS, slice_function_doc},
    {"inside", (PyCFunction)inside_function, METH_VARARGS | METH_KEYWORDS, inside_function_doc},
    {"read_gds", (PyCFunction)read_gds_function, METH_VARARGS | METH_KEYWORDS,
     read_gds_function_doc},
    {"read_rawcells", (PyCFunction)read_rawcells_function, METH_VARARGS,
     read_rawcells_function_doc},
    {"gds_units", (PyCFunction)gds_units_function, METH_VARARGS, gds_units_function_doc},
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
    if (PyType_Ready(&curve_object_type) < 0) {
        Py_XDECREF(module);
        return -1;
    }
    Py_INCREF(&curve_object_type);
    if (PyModule_AddObject(module, "Curve", (PyObject*)&curve_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_XDECREF(module);
        return -1;
    }

    polygon_object_type.tp_dealloc = (destructor)polygon_object_dealloc;
    polygon_object_type.tp_init = (initproc)polygon_object_init;
    polygon_object_type.tp_methods = polygon_object_methods;
    polygon_object_type.tp_getset = polygon_object_getset;
    polygon_object_type.tp_str = (reprfunc)polygon_object_str;
    if (PyType_Ready(&polygon_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_XDECREF(module);
        return -1;
    }
    Py_INCREF(&polygon_object_type);
    if (PyModule_AddObject(module, "Polygon", (PyObject*)&polygon_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_XDECREF(module);
        return -1;
    }

    reference_object_type.tp_dealloc = (destructor)reference_object_dealloc;
    reference_object_type.tp_init = (initproc)reference_object_init;
    reference_object_type.tp_methods = reference_object_methods;
    reference_object_type.tp_getset = reference_object_getset;
    reference_object_type.tp_str = (reprfunc)reference_object_str;
    if (PyType_Ready(&reference_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_XDECREF(module);
        return -1;
    }
    Py_INCREF(&reference_object_type);
    if (PyModule_AddObject(module, "Reference", (PyObject*)&reference_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_XDECREF(module);
        return -1;
    }

    flexpath_object_type.tp_dealloc = (destructor)flexpath_object_dealloc;
    flexpath_object_type.tp_init = (initproc)flexpath_object_init;
    flexpath_object_type.tp_methods = flexpath_object_methods;
    flexpath_object_type.tp_getset = flexpath_object_getset;
    flexpath_object_type.tp_str = (reprfunc)flexpath_object_str;
    if (PyType_Ready(&flexpath_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_XDECREF(module);
        return -1;
    }
    Py_INCREF(&flexpath_object_type);
    if (PyModule_AddObject(module, "FlexPath", (PyObject*)&flexpath_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_DECREF(&flexpath_object_type);
        Py_XDECREF(module);
        return -1;
    }

    robustpath_object_type.tp_dealloc = (destructor)robustpath_object_dealloc;
    robustpath_object_type.tp_init = (initproc)robustpath_object_init;
    robustpath_object_type.tp_methods = robustpath_object_methods;
    robustpath_object_type.tp_getset = robustpath_object_getset;
    robustpath_object_type.tp_str = (reprfunc)robustpath_object_str;
    if (PyType_Ready(&robustpath_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_DECREF(&flexpath_object_type);
        Py_XDECREF(module);
        return -1;
    }
    Py_INCREF(&robustpath_object_type);
    if (PyModule_AddObject(module, "RobustPath", (PyObject*)&robustpath_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_DECREF(&flexpath_object_type);
        Py_DECREF(&robustpath_object_type);
        Py_XDECREF(module);
        return -1;
    }

    label_object_type.tp_dealloc = (destructor)label_object_dealloc;
    label_object_type.tp_init = (initproc)label_object_init;
    label_object_type.tp_methods = label_object_methods;
    label_object_type.tp_getset = label_object_getset;
    label_object_type.tp_str = (reprfunc)label_object_str;
    if (PyType_Ready(&label_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_DECREF(&flexpath_object_type);
        Py_DECREF(&robustpath_object_type);
        Py_XDECREF(module);
        return -1;
    }
    Py_INCREF(&label_object_type);
    if (PyModule_AddObject(module, "Label", (PyObject*)&label_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_DECREF(&flexpath_object_type);
        Py_DECREF(&robustpath_object_type);
        Py_DECREF(&label_object_type);
        Py_XDECREF(module);
        return -1;
    }

    cell_object_type.tp_dealloc = (destructor)cell_object_dealloc;
    cell_object_type.tp_init = (initproc)cell_object_init;
    cell_object_type.tp_methods = cell_object_methods;
    cell_object_type.tp_getset = cell_object_getset;
    cell_object_type.tp_str = (reprfunc)cell_object_str;
    if (PyType_Ready(&cell_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_DECREF(&flexpath_object_type);
        Py_DECREF(&robustpath_object_type);
        Py_DECREF(&label_object_type);
        Py_XDECREF(module);
        return -1;
    }
    Py_INCREF(&cell_object_type);
    if (PyModule_AddObject(module, "Cell", (PyObject*)&cell_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_DECREF(&flexpath_object_type);
        Py_DECREF(&robustpath_object_type);
        Py_DECREF(&label_object_type);
        Py_DECREF(&cell_object_type);
        Py_XDECREF(module);
        return -1;
    }

    rawcell_object_type.tp_dealloc = (destructor)rawcell_object_dealloc;
    rawcell_object_type.tp_init = (initproc)rawcell_object_init;
    rawcell_object_type.tp_methods = rawcell_object_methods;
    rawcell_object_type.tp_getset = rawcell_object_getset;
    rawcell_object_type.tp_str = (reprfunc)rawcell_object_str;
    if (PyType_Ready(&rawcell_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_DECREF(&flexpath_object_type);
        Py_DECREF(&robustpath_object_type);
        Py_DECREF(&label_object_type);
        Py_DECREF(&cell_object_type);
        Py_XDECREF(module);
        return -1;
    }
    Py_INCREF(&rawcell_object_type);
    if (PyModule_AddObject(module, "RawCell", (PyObject*)&rawcell_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_DECREF(&flexpath_object_type);
        Py_DECREF(&robustpath_object_type);
        Py_DECREF(&label_object_type);
        Py_DECREF(&cell_object_type);
        Py_DECREF(&rawcell_object_type);
        Py_XDECREF(module);
        return -1;
    }

    library_object_type.tp_dealloc = (destructor)library_object_dealloc;
    library_object_type.tp_init = (initproc)library_object_init;
    library_object_type.tp_methods = library_object_methods;
    library_object_type.tp_getset = library_object_getset;
    library_object_type.tp_str = (reprfunc)library_object_str;
    if (PyType_Ready(&library_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_DECREF(&flexpath_object_type);
        Py_DECREF(&robustpath_object_type);
        Py_DECREF(&label_object_type);
        Py_DECREF(&cell_object_type);
        Py_DECREF(&rawcell_object_type);
        Py_XDECREF(module);
        return -1;
    }
    Py_INCREF(&library_object_type);
    if (PyModule_AddObject(module, "Library", (PyObject*)&library_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_DECREF(&flexpath_object_type);
        Py_DECREF(&robustpath_object_type);
        Py_DECREF(&label_object_type);
        Py_DECREF(&cell_object_type);
        Py_DECREF(&rawcell_object_type);
        Py_DECREF(&library_object_type);
        Py_XDECREF(module);
        return -1;
    }

    gdswriter_object_type.tp_dealloc = (destructor)gdswriter_object_dealloc;
    gdswriter_object_type.tp_init = (initproc)gdswriter_object_init;
    gdswriter_object_type.tp_methods = gdswriter_object_methods;
    // gdswriter_object_type.tp_getset = gdswriter_object_getset;
    gdswriter_object_type.tp_str = (reprfunc)gdswriter_object_str;
    if (PyType_Ready(&gdswriter_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_DECREF(&flexpath_object_type);
        Py_DECREF(&robustpath_object_type);
        Py_DECREF(&label_object_type);
        Py_DECREF(&cell_object_type);
        Py_DECREF(&rawcell_object_type);
        Py_DECREF(&library_object_type);
        Py_XDECREF(module);
        return -1;
    }
    Py_INCREF(&gdswriter_object_type);
    if (PyModule_AddObject(module, "GdsWriter", (PyObject*)&gdswriter_object_type) < 0) {
        Py_DECREF(&curve_object_type);
        Py_DECREF(&polygon_object_type);
        Py_DECREF(&reference_object_type);
        Py_DECREF(&flexpath_object_type);
        Py_DECREF(&robustpath_object_type);
        Py_DECREF(&label_object_type);
        Py_DECREF(&cell_object_type);
        Py_DECREF(&rawcell_object_type);
        Py_DECREF(&library_object_type);
        Py_DECREF(&gdswriter_object_type);
        Py_XDECREF(module);
        return -1;
    }

    return 0;
}

static PyModuleDef_Slot gdstk_slots[] = {{Py_mod_exec, (void*)gdstk_exec}, {0, NULL}};

static struct PyModuleDef gdstk_module = {PyModuleDef_HEAD_INIT,
                                          "gdstk",
                                          gdstk_module_doc,
                                          0,
                                          gdstk_methods,
                                          gdstk_slots,
                                          NULL,
                                          NULL,
                                          NULL};

PyMODINIT_FUNC PyInit_gdstk(void) {
    PyObject* module = PyModuleDef_Init(&gdstk_module);
    if (!module) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to initialize module.");
        return NULL;
    }
    import_array();
    return module;
}

}  // extern "C"
