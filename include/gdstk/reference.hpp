/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_REFERENCE
#define GDSTK_HEADER_REFERENCE

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>

#include "flexpath.hpp"
#include "label.hpp"
#include "polygon.hpp"
#include "property.hpp"
#include "robustpath.hpp"
#include "vec.hpp"

namespace gdstk {

struct Cell;
struct RawCell;
struct GeometryInfo;

enum struct ReferenceType { Cell = 0, RawCell, Name };

struct Reference {
    ReferenceType type;
    // References by name or rawcell are limited in their use.  Most cell
    // functions will not work on them.
    union {
        Cell* cell;
        RawCell* rawcell;
        char* name;
    };
    Vec2 origin;
    double rotation;  // in radians
    double magnification;
    bool x_reflection;
    Repetition repetition;
    Property* properties;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void init(Cell* cell_) {
        type = ReferenceType::Cell;
        cell = cell_;
        magnification = 1.0;
    }

    void init(RawCell* rawcell_) {
        type = ReferenceType::RawCell;
        rawcell = rawcell_;
        magnification = 1.0;
    }

    void init(const char* name_) {
        type = ReferenceType::Name;
        name = copy_string(name_, NULL);
        magnification = 1.0;
    }

    void print() const;

    void clear();

    // This reference instance must be zeroed before copy_from
    void copy_from(const Reference& reference);

    // Calculate the bounding box of this reference and return the lower left
    // and upper right corners in min and max, respectively.  If the bounding
    // box cannot be calculated, return min.x > max.x.  The cached version can
    // be used to retain the cache for latter use, but it is the user
    // responsibility to invalidate the cache if any geometry changes.
    void bounding_box(Vec2& min, Vec2& max) const;
    void bounding_box(Vec2& min, Vec2& max, Map<GeometryInfo>& cache) const;

    // Append the convex hull of this reference to result.  The cached version
    // follows the same rules explained for bounding_box.
    void convex_hull(Array<Vec2>& result) const;
    void convex_hull(Array<Vec2>& result, Map<GeometryInfo>& cache) const;

    // Transformations are applied in the order of arguments, starting with
    // magnification and translating by origin at the end.  This is equivalent
    // to the transformation defined by a Reference with the same arguments.
    void transform(double mag, bool x_refl, double rot, const Vec2 orig);

    // Append the copies of this reference defined by its repetition to result.
    void apply_repetition(Array<Reference*>& result);

    // Applies the transformation and repetition defined by this reference to
    // the points in point_array, appending the results to the same array.
    void repeat_and_transform(Array<Vec2>& point_array) const;

    // These functions create and append the elements that are created by this
    // reference to the result array.  Argument depth controls how many levels
    // of references should be included (references of references); if it is
    // negative, all levels are included.  If include_paths is true, the
    // polygonal representation of paths are also included in polygons.  If
    // filter is true, only polygons in the indicated layer and data type are
    // created.
    void get_polygons(bool apply_repetitions, bool include_paths, int64_t depth, bool filter,
                      Tag tag, Array<Polygon*>& result) const;
    void get_flexpaths(bool apply_repetitions, int64_t depth, bool filter, Tag tag,
                       Array<FlexPath*>& result) const;
    void get_robustpaths(bool apply_repetitions, int64_t depth, bool filter, Tag tag,
                         Array<RobustPath*>& result) const;
    void get_labels(bool apply_repetitions, int64_t depth, bool filter, Tag tag,
                    Array<Label*>& result) const;

    // These functions output the reference in the GDSII and SVG formats.  They
    // are not supposed to be called by the user.
    ErrorCode to_gds(FILE* out, double scaling) const;
    ErrorCode to_svg(FILE* out, double scaling, uint32_t precision) const;
};

}  // namespace gdstk

#endif
