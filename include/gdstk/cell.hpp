/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_CELL
#define GDSTK_HEADER_CELL

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "array.hpp"
#include "flexpath.hpp"
#include "label.hpp"
#include "map.hpp"
#include "polygon.hpp"
#include "reference.hpp"
#include "robustpath.hpp"
#include "set.hpp"
#include "style.hpp"
#include "tagmap.hpp"

namespace gdstk {

// Must return true if the first argument is ordered before (is less than) the
// second argument.
typedef bool (*PolygonComparisonFunction)(Polygon* const&, Polygon* const&);

// This structure is used for caching bounding box and convex hull results from
// cells.  This is a snapshot of the cells at a specific point in time.  It
// must be invalidated whenever the cell contents changes.
struct GeometryInfo {
    Array<Vec2> convex_hull;
    Vec2 bounding_box_min;
    Vec2 bounding_box_max;

    // These flags indicate whether the convex hull and bounding box values are
    // valid, even when convex_hull.count == 0 or bounding_box_min.x >
    // bounding_box_max.x (empty cell)
    bool convex_hull_valid;
    bool bounding_box_valid;

    void clear() {
        convex_hull.clear();
        convex_hull_valid = false;
        bounding_box_valid = false;
    }
};

struct Cell {
    // NULL-terminated string with cell name.  The GDSII specification allows
    // only ASCII-encoded strings.  The OASIS specification restricts the
    // allowed characters to the range 0x21–0x7E (the space character, 0x20, is
    // forbidden).  Gdstk does NOT enforce either rule.
    //
    // Cells in a library are identified by their name, so names must be
    // unique.  This rule is not enforced in Gdstk, but it is assumed to be
    // followed (various hash maps are built based on cell names).
    char* name;

    // Elements should be added to (or removed from) the cell using these arrays
    Array<Polygon*> polygon_array;
    Array<Reference*> reference_array;
    Array<FlexPath*> flexpath_array;
    Array<RobustPath*> robustpath_array;
    Array<Label*> label_array;

    Property* properties;

    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void init(const char* name_) { name = copy_string(name_, NULL); }

    void print(bool all) const;

    void clear();

    void free_all() {
        for (uint64_t j = 0; j < polygon_array.count; j++) {
            polygon_array[j]->clear();
            free_allocation(polygon_array[j]);
        }
        for (uint64_t j = 0; j < flexpath_array.count; j++) {
            flexpath_array[j]->clear();
            free_allocation(flexpath_array[j]);
        }
        for (uint64_t j = 0; j < robustpath_array.count; j++) {
            robustpath_array[j]->clear();
            free_allocation(robustpath_array[j]);
        }
        for (uint64_t j = 0; j < reference_array.count; j++) {
            reference_array[j]->clear();
            free_allocation(reference_array[j]);
        }
        for (uint64_t j = 0; j < label_array.count; j++) {
            label_array[j]->clear();
            free_allocation(label_array[j]);
        }
        clear();
    }

    // Bounding box corners are returned in min and max.  For an empty cell,
    // return min.x > max.x.  Internally, this function simply calls the
    // caching version with an empty cache.
    void bounding_box(Vec2& min, Vec2& max) const;
    // Caching version of the bounding box calculation.  The returned
    // GeometryInfo is guaranteed to have the information about this cell
    // instance (bounding_box_valid == true).
    GeometryInfo bounding_box(Map<GeometryInfo>& cache) const;

    // The convex hull of the cell is appended to result (it doesn't need to be
    // empty).  Internally, this function simply calls the caching version with
    // an empty cache.
    void convex_hull(Array<Vec2>& result) const;
    // Caching version of the convex hull calculation.
    GeometryInfo convex_hull(Map<GeometryInfo>& cache) const;

    // This cell instance must be zeroed before copy_from.  If a new_name is
    // NULL, use the same name as the source cell.  If deep_copy == true, new
    // elements (polygons, paths, references, and labels) are allocated and
    // copied from the source cell.  Otherwise, the same pointers are used.
    void copy_from(const Cell& cell, const char* new_name, bool deep_copy);

    // Append a (newly allocated) copy of all the polygons in the cell to
    // result.  If paths are included, their polygonal representation is
    // calculated and also appended.  Polygons from references are included up
    // to depth levels, i.e., if depth == 0, no polygons from references are
    // included, depth == 1 includes polygons from referenced cells (with their
    // transformation properly applied), but not from references thereof, and
    // so on.  Depth < 0, removes the limit in the recursion depth.  If filter
    // is true, only polygons with the indicated tag are appended.
    void get_polygons(bool apply_repetitions, bool include_paths, int64_t depth, bool filter,
                      Tag tag, Array<Polygon*>& result) const;

    // Similar to get_polygons, but for paths and labels.
    void get_flexpaths(bool apply_repetitions, int64_t depth, bool filter, Tag tag,
                       Array<FlexPath*>& result) const;
    void get_robustpaths(bool apply_repetitions, int64_t depth, bool filter, Tag tag,
                         Array<RobustPath*>& result) const;
    void get_labels(bool apply_repetitions, int64_t depth, bool filter, Tag tag,
                    Array<Label*>& result) const;

    // Insert all dependencies in result.  Dependencies are cells that appear
    // in this cell's references. If recursive, include the whole dependency
    // tree (dependencies of dependencies).
    void get_dependencies(bool recursive, Map<Cell*>& result) const;
    void get_raw_dependencies(bool recursive, Map<RawCell*>& result) const;

    // Append all tags found in polygons and paths/labels to result.
    // References are not included in the result.
    void get_shape_tags(Set<Tag>& result) const;
    void get_label_tags(Set<Tag>& result) const;

    // Transform a cell hierarchy into a flat cell, with no dependencies, by
    // inserting the elements from this cell's references directly into the
    // cell (with the corresponding transformations).  Removed references are
    // appended to removed_references.
    void flatten(bool apply_repetitions, Array<Reference*>& removed_references);

    // Change the tags of all elements in this cell.  Map keys are the current
    // tags and map values are the desired new tags.  Elements in references
    // are not remapped (use get_dependencies to loop over and remap them).
    void remap_tags(const TagMap& map);

    // These functions output the cell and its contents in the GDSII and SVG
    // formats.  They are not supposed to be called by the user.  Use
    // Library.write_gds and Cell.write_svg instead.
    ErrorCode to_gds(FILE* out, double scaling, uint64_t max_points, double precision,
                     const tm* timestamp) const;
    ErrorCode to_svg(FILE* out, double scaling, uint32_t precision, const char* attributes,
                     PolygonComparisonFunction comp) const;

    // Output this cell to filename in SVG format.  The geometry is drawn in
    // the default units (px), but can be scaled freely.  Argument precision
    // defines the maximum desired precision for floating point representation
    // in the SVG file.  Arguments shape_style and label_style, if not NULL,
    // can de used to customize the SVG style of elements by tag.  If
    // background is not NULL, it should be a valid SVG color for the image
    // background.  Argument pad defines the margin (in px) added around the
    // cell bounding box, unless pad_as_percentage == true, in which case it is
    // interpreted as a percentage of the largest bounding box dimension.
    // Argument comp in to_svg can be used to sort the polygons in the SVG
    // output, which affects their draw order.
    ErrorCode write_svg(const char* filename, double scaling, uint32_t precision,
                        StyleMap* shape_style, StyleMap* label_style, const char* background,
                        double pad, bool pad_as_percentage, PolygonComparisonFunction comp) const;
};

}  // namespace gdstk

#endif
