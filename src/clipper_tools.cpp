/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/
#define _USE_MATH_DEFINES

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <gdstk/allocator.hpp>
#include <gdstk/array.hpp>
#include <gdstk/clipper_tools.hpp>
#include <gdstk/polygon.hpp>
#include <gdstk/sort.hpp>
#include <gdstk/utils.hpp>
#include <gdstk/vec.hpp>

// Clipper2
#include <clipper2/clipper.h>

namespace gdstk {

using Clipper2Lib::Point64;
using Clipper2Lib::Path64;
using Clipper2Lib::Paths64;
using Clipper2Lib::PolyPath64;
using Clipper2Lib::PolyTree64;

static inline Path64 polygon_to_path(const Polygon& polygon, double scaling) {
    bool reverse = polygon.signed_area() < 0;
    uint64_t num = polygon.point_array.count;
    Path64 path(num);
    const Vec2* p = reverse ? polygon.point_array.items + num - 1 : polygon.point_array.items;
    Point64* q = &path[0];
    if (reverse) {
        for (; num > 0; num--) {
            q->x = llround(scaling * p->x);
            q->y = llround(scaling * p->y);
            p--;
            q++;
        }
    } else {
        for (; num > 0; num--) {
            q->x = llround(scaling * p->x);
            q->y = llround(scaling * p->y);
            p++;
            q++;
        }
    }
    return path;
}

static inline Paths64 polygons_to_paths(const Array<Polygon*>& polygon_array, double scaling) {
    uint64_t num = polygon_array.count;
    Paths64 paths;
    paths.reserve(num);
    for (uint64_t i = 0; i < num; i++) paths.push_back(polygon_to_path(*polygon_array[i], scaling));
    return paths;
}

static inline Polygon* path_to_polygon(const Path64& path, double scaling) {
    const double invscaling = 1 / scaling;
    uint64_t num = path.size();
    Polygon* polygon = (Polygon*)allocate_clear(sizeof(Polygon));
    polygon->point_array.ensure_slots(num);
    polygon->point_array.count = num;
    Vec2* p = polygon->point_array.items;
    const Point64* q = &path[0];
    for (; num > 0; num--) {
        p->x = invscaling * q->x;
        p->y = invscaling * q->y;
        p++;
        q++;
    }
    return polygon;
}

// NOTE: SortingPath objects are stored in gdstk::Array, which moves its items
// with realloc/plain assignment into uninitialized memory. Storing an
// std::vector iterator here is undefined behavior with MSVC debug iterators
// (_ITERATOR_DEBUG_LEVEL=2), so the minimal point is kept as an index instead.
struct SortingPath {
    const Path64* path;
    uint64_t min_index;
};

static inline bool point_less(const Point64& p1, const Point64& p2) {
    return p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y);
}

static inline bool path_less(const SortingPath& p1, const SortingPath& p2) {
    return point_less((*p1.path)[p1.min_index], (*p2.path)[p2.min_index]);
}

static Path64 link_holes(const PolyPath64* node, ErrorCode& error_code) {
    /*
    static int dbg_counter = 0;
    char dbg_name[16];
    snprintf(dbg_name, COUNT(dbg_name), "d%d.gds", dbg_counter++);
    Library dbg_library = {.name = dbg_name, .unit = 1e-6, .precision = 1e-9};
    Cell dbg_cell = {.name = dbg_name};
    dbg_library.cell_array.append(&dbg_cell);
    Polygon* dbg_poly = (Polygon*)allocate_clear(sizeof(Polygon));
    dbg_cell.polygon_array.append(dbg_poly);
    dbg_poly->tag = make_tag(1, 0);
    ClipperLib::Path dbg_path = node->Contour;
    for (ClipperLib::Path::iterator pt = dbg_path.begin(); pt != dbg_path.end(); pt++)
        dbg_poly->point_array.append(Vec2{(double)pt->X, (double)pt->Y});
    for (ClipperLib::PolyNodes::iterator child = node->Childs.begin(); child != node->Childs.end();
         child++) {
        dbg_poly = (Polygon*)allocate_clear(sizeof(Polygon));
        dbg_cell.polygon_array.append(dbg_poly);
        dbg_path = (*child)->Contour;
        for (ClipperLib::Path::iterator pt = dbg_path.begin(); pt != dbg_path.end(); pt++)
            dbg_poly->point_array.append(Vec2{(double)pt->X, (double)pt->Y});
    }
    printf("Debug library %s written with %ld polygons\n", dbg_name, dbg_cell.polygon_array.count);
    dbg_library.write_gds(dbg_name, 0, NULL);
    */
    Path64 contour = node->Polygon();
    uint64_t count = contour.size();

    Array<SortingPath> holes = {};
    holes.ensure_slots(node->Count());

    for (size_t i = 0; i < node->Count(); i++) {
        const Path64& child_poly = node->Child(i)->Polygon();
        count += child_poly.size() + 3;
        uint64_t min_index = 0;
        for (uint64_t point = 1; point < child_poly.size(); point++) {
            if (point_less(child_poly[point], child_poly[min_index])) {
                min_index = point;
            }
        }
        holes.append({&child_poly, min_index});
    }
    contour.reserve(count);

    sort(holes, path_less);

    for (uint64_t i = 0; i < holes.count; i++) {
        // holes are guaranteed to be oriented opposite to their parent
        const Path64::const_iterator hole_min = holes[i].path->begin() + holes[i].min_index;
        const Path64::const_iterator p_end = contour.end();
        Path64::iterator p_closest = contour.end();
        Path64::iterator p_prev = contour.end() - 1;
        Path64::iterator p_next = contour.begin();
        int64_t xnew = 0;
        for (; p_next != p_end; p_prev = p_next++) {
            if ((p_next->y <= hole_min->y && hole_min->y < p_prev->y) ||
                (p_prev->y < hole_min->y && hole_min->y <= p_next->y)) {
                // Avoid integer overflow in the multiplication
                double temp = (double)(p_prev->x - p_next->x) * (double)(hole_min->y - p_next->y) /
                              (double)(p_prev->y - p_next->y);
                int64_t x = p_next->x + (int64_t)llround(temp);
                if ((x > xnew || p_closest == contour.end()) && x <= hole_min->x) {
                    xnew = x;
                    p_closest = p_next;
                }
            } else if ((p_next->y == hole_min->y && p_prev->y == hole_min->y) &&
                       ((p_next->x <= hole_min->x && hole_min->x <= p_prev->x) ||
                        (p_prev->x <= hole_min->x && hole_min->x <= p_next->x))) {
                xnew = hole_min->x;
                p_closest = p_next;
                break;
            }
        }

        if (p_closest == contour.end()) {
            if (error_logger)
                fprintf(error_logger, "[GDSTK] Unable to link hole in boolean operation.\n");
            error_code = ErrorCode::BooleanError;
        } else {
            Point64 p_new(xnew, hole_min->y);
            if (p_new.x != p_closest->x || p_new.y != p_closest->y)
                p_closest = contour.insert(p_closest, p_new);
            p_closest = contour.insert(p_closest, holes[i].path->begin(), hole_min + 1);
            p_closest = contour.insert(p_closest, hole_min, holes[i].path->end());
            contour.insert(p_closest, p_new);
        }
    }
    holes.clear();
    return contour;
}

static void process_node(const PolyPath64* node, double scaling,
                         Array<Polygon*>& polygon_array, ErrorCode& error_code) {
    if (!node->IsHole()) {
        if (node->Count() > 0) {
            polygon_array.append(path_to_polygon(link_holes(node, error_code), scaling));
        } else {
            polygon_array.append(path_to_polygon(node->Polygon(), scaling));
        }
    }
    for (size_t i = 0; i < node->Count(); i++) {
        process_node(node->Child(i), scaling, polygon_array, error_code);
    }
}

static void tree_to_polygons(const PolyTree64& tree, double scaling,
                             Array<Polygon*>& polygon_array, ErrorCode& error_code) {
    for (size_t i = 0; i < tree.Count(); i++) {
        process_node(tree.Child(i), scaling, polygon_array, error_code);
    }
}

ErrorCode boolean(const Array<Polygon*>& polys1, const Array<Polygon*>& polys2, Operation operation,
                  double scaling, Array<Polygon*>& result) {
    Clipper2Lib::ClipType ct_operation = Clipper2Lib::ClipType::Union;
    switch (operation) {
        case Operation::Or:
            ct_operation = Clipper2Lib::ClipType::Union;
            break;
        case Operation::And:
            ct_operation = Clipper2Lib::ClipType::Intersection;
            break;
        case Operation::Xor:
            ct_operation = Clipper2Lib::ClipType::Xor;
            break;
        case Operation::Not:
            ct_operation = Clipper2Lib::ClipType::Difference;
    }

    Paths64 paths1 = polygons_to_paths(polys1, scaling);
    Paths64 paths2 = polygons_to_paths(polys2, scaling);

    Clipper2Lib::Clipper64 clpr;
    clpr.AddSubject(paths1);
    clpr.AddClip(paths2);

    PolyTree64 solution;
    clpr.Execute(ct_operation, Clipper2Lib::FillRule::NonZero, solution);

    ErrorCode error_code = ErrorCode::NoError;
    tree_to_polygons(solution, scaling, result, error_code);
    return error_code;
}

ErrorCode offset(const Array<Polygon*>& polygons, double distance, OffsetJoin join,
                 double tolerance, double scaling, bool use_union, Array<Polygon*>& result) {
    Clipper2Lib::JoinType jt_join = Clipper2Lib::JoinType::Bevel;
    double miter_limit = 2.0;
    double arc_tolerance = 0.0;
    switch (join) {
        case OffsetJoin::Bevel:
            jt_join = Clipper2Lib::JoinType::Bevel;
            break;
        case OffsetJoin::Miter:
            jt_join = Clipper2Lib::JoinType::Square;
            miter_limit = tolerance;
            break;
        case OffsetJoin::Round:
            jt_join = Clipper2Lib::JoinType::Round;
            arc_tolerance = fabs(distance) * scaling * (1.0 - cos(M_PI / tolerance));
    }

    Clipper2Lib::ClipperOffset clprof(miter_limit, arc_tolerance);
    Paths64 original_polys = polygons_to_paths(polygons, scaling);
    if (use_union) {
        Clipper2Lib::Clipper64 clpr;
        clpr.AddSubject(original_polys);
        Paths64 joined_polys;
        clpr.Execute(Clipper2Lib::ClipType::Union, Clipper2Lib::FillRule::NonZero, joined_polys);
        clprof.AddPaths(joined_polys, jt_join, Clipper2Lib::EndType::Polygon);
    } else {
        clprof.AddPaths(original_polys, jt_join, Clipper2Lib::EndType::Polygon);
    }

    PolyTree64 solution;
    clprof.Execute(distance * scaling, solution);

    ErrorCode error_code = ErrorCode::NoError;
    tree_to_polygons(solution, scaling, result, error_code);
    return error_code;
}

ErrorCode slice(const Polygon& polygon, const Array<double>& positions, bool x_axis, double scaling,
                Array<Polygon*>* result) {
    ErrorCode error_code = ErrorCode::NoError;
    Paths64 subj;
    subj.push_back(polygon_to_path(polygon, scaling));

    const Path64& subj_path = subj[0];
    Clipper2Lib::Rect64 bb = Clipper2Lib::GetBounds(subj_path);

    Paths64 clip(1, Path64(4));
    clip[0][0].x = clip[0][3].x = bb.left;
    clip[0][1].x = clip[0][2].x = bb.right;
    clip[0][0].y = clip[0][1].y = bb.top;
    clip[0][2].y = clip[0][3].y = bb.bottom;

    int64_t pos = x_axis ? bb.left : bb.top;
    for (uint64_t i = 0; i <= positions.count; i++) {
        if (x_axis) {
            clip[0][0].x = clip[0][3].x = pos;
            pos = i < positions.count ? llround(scaling * positions[i]) : bb.right;
            clip[0][1].x = clip[0][2].x = pos;
            if (clip[0][1].x == clip[0][0].x) continue;
        } else {
            clip[0][0].y = clip[0][1].y = pos;
            pos = i < positions.count ? llround(scaling * positions[i]) : bb.bottom;
            clip[0][2].y = clip[0][3].y = pos;
            if (clip[0][2].y == clip[0][0].y) continue;
        }

        Clipper2Lib::Clipper64 clpr;
        clpr.AddSubject(subj);
        clpr.AddClip(clip);

        PolyTree64 solution;
        clpr.Execute(Clipper2Lib::ClipType::Intersection, Clipper2Lib::FillRule::NonZero, solution);

        tree_to_polygons(solution, scaling, result[i], error_code);
    }

    return error_code;
}

}  // namespace gdstk
