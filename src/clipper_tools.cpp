/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define __STDC_FORMAT_MACROS 1
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

// Clipper
#include <clipper/clipper.hpp>

namespace gdstk {

static inline ClipperLib::Path polygon_to_path(const Polygon& polygon, double scaling) {
    bool reverse = polygon.signed_area() < 0;
    uint64_t num = polygon.point_array.count;
    ClipperLib::Path path(num);
    const Vec2* p = reverse ? polygon.point_array.items + num - 1 : polygon.point_array.items;
    ClipperLib::IntPoint* q = &path[0];
    if (reverse) {
        for (; num > 0; num--) {
            q->X = llround(scaling * p->x);
            q->Y = llround(scaling * p->y);
            p--;
            q++;
        }
    } else {
        for (; num > 0; num--) {
            q->X = llround(scaling * p->x);
            q->Y = llround(scaling * p->y);
            p++;
            q++;
        }
    }
    return path;
}

static inline ClipperLib::Paths polygons_to_paths(const Array<Polygon*>& polygon_array,
                                                  double scaling) {
    uint64_t num = polygon_array.count;
    ClipperLib::Paths paths;
    paths.reserve(num);
    for (uint64_t i = 0; i < num; i++) paths.push_back(polygon_to_path(*polygon_array[i], scaling));
    return paths;
}

static inline Polygon* path_to_polygon(const ClipperLib::Path& path, double scaling) {
    const double invscaling = 1 / scaling;
    uint64_t num = path.size();
    Polygon* polygon = (Polygon*)allocate_clear(sizeof(Polygon));
    polygon->point_array.ensure_slots(num);
    polygon->point_array.count = num;
    Vec2* p = polygon->point_array.items;
    const ClipperLib::IntPoint* q = &path[0];
    for (; num > 0; num--) {
        p->x = invscaling * q->X;
        p->y = invscaling * q->Y;
        p++;
        q++;
    }
    return polygon;
}

struct SortingPath {
    ClipperLib::Path* path;
    ClipperLib::Path::iterator min_point;
};

static inline bool point_less(const ClipperLib::IntPoint& p1, const ClipperLib::IntPoint& p2) {
    return p1.X < p2.X || (p1.X == p2.X && p1.Y < p2.Y);
}

static inline bool path_less(const SortingPath& p1, const SortingPath& p2) {
    return point_less(*p1.min_point, *p2.min_point);
}

static void link_holes(ClipperLib::PolyNode* node, ErrorCode& error_code) {
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

    Array<SortingPath> holes = {};
    holes.ensure_slots(node->ChildCount());

    ClipperLib::Path* contour = &node->Contour;
    uint64_t count = contour->size();
    for (ClipperLib::PolyNodes::iterator child = node->Childs.begin(); child != node->Childs.end();
         child++) {
        count += (*child)->Contour.size() + 3;
        SortingPath sp = {&(*child)->Contour};
        sp.min_point = sp.path->begin();
        for (ClipperLib::Path::iterator point = sp.path->begin(); point != sp.path->end();
             point++) {
            if (point_less(*point, *sp.min_point)) {
                sp.min_point = point;
            }
        }
        holes.append(sp);
    }
    contour->reserve(count);

    sort(holes, path_less);

    for (uint64_t i = 0; i < holes.count; i++) {
        // holes are guaranteed to be oriented opposite to their parent
        const ClipperLib::Path::iterator hole_min = holes[i].min_point;
        const ClipperLib::Path::iterator p_end = contour->end();
        ClipperLib::Path::iterator p_closest = contour->end();
        ClipperLib::Path::iterator p_prev = contour->end() - 1;
        ClipperLib::Path::iterator p_next = contour->begin();
        ClipperLib::cInt xnew = 0;
        for (; p_next != p_end; p_prev = p_next++) {
            if ((p_next->Y <= hole_min->Y && hole_min->Y < p_prev->Y) ||
                (p_prev->Y < hole_min->Y && hole_min->Y <= p_next->Y)) {
                // Avoid integer overflow in the multiplication
                double temp = (double)(p_prev->X - p_next->X) * (double)(hole_min->Y - p_next->Y) /
                              (double)(p_prev->Y - p_next->Y);
                ClipperLib::cInt x = p_next->X + (ClipperLib::cInt)llround(temp);
                if ((x > xnew || p_closest == p_end) && x <= hole_min->X) {
                    xnew = x;
                    p_closest = p_next;
                }
            } else if ((p_next->Y == hole_min->Y && p_prev->Y == hole_min->Y) &&
                       ((p_next->X <= hole_min->X && hole_min->X <= p_prev->X) ||
                        (p_prev->X <= hole_min->X && hole_min->X <= p_next->X))) {
                xnew = hole_min->X;
                p_closest = p_next;
                break;
            }
        }

        if (p_closest == p_end) {
            if (error_logger)
                fprintf(error_logger, "[GDSTK] Unable to link hole in boolean operation.\n");
            error_code = ErrorCode::BooleanError;
        } else {
            ClipperLib::IntPoint p_new(xnew, hole_min->Y);
            if (p_new.X != p_closest->X || p_new.Y != p_closest->Y)
                p_closest = contour->insert(p_closest, p_new);
            p_closest = contour->insert(p_closest, holes[i].path->begin(), hole_min + 1);
            p_closest = contour->insert(p_closest, hole_min, holes[i].path->end());
            contour->insert(p_closest, p_new);
        }
    }
    holes.clear();
}

static void tree_to_polygons(const ClipperLib::PolyTree& tree, double scaling,
                             Array<Polygon*>& polygon_array, ErrorCode& error_code) {
    ClipperLib::PolyNode* node = tree.GetFirst();
    while (node) {
        if (!node->IsHole()) {
            if (node->ChildCount() > 0) {
                link_holes(node, error_code);
            }
            polygon_array.append(path_to_polygon(node->Contour, scaling));
        }
        node = node->GetNext();
    }
}

static void bounding_box(ClipperLib::Path& points, ClipperLib::cInt* bb) {
    bb[0] = points[0].X;
    bb[1] = points[0].X;
    bb[2] = points[0].Y;
    bb[3] = points[0].Y;
    for (ClipperLib::Path::iterator it = points.begin(); it != points.end(); it++) {
        if (it->X < bb[0]) bb[0] = it->X;
        if (it->X > bb[1]) bb[1] = it->X;
        if (it->Y < bb[2]) bb[2] = it->Y;
        if (it->Y > bb[3]) bb[3] = it->Y;
    }
}

ErrorCode boolean(const Array<Polygon*>& polys1, const Array<Polygon*>& polys2, Operation operation,
                  double scaling, Array<Polygon*>& result) {
    ClipperLib::ClipType ct_operation = ClipperLib::ctUnion;
    switch (operation) {
        case Operation::Or:
            ct_operation = ClipperLib::ctUnion;
            break;
        case Operation::And:
            ct_operation = ClipperLib::ctIntersection;
            break;
        case Operation::Xor:
            ct_operation = ClipperLib::ctXor;
            break;
        case Operation::Not:
            ct_operation = ClipperLib::ctDifference;
    }

    ClipperLib::Paths paths1 = polygons_to_paths(polys1, scaling);
    ClipperLib::Paths paths2 = polygons_to_paths(polys2, scaling);

    // NOTE: ioStrictlySimple seems to hang on complex layouts
    // ClipperLib::Clipper clpr(ClipperLib::ioStrictlySimple);
    ClipperLib::Clipper clpr;
    clpr.AddPaths(paths1, ClipperLib::ptSubject, true);
    clpr.AddPaths(paths2, ClipperLib::ptClip, true);

    ClipperLib::PolyTree solution;
    clpr.Execute(ct_operation, solution, ClipperLib::pftNonZero, ClipperLib::pftNonZero);

    ErrorCode error_code = ErrorCode::NoError;
    tree_to_polygons(solution, scaling, result, error_code);
    return error_code;
}

ErrorCode offset(const Array<Polygon*>& polygons, double distance, OffsetJoin join,
                 double tolerance, double scaling, bool use_union, Array<Polygon*>& result) {
    ClipperLib::JoinType jt_join = ClipperLib::jtSquare;
    ClipperLib::ClipperOffset clprof;
    switch (join) {
        case OffsetJoin::Bevel:
            jt_join = ClipperLib::jtSquare;
            break;
        case OffsetJoin::Miter:
            jt_join = ClipperLib::jtMiter;
            clprof.MiterLimit = tolerance;
            break;
        case OffsetJoin::Round:
            jt_join = ClipperLib::jtRound;
            clprof.ArcTolerance = distance * scaling * (1.0 - cos(M_PI / tolerance));
    }

    ClipperLib::Paths original_polys = polygons_to_paths(polygons, scaling);
    if (use_union) {
        ClipperLib::Clipper clpr;
        clpr.AddPaths(original_polys, ClipperLib::ptSubject, true);
        ClipperLib::PolyTree joined_tree;
        clpr.Execute(ClipperLib::ctUnion, joined_tree, ClipperLib::pftNonZero,
                     ClipperLib::pftNonZero);
        ClipperLib::Paths joined_polys;
        ClipperLib::PolyTreeToPaths(joined_tree, joined_polys);
        clprof.AddPaths(joined_polys, jt_join, ClipperLib::etClosedPolygon);
    } else {
        clprof.AddPaths(original_polys, jt_join, ClipperLib::etClosedPolygon);
    }

    ClipperLib::PolyTree solution;
    clprof.Execute(solution, distance * scaling);

    ErrorCode error_code = ErrorCode::NoError;
    tree_to_polygons(solution, scaling, result, error_code);
    return error_code;
}

ErrorCode slice(const Polygon& polygon, const Array<double>& positions, bool x_axis, double scaling,
                Array<Polygon*>* result) {
    ErrorCode error_code = ErrorCode::NoError;
    ClipperLib::Paths subj;
    subj.push_back(polygon_to_path(polygon, scaling));

    ClipperLib::cInt bb[4];
    bounding_box(subj[0], bb);

    ClipperLib::Paths clip(1, ClipperLib::Path(4));
    clip[0][0].X = clip[0][3].X = bb[0];
    clip[0][1].X = clip[0][2].X = bb[1];
    clip[0][0].Y = clip[0][1].Y = bb[2];
    clip[0][2].Y = clip[0][3].Y = bb[3];

    ClipperLib::cInt pos = x_axis ? bb[0] : bb[2];
    for (uint64_t i = 0; i <= positions.count; i++) {
        if (x_axis) {
            clip[0][0].X = clip[0][3].X = pos;
            pos = i < positions.count ? llround(scaling * positions[i]) : bb[1];
            clip[0][1].X = clip[0][2].X = pos;
            if (clip[0][1].X == clip[0][0].X) continue;
        } else {
            clip[0][0].Y = clip[0][1].Y = pos;
            pos = i < positions.count ? llround(scaling * positions[i]) : bb[3];
            clip[0][2].Y = clip[0][3].Y = pos;
            if (clip[0][2].Y == clip[0][0].Y) continue;
        }

        // NOTE: ioStrictlySimple seems to hang on complex layouts
        // ClipperLib::Clipper clpr(ClipperLib::ioStrictlySimple);
        ClipperLib::Clipper clpr;
        clpr.AddPaths(subj, ClipperLib::ptSubject, true);
        clpr.AddPaths(clip, ClipperLib::ptClip, true);

        ClipperLib::PolyTree solution;
        clpr.Execute(ClipperLib::ctIntersection, solution, ClipperLib::pftNonZero,
                     ClipperLib::pftNonZero);

        tree_to_polygons(solution, scaling, result[i], error_code);
    }

    return error_code;
}

}  // namespace gdstk
