/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "clipper_tools.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>

#include "allocator.h"
#include "array.h"
#include "clipperlib/clipper.hpp"
#include "polygon.h"
#include "utils.h"
#include "vec.h"

namespace gdstk {

static ClipperLib::Path polygon_to_path(const Polygon& polygon, double scaling) {
    uint64_t num = polygon.point_array.count;
    ClipperLib::Path path(num);
    const Vec2* p = polygon.point_array.items;
    ClipperLib::IntPoint* q = &path[0];
    for (; num > 0; num--) {
        q->X = llround(scaling * p->x);
        q->Y = llround(scaling * p->y);
        p++;
        q++;
    }
    return path;
}

static void path_to_polygon(const ClipperLib::Path& path, double scaling, Polygon& polygon) {
    double invscaling = 1 / scaling;
    uint64_t num = path.size();
    polygon.point_array.ensure_slots(num);
    polygon.point_array.count = num;
    Vec2* p = polygon.point_array.items;
    const ClipperLib::IntPoint* q = &path[0];
    for (; num > 0; num--) {
        p->x = invscaling * q->X;
        p->y = invscaling * q->Y;
        p++;
        q++;
    }
}

static ClipperLib::Paths polygons_to_paths(const Array<Polygon*>& polygon_array, double scaling) {
    uint64_t num = polygon_array.count;
    ClipperLib::Paths paths;
    paths.reserve(num);
    for (uint64_t i = 0; i < num; i++) paths.push_back(polygon_to_path(*polygon_array[i], scaling));
    return paths;
}

static void paths_to_polygons(const ClipperLib::Paths& paths, double scaling,
                              Array<Polygon*>& polygon_array) {
    uint64_t num = paths.size();
    polygon_array.ensure_slots(num);
    Polygon** poly = polygon_array.items + polygon_array.count;
    for (uint64_t i = 0; i < num; i++, poly++) {
        *poly = (Polygon*)allocate_clear(sizeof(Polygon));
        path_to_polygon(paths[i], scaling, **poly);
    }
    polygon_array.count += num;
}

static inline bool point_compare(const ClipperLib::IntPoint& p1, const ClipperLib::IntPoint& p2) {
    return p1.X < p2.X || (p1.X == p2.X && p1.Y < p2.Y);
}

static bool path_compare(ClipperLib::Path& p1, ClipperLib::Path& p2) {
    ClipperLib::Path::iterator pt1 = min_element(p1.begin(), p1.end(), point_compare);
    ClipperLib::Path::iterator pt2 = min_element(p2.begin(), p2.end(), point_compare);
    return point_compare(*pt1, *pt2);
}

static ClipperLib::Path link_holes(ClipperLib::PolyNode* node) {
    ClipperLib::Paths holes;
    holes.reserve(node->ChildCount());

    ClipperLib::Path result = node->Contour;
    uint64_t count = result.size();
    for (ClipperLib::PolyNodes::iterator child = node->Childs.begin(); child != node->Childs.end();
         child++) {
        count += (*child)->Contour.size() + 3;
        holes.push_back((*child)->Contour);
    }
    result.reserve(count);

    sort(holes.begin(), holes.end(), path_compare);

    for (ClipperLib::Paths::iterator h = holes.begin(); h != holes.end(); h++) {
        // holes are guaranteed to be oriented opposite to their parent
        ClipperLib::Path::iterator p = min_element(h->begin(), h->end(), point_compare);
        ClipperLib::Path::iterator p1 = result.end();
        ClipperLib::Path::iterator pprev = --result.end();
        ClipperLib::Path::iterator pnext = result.begin();
        ClipperLib::cInt xnew = 0;
        for (; pnext != result.end(); pprev = pnext++) {
            if ((pnext->Y <= p->Y && p->Y < pprev->Y) || (pprev->Y < p->Y && p->Y <= pnext->Y)) {
                ClipperLib::cInt x =
                    pnext->X + ((pprev->X - pnext->X) * (p->Y - pnext->Y)) / (pprev->Y - pnext->Y);
                if ((x > xnew || p1 == result.end()) && x <= p->X) {
                    xnew = x;
                    p1 = pnext;
                }
            }
        }

        ClipperLib::IntPoint pnew(xnew, p->Y);
        if (pnew.X != p1->X || pnew.Y != p1->Y) result.insert(p1, pnew);
        result.insert(p1, h->begin(), p + 1);
        result.insert(p1, p, h->end());
        result.insert(p1, pnew);
    }

    return result;
}

static ClipperLib::Paths tree2paths(const ClipperLib::PolyTree& tree) {
    ClipperLib::Paths result;
    result.reserve(tree.ChildCount());
    ClipperLib::PolyNode* node = tree.GetFirst();
    while (node) {
        if (!node->IsHole()) {
            if (node->ChildCount() > 0)
                result.push_back(link_holes(node));
            else
                result.push_back(node->Contour);
        }
        node = node->GetNext();
    }
    return result;
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

void boolean(const Array<Polygon*>& polys1, const Array<Polygon*>& polys2, Operation operation,
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

    ClipperLib::Clipper clpr;
    clpr.AddPaths(paths1, ClipperLib::ptSubject, true);
    clpr.AddPaths(paths2, ClipperLib::ptClip, true);

    ClipperLib::PolyTree solution;
    clpr.Execute(ct_operation, solution, ClipperLib::pftNonZero, ClipperLib::pftNonZero);
    ClipperLib::Paths result_paths = tree2paths(solution);

    paths_to_polygons(result_paths, scaling, result);
}

void offset(const Array<Polygon*>& polygons, double distance, OffsetJoin join, double tol,
            double scaling, bool use_union, Array<Polygon*>& result) {
    ClipperLib::JoinType jt_join = ClipperLib::jtSquare;
    ClipperLib::ClipperOffset clprof;
    switch (join) {
        case OffsetJoin::Bevel:
            jt_join = ClipperLib::jtSquare;
            break;
        case OffsetJoin::Miter:
            jt_join = ClipperLib::jtMiter;
            clprof.MiterLimit = tol;
            break;
        case OffsetJoin::Round:
            jt_join = ClipperLib::jtRound;
            clprof.ArcTolerance = distance * scaling * (1.0 - cos(M_PI / tol));
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
    ClipperLib::Paths result_paths = tree2paths(solution);

    paths_to_polygons(result_paths, scaling, result);
}

void slice(const Polygon& polygon, const Array<double>& positions, bool x_axis, double scaling,
           Array<Polygon*>* result) {
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
        } else {
            clip[0][0].Y = clip[0][1].Y = pos;
            pos = i < positions.count ? llround(scaling * positions[i]) : bb[3];
            clip[0][2].Y = clip[0][3].Y = pos;
        }

        ClipperLib::Clipper clpr;
        clpr.AddPaths(subj, ClipperLib::ptSubject, true);
        clpr.AddPaths(clip, ClipperLib::ptClip, true);

        ClipperLib::PolyTree solution;
        clpr.Execute(ClipperLib::ctIntersection, solution, ClipperLib::pftNonZero,
                     ClipperLib::pftNonZero);

        ClipperLib::Paths result_paths = tree2paths(solution);
        paths_to_polygons(result_paths, scaling, result[i]);
    }
}

void inside(const Array<Polygon*>& groups, const Array<Polygon*>& polygons,
            ShortCircuit short_circuit, double scaling, Array<bool>& result) {
    ClipperLib::Paths groups_paths = polygons_to_paths(groups, scaling);
    ClipperLib::Paths paths = polygons_to_paths(polygons, scaling);

    uint64_t num_groups = groups.count;
    uint64_t num_polygons = polygons.count;

    ClipperLib::cInt* paths_bb =
        (ClipperLib::cInt*)allocate(sizeof(ClipperLib::cInt) * 4 * num_polygons);
    for (uint64_t p = 0; p < num_polygons; p++) bounding_box(paths[p], paths_bb + 4 * p);

    if (short_circuit == ShortCircuit::None) {
        uint64_t num = 0;
        for (uint64_t i = 0; i < num_groups; i++) num += groups[i]->point_array.count;
        result.ensure_slots(num);

        ClipperLib::cInt all_bb[4];
        all_bb[0] = paths_bb[0 * 4 + 0];
        all_bb[1] = paths_bb[0 * 4 + 1];
        all_bb[2] = paths_bb[0 * 4 + 2];
        all_bb[3] = paths_bb[0 * 4 + 3];
        for (uint64_t p = 1; p < num_polygons; p++) {
            if (all_bb[0] > paths_bb[p * 4 + 0]) all_bb[0] = paths_bb[p * 4 + 0];
            if (all_bb[1] < paths_bb[p * 4 + 1]) all_bb[1] = paths_bb[p * 4 + 1];
            if (all_bb[2] > paths_bb[p * 4 + 2]) all_bb[2] = paths_bb[p * 4 + 2];
            if (all_bb[3] < paths_bb[p * 4 + 3]) all_bb[3] = paths_bb[p * 4 + 3];
        }

        for (uint64_t i = 0; i < num_groups; i++) {
            uint64_t num_points = groups[i]->point_array.count;
            for (uint64_t j = 0; j < num_points; j++) {
                bool in = false;
                if (groups_paths[i][j].X >= all_bb[0] && groups_paths[i][j].X <= all_bb[1] &&
                    groups_paths[i][j].Y >= all_bb[2] && groups_paths[i][j].Y <= all_bb[3])
                    for (uint64_t p = 0; p < num_polygons && in == false; p++)
                        if (groups_paths[i][j].X >= paths_bb[p * 4 + 0] &&
                            groups_paths[i][j].X <= paths_bb[p * 4 + 1] &&
                            groups_paths[i][j].Y >= paths_bb[p * 4 + 2] &&
                            groups_paths[i][j].Y <= paths_bb[p * 4 + 3] &&
                            PointInPolygon(groups_paths[i][j], paths[p]) != 0)
                            in = true;
                result.append_unsafe(in);
            }
        }
    } else if (short_circuit == ShortCircuit::Any) {
        uint64_t num = num_groups;
        result.ensure_slots(num);
        for (uint64_t j = 0; j < num_groups; j++) {
            ClipperLib::cInt group_bb[4];
            bool in = false;
            uint64_t num_points = groups[j]->point_array.count;
            bounding_box(groups_paths[j], group_bb);

            for (uint64_t p = 0; p < num_polygons && in == false; p++)
                if (group_bb[0] <= paths_bb[p * 4 + 1] && group_bb[1] >= paths_bb[p * 4 + 0] &&
                    group_bb[2] <= paths_bb[p * 4 + 3] && group_bb[3] >= paths_bb[p * 4 + 2])
                    for (uint64_t i = 0; i < num_points && in == false; i++)
                        if (groups_paths[j][i].X >= paths_bb[p * 4 + 0] &&
                            groups_paths[j][i].X <= paths_bb[p * 4 + 1] &&
                            groups_paths[j][i].Y >= paths_bb[p * 4 + 2] &&
                            groups_paths[j][i].Y <= paths_bb[p * 4 + 3] &&
                            PointInPolygon(groups_paths[j][i], paths[p]) != 0)
                            in = true;
            result.append_unsafe(in);
        }
    } else if (short_circuit == ShortCircuit::All) {
        uint64_t num = num_groups;
        result.ensure_slots(num);

        ClipperLib::cInt all_bb[4];
        all_bb[0] = paths_bb[0 * 4 + 0];
        all_bb[1] = paths_bb[0 * 4 + 1];
        all_bb[2] = paths_bb[0 * 4 + 2];
        all_bb[3] = paths_bb[0 * 4 + 3];
        for (uint64_t p = 1; p < num_polygons; p++) {
            if (all_bb[0] > paths_bb[p * 4 + 0]) all_bb[0] = paths_bb[p * 4 + 0];
            if (all_bb[1] < paths_bb[p * 4 + 1]) all_bb[1] = paths_bb[p * 4 + 1];
            if (all_bb[2] > paths_bb[p * 4 + 2]) all_bb[2] = paths_bb[p * 4 + 2];
            if (all_bb[3] < paths_bb[p * 4 + 3]) all_bb[3] = paths_bb[p * 4 + 3];
        }

        for (uint64_t j = 0; j < num_groups; j++) {
            bool in = true;
            uint64_t num_points = groups[j]->point_array.count;

            for (uint64_t i = 0; i < num_points && in == true; i++) {
                bool this_in = false;
                if (groups_paths[j][i].X >= all_bb[0] && groups_paths[j][i].X <= all_bb[1] &&
                    groups_paths[j][i].Y >= all_bb[2] && groups_paths[j][i].Y <= all_bb[3])
                    for (uint64_t p = 0; p < num_polygons && this_in == false; p++)
                        if (groups_paths[j][i].X >= paths_bb[p * 4 + 0] &&
                            groups_paths[j][i].X <= paths_bb[p * 4 + 1] &&
                            groups_paths[j][i].Y >= paths_bb[p * 4 + 2] &&
                            groups_paths[j][i].Y <= paths_bb[p * 4 + 3] &&
                            PointInPolygon(groups_paths[j][i], paths[p]) != 0)
                            this_in = true;
                in = this_in;
            }
            result.append_unsafe(in);
        }
    }
    free_allocation(paths_bb);
}

}  // namespace gdstk
