/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include <gdstk/gdstk.hpp>

using namespace gdstk;

void example_polygons(Cell& out_cell) {
    Vec2 points[] = {{0, 0}, {2, 2}, {2, 6}, {-6, 6}, {-6, -6}, {-4, -4}, {-4, 4}, {0, 4}};

    // This has to be heap-allocated so that it doesn't go out of scope once
    // the function returns.  We also don't worry about leaking it at the end
    // of the program.  The OS will take care of it.
    Polygon* poly = (Polygon*)allocate_clear(sizeof(Polygon));
    poly->point_array.extend({.capacity = 0, .count = COUNT(points), .items = points});
    out_cell.polygon_array.append(poly);
}

void example_holes(Cell& out_cell) {
    Vec2 points[] = {{0, 0}, {5, 0}, {5, 5}, {0, 5}, {0, 0},
                     {2, 2}, {2, 3}, {3, 3}, {3, 2}, {2, 2}};
    Polygon* poly = (Polygon*)allocate_clear(sizeof(Polygon));
    poly->point_array.extend({.capacity = 0, .count = COUNT(points), .items = points});
    out_cell.polygon_array.append(poly);
}

void example_circles(Cell& out_cell) {
    Polygon* circle = (Polygon*)allocate_clear(sizeof(Polygon));
    *circle = ellipse(Vec2{0, 0}, 2, 2, 0, 0, 0, 0, 0.01, 0);
    out_cell.polygon_array.append(circle);

    Polygon* ellipse_ = (Polygon*)allocate_clear(sizeof(Polygon));
    *ellipse_ = ellipse(Vec2{4, 0}, 1, 2, 0, 0, 0, 0, 1e-4, 0);
    out_cell.polygon_array.append(ellipse_);

    Polygon* arc = (Polygon*)allocate_clear(sizeof(Polygon));
    *arc = ellipse(Vec2{2, 4}, 2, 2, 1, 1, -0.2 * M_PI, 1.2 * M_PI, 0.01, 0);
    out_cell.polygon_array.append(arc);
}

void example_curves1(Cell& out_cell) {
    Vec2 points[] = {{1, 0}, {2, 1}, {2, 2}, {0, 2}};

    // Curve points will be copied to the polygons, so allocating the curve on
    // the stack is fine.
    Curve c1 = {};
    c1.init(Vec2{0, 0}, 0.01);
    c1.segment({.capacity = 0, .count = COUNT(points), .items = points}, false);

    Polygon* p1 = (Polygon*)allocate_clear(sizeof(Polygon));
    p1->point_array.extend(c1.point_array);
    out_cell.polygon_array.append(p1);
    c1.clear();

    Curve c2 = {};
    c2.init(Vec2{3, 1}, 0.01);
    c2.segment({.capacity = 0, .count = COUNT(points), .items = points}, true);

    Polygon* p2 = (Polygon*)allocate_clear(sizeof(Polygon));
    p2->point_array.extend(c2.point_array);
    out_cell.polygon_array.append(p2);
    c2.clear();
}

void example_curves2(Cell& out_cell) {
    Curve c3 = {};
    c3.init(Vec2{0, 2}, 0.01);
    c3.segment(4 * cplx_from_angle(M_PI / 6), true);
    c3.arc(4, 2, M_PI / 2, -M_PI / 2, 0);

    Polygon* p3 = (Polygon*)allocate_clear(sizeof(Polygon));
    p3->point_array.extend(c3.point_array);
    out_cell.polygon_array.append(p3);
    c3.clear();
}

void example_curves3(Cell& out_cell) {
    Curve c4 = {};
    c4.init(Vec2{0, 0}, 1e-3);

    Vec2 points1[] = {{0, 1}, {1, 1}, {1, 0}};
    c4.cubic({.capacity = 0, .count = COUNT(points1), .items = points1}, false);

    Vec2 points2[] = {{1, -1}, {1, 0}};
    c4.cubic_smooth({.capacity = 0, .count = COUNT(points2), .items = points2}, true);

    Vec2 points3[] = {{0.5, 1}, {1, 0}};
    c4.quadratic({.capacity = 0, .count = COUNT(points3), .items = points3}, true);

    c4.quadratic_smooth(Vec2{1, 0}, true);

    Vec2 points4[] = {{4, -1}, {3, -2}, {2, -1.5}, {1, -2}, {0, -1}, {0, 0}};
    double angles[COUNT(points4) + 1] = {};
    bool angle_constraints[COUNT(points4) + 1] = {};
    Vec2 tension[COUNT(points4) + 1] = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};

    c4.interpolation({.capacity = 0, .count = COUNT(points4), .items = points4}, angles,
                     angle_constraints, tension, 1, 1, false, false);

    // The last point will coincide with the first (this can be checked at
    // runtime with the `closed` method), so we remove it.
    if (c4.closed()) {
        c4.remove(c4.point_array.count - 1);
    }
    Polygon* p4 = (Polygon*)allocate_clear(sizeof(Polygon));
    p4->point_array.extend(c4.point_array);
    out_cell.polygon_array.append(p4);
    c4.clear();
}

void example_transformations(Cell& out_cell) {
    Polygon* poly = (Polygon*)allocate_clear(sizeof(Polygon));
    *poly = rectangle(Vec2{-2, -2}, Vec2{2, 2}, 0);
    poly->rotate(M_PI / 4, Vec2{0, 0});
    poly->scale(Vec2{1, 0.5}, Vec2{0, 0});
    out_cell.polygon_array.append(poly);
}

void example_layerdatatype(Cell& out_cell) {
    Tag t_full_etch = make_tag(1, 3);
    Tag t_partial_etch = make_tag(2, 3);
    Tag t_lift_off = make_tag(0, 7);

    Polygon* poly[4];
    for (uint64_t i = 0; i < COUNT(poly); i++) poly[i] = (Polygon*)allocate_clear(sizeof(Polygon));

    *poly[0] = rectangle(Vec2{-3, -3}, Vec2{3, 3}, t_full_etch);
    *poly[1] = rectangle(Vec2{-5, -3}, Vec2{-3, 3}, t_partial_etch);
    *poly[2] = rectangle(Vec2{5, -3}, Vec2{3, 3}, t_partial_etch);
    *poly[3] = regular_polygon(Vec2{0, 0}, 2, 6, 0, t_lift_off);

    out_cell.polygon_array.extend({.capacity = 0, .count = COUNT(poly), .items = poly});
}

int main(int argc, char* argv[]) {
    Library lib = {};
    lib.init("Getting started", 1e-6, 1e-9);

    Cell* polygons_cell = (Cell*)allocate_clear(sizeof(Cell));
    polygons_cell->name = copy_string("Polygons", NULL);
    example_polygons(*polygons_cell);
    lib.cell_array.append(polygons_cell);

    Cell* holes_cell = (Cell*)allocate_clear(sizeof(Cell));
    holes_cell->name = copy_string("Holes", NULL);
    example_holes(*holes_cell);
    lib.cell_array.append(holes_cell);

    Cell* circles_cell = (Cell*)allocate_clear(sizeof(Cell));
    circles_cell->name = copy_string("Circles", NULL);
    example_circles(*circles_cell);
    lib.cell_array.append(circles_cell);

    Cell* curves1_cell = (Cell*)allocate_clear(sizeof(Cell));
    curves1_cell->name = copy_string("Curves 1", NULL);
    example_curves1(*curves1_cell);
    lib.cell_array.append(curves1_cell);

    Cell* curves2_cell = (Cell*)allocate_clear(sizeof(Cell));
    curves2_cell->name = copy_string("Curves 2", NULL);
    example_curves2(*curves2_cell);
    lib.cell_array.append(curves2_cell);

    Cell* curves3_cell = (Cell*)allocate_clear(sizeof(Cell));
    curves3_cell->name = copy_string("Curves 3", NULL);
    example_curves3(*curves3_cell);
    lib.cell_array.append(curves3_cell);

    Cell* transformations_cell = (Cell*)allocate_clear(sizeof(Cell));
    transformations_cell->name = copy_string("Transformations", NULL);
    example_transformations(*transformations_cell);
    lib.cell_array.append(transformations_cell);

    Cell* layerdatatype_cell = (Cell*)allocate_clear(sizeof(Cell));
    layerdatatype_cell->name = copy_string("Layer and datatype", NULL);
    example_layerdatatype(*layerdatatype_cell);
    lib.cell_array.append(layerdatatype_cell);

    lib.write_gds("polygons.gds", 0, NULL);

    lib.free_all();
    return 0;
}
