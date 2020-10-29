/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <cstdio>

#include "gdstk.h"

using namespace gdstk;

void example_polygons(Cell& out_cell) {
    Vec2 points[] = {{0, 0}, {2, 2}, {2, 6}, {-6, 6}, {-6, -6}, {-4, -4}, {-4, 4}, {0, 4}};
    // This array CANNOT be appended to!
    const Array<Vec2> point_array = {.size = COUNT(points), .items = points};

    // This has to be heap-allocated so that it doesn't go out of scope once the function returns.
    // We also don't worry about leaking it at the end of the program.  The OS will take care of it.
    Polygon* poly = (Polygon*)calloc(1, sizeof(Polygon));
    poly->point_array.extend(point_array);
    out_cell.polygon_array.append(poly);
}

void example_holes(Cell& out_cell) {
    Vec2 points[] = {{0, 0}, {5, 0}, {5, 5}, {0, 5}, {0, 0},
                     {2, 2}, {2, 3}, {3, 3}, {3, 2}, {2, 2}};
    const Array<Vec2> point_array = {.size = COUNT(points), .items = points};
    Polygon* poly = (Polygon*)calloc(1, sizeof(Polygon));
    poly->point_array.extend(point_array);
    out_cell.polygon_array.append(poly);
}

void example_circles(Cell& out_cell) {
    Polygon* circle = (Polygon*)calloc(1, sizeof(Polygon));
    *circle = ellipse(Vec2{0, 0}, 2, 2, 0, 0, 0, 0, 0.01, 0, 0);
    out_cell.polygon_array.append(circle);

    Polygon* ellipse_ = (Polygon*)calloc(1, sizeof(Polygon));
    *ellipse_ = ellipse(Vec2{4, 0}, 1, 2, 0, 0, 0, 0, 1e-4, 0, 0);
    out_cell.polygon_array.append(ellipse_);

    Polygon* arc = (Polygon*)calloc(1, sizeof(Polygon));
    *arc = ellipse(Vec2{2, 4}, 2, 2, 1, 1, -0.2 * M_PI, 1.2 * M_PI, 0.01, 0, 0);
    out_cell.polygon_array.append(arc);
}

void example_curves1(Cell& out_cell) {
    Vec2 points[] = {{1, 0}, {2, 1}, {2, 2}, {0, 2}};
    const Array<Vec2> point_array = {.size = COUNT(points), .items = points};

    // Curve points will be copied to the polygons, so allocating the curve on the stack is fine.
    Curve c1 = {.tolerance = 0.01};
    c1.append(Vec2{0, 0});
    c1.segment(point_array, false);

    Polygon* p1 = (Polygon*)calloc(1, sizeof(Polygon));
    p1->point_array.extend(c1.point_array);
    out_cell.polygon_array.append(p1);

    Curve c2 = {.tolerance = 0.01};
    c2.append(Vec2{3, 1});
    c2.segment(point_array, true);

    Polygon* p2 = (Polygon*)calloc(1, sizeof(Polygon));
    p2->point_array.extend(c2.point_array);
    out_cell.polygon_array.append(p2);
}

void example_curves2(Cell& out_cell) {
    Vec2 points[] = {4 * cplx_from_angle(M_PI / 6)};
    const Array<Vec2> point_array = {.size = COUNT(points), .items = points};

    Curve c3 = {.tolerance = 0.01};
    c3.append(Vec2{0, 2});
    c3.segment(point_array, true);
    c3.arc(4, 2, M_PI / 2, -M_PI / 2, 0);

    Polygon* p3 = (Polygon*)calloc(1, sizeof(Polygon));
    p3->point_array.extend(c3.point_array);
    out_cell.polygon_array.append(p3);
}

void example_curves3(Cell& out_cell) {
    Curve c4 = {.tolerance = 1e-3};
    c4.append(Vec2{0, 0});

    Vec2 points1[] = {{0, 1}, {1, 1}, {1, 0}};
    const Array<Vec2> point_array1 = {.size = COUNT(points1), .items = points1};
    c4.cubic(point_array1, false);

    Vec2 points2[] = {{1, -1}, {1, 0}};
    const Array<Vec2> point_array2 = {.size = COUNT(points2), .items = points2};
    c4.cubic_smooth(point_array2, true);

    Vec2 points3[] = {{0.5, 1}, {1, 0}};
    const Array<Vec2> point_array3 = {.size = COUNT(points3), .items = points3};
    c4.quadratic(point_array3, true);

    Vec2 points4[] = {{1, 0}};
    const Array<Vec2> point_array4 = {.size = COUNT(points4), .items = points4};
    c4.quadratic_smooth(point_array4, true);

    Vec2 points5[] = {{4, -1}, {3, -2}, {2, -1.5}, {1, -2}, {0, -1}, {0, 0}};
    double angles[COUNT(points5) + 1] = {0};
    bool angle_constraints[COUNT(points5) + 1] = {0};
    Vec2 tension[COUNT(points5) + 1] = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    const Array<Vec2> point_array5 = {.size = COUNT(points5), .items = points5};

    c4.interpolation(point_array5, angles, angle_constraints, tension, 1, 1, false, false);

    // The last point will coincide with the first (this can be checked at runtime with
    // the `closed` method), so we remove it.
    if (c4.closed()) {
        c4.remove(c4.point_array.size - 1);
    }
    Polygon* p4 = (Polygon*)calloc(1, sizeof(Polygon));
    p4->point_array.extend(c4.point_array);
    out_cell.polygon_array.append(p4);
}

void example_transformations(Cell& out_cell) {
    Polygon* poly = (Polygon*)calloc(1, sizeof(Polygon));
    *poly = rectangle(Vec2{-2, -2}, Vec2{2, 2}, 0, 0);
    poly->rotate(M_PI / 4, Vec2{0, 0});
    poly->scale(Vec2{1, 0.5}, Vec2{0, 0});
    out_cell.polygon_array.append(poly);
}

void example_layerdatatype(Cell& out_cell) {
    int16_t layer_full_etch = 1;
    int16_t dt_full_etch = 3;
    int16_t layer_partial_etch = 2;
    int16_t dt_partial_etch = 3;
    int16_t layer_lift_off = 0;
    int16_t dt_lift_off = 7;

    Polygon* poly = (Polygon*)calloc(4, sizeof(Polygon));
    poly[0] = rectangle(Vec2{-3, -3}, Vec2{3, 3}, layer_full_etch, dt_full_etch);
    poly[1] = rectangle(Vec2{-5, -3}, Vec2{-3, 3}, layer_partial_etch, dt_partial_etch);
    poly[2] = rectangle(Vec2{5, -3}, Vec2{3, 3}, layer_partial_etch, dt_partial_etch);
    poly[3] = regular_polygon(Vec2{0, 0}, 2, 6, 0, layer_lift_off, dt_lift_off);

    Polygon* p[] = {poly, poly + 1, poly + 2, poly + 3};
    const Array<Polygon*> poly_array = {.size = 4, .items = p};
    out_cell.polygon_array.extend(poly_array);
}

int main(int argc, char* argv[]) {
    char lib_name[] = "Getting started";
    Library lib = {
        .name = lib_name,
        .unit = 1e-6,
        .precision = 1e-9,
    };

    char polygons_cell_name[] = "Polygons";
    Cell polygons_cell = {.name = polygons_cell_name};
    example_polygons(polygons_cell);
    lib.cell_array.append(&polygons_cell);

    char holes_cell_name[] = "Holes";
    Cell holes_cell = {.name = holes_cell_name};
    example_holes(holes_cell);
    lib.cell_array.append(&holes_cell);

    char circles_cell_name[] = "Circles";
    Cell circles_cell = {.name = circles_cell_name};
    example_circles(circles_cell);
    lib.cell_array.append(&circles_cell);

    char curves1_cell_name[] = "Curves 1";
    Cell curves1_cell = {.name = curves1_cell_name};
    example_curves1(curves1_cell);
    lib.cell_array.append(&curves1_cell);

    char curves2_cell_name[] = "Curves 2";
    Cell curves2_cell = {.name = curves2_cell_name};
    example_curves2(curves2_cell);
    lib.cell_array.append(&curves2_cell);

    char curves3_cell_name[] = "Curves 3";
    Cell curves3_cell = {.name = curves3_cell_name};
    example_curves3(curves3_cell);
    lib.cell_array.append(&curves3_cell);

    char transformations_cell_name[] = "Transformations";
    Cell transformations_cell = {.name = transformations_cell_name};
    example_transformations(transformations_cell);
    lib.cell_array.append(&transformations_cell);

    char layerdatatype_cell_name[] = "Layer and datatype";
    Cell layerdatatype_cell = {.name = layerdatatype_cell_name};
    example_layerdatatype(layerdatatype_cell);
    lib.cell_array.append(&layerdatatype_cell);

    FILE* output = fopen("polygons.gds", "wb");
    lib.write_gds(output, 0, NULL);
    fclose(output);

    return 0;
}
