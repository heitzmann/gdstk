/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "reference.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "cell.h"
#include "rawcell.h"
#include "utils.h"

namespace gdstk {

void Reference::print() const {
    switch (type) {
        case ReferenceType::Cell:
            printf("Reference <%p> to Cell %s <%p>", this, cell->name, cell);
            break;
        case ReferenceType::RawCell:
            printf("Reference <%p> to RawCell %s <%p>", this, rawcell->name, cell);
            break;
        default:
            printf("Reference <%p> to %s", this, name);
    }
    printf(
        ", at (%lg, %lg), %lg rad, mag %lg, reflection %d, array %hd x %hd (%lg x %lg), properties <%p>, owner <%p>\n",
        origin.x, origin.y, rotation, magnification, x_reflection, columns, rows, spacing.x,
        spacing.y, properties, owner);
}

void Reference::clear() {
    if (type == ReferenceType::Name) {
        free(name);
        name = NULL;
    }
    properties_clear(properties);
    properties = NULL;
}

void Reference::copy_from(const Reference& reference) {
    type = reference.type;
    if (reference.type == ReferenceType::Name) {
        int64_t len = 1 + strlen(reference.name);
        name = (char*)malloc(sizeof(char) * len);
        memcpy(name, reference.name, len);
    } else {
        cell = reference.cell;
    }
    origin = reference.origin;
    rotation = reference.rotation;
    magnification = reference.magnification;
    x_reflection = reference.x_reflection;
    columns = reference.columns;
    rows = reference.rows;
    spacing = reference.spacing;
    properties = properties_copy(reference.properties);
}

void Reference::bounding_box(Vec2& min, Vec2& max) const {
    min.x = min.y = DBL_MAX;
    max.x = max.y = -DBL_MAX;
    // This is expensive, but necessary for a precise bounding box
    Array<Polygon*> array = polygons(true, -1);
    Polygon** poly = array.items;
    for (int64_t i = 0; i < array.size; i++, poly++) {
        Vec2 pmin, pmax;
        (*poly)->bounding_box(pmin, pmax);
        if (pmin.x < min.x) min.x = pmin.x;
        if (pmin.y < min.y) min.y = pmin.y;
        if (pmax.x > max.x) max.x = pmax.x;
        if (pmax.y > max.y) max.y = pmax.y;
        (*poly)->clear();
        free(*poly);
    }
    array.clear();
}

void Reference::transform(double mag, const Vec2 trans, bool x_refl, double rot, const Vec2 orig) {
    const int r1 = x_refl ? -1 : 1;
    const double crot = cos(rot);
    const double srot = sin(rot);
    const double x = origin.x;
    const double y = origin.y;
    origin.x = orig.x + mag * (x * crot - r1 * y * srot) + trans.x * crot - r1 * trans.y * srot;
    origin.y = orig.y + mag * (x * srot + r1 * y * crot) + trans.x * srot + r1 * trans.y * crot;
    rotation = r1 * rotation + rot;
    magnification *= mag;
    x_reflection ^= x_refl;
    spacing *= mag;
}

// Depth is passed as-is to Cell::get_polygons, where it is inspected and applied.
Array<Polygon*> Reference::polygons(bool include_paths, int64_t depth) const {
    Array<Polygon*> result = {0};
    if (type != ReferenceType::Cell) return result;

    Array<Polygon*> array = cell->get_polygons(include_paths, depth);
    uint32_t factor = rows * columns;
    result.ensure_slots(array.size * factor);
    result.size = array.size * factor;

    Polygon** src = array.items;
    Polygon** dst = result.items;
    Vec2 translation;
    for (int64_t i = 0; i < array.size; i++, src++) {
        for (int64_t r = rows - 1; r >= 0; r--) {
            translation.y = r * spacing.y;
            for (int64_t c = columns - 1; c >= 0; c--) {
                translation.x = c * spacing.x;
                // Avoid an extra allocation by moving the last polygon.
                if (r == 0 && c == 0)
                    *dst = *src;
                else {
                    *dst = (Polygon*)calloc(1, sizeof(Polygon));
                    (*dst)->copy_from(**src);
                }
                (*dst)->transform(magnification, translation, x_reflection, rotation, origin);
                dst++;
            }
        }
    }
    array.clear();

    return result;
}

Array<FlexPath*> Reference::flexpaths(int64_t depth) const {
    Array<FlexPath*> result = {0};
    if (type != ReferenceType::Cell) return result;

    Array<FlexPath*> array = cell->get_flexpaths(depth);
    uint32_t factor = rows * columns;
    result.ensure_slots(array.size * factor);
    result.size = array.size * factor;

    FlexPath** src = array.items;
    FlexPath** dst = result.items;
    Vec2 translation;
    for (int64_t i = 0; i < array.size; i++, src++) {
        for (int64_t r = rows - 1; r >= 0; r--) {
            translation.y = r * spacing.y;
            for (int64_t c = columns - 1; c >= 0; c--) {
                translation.x = c * spacing.x;
                if (r == 0 && c == 0)
                    *dst = *src;
                else {
                    *dst = (FlexPath*)calloc(1, sizeof(FlexPath));
                    (*dst)->copy_from(**src);
                }
                (*dst)->transform(magnification, translation, x_reflection, rotation, origin);
                dst++;
            }
        }
    }
    array.clear();

    return result;
}

Array<RobustPath*> Reference::robustpaths(int64_t depth) const {
    Array<RobustPath*> result = {0};
    if (type != ReferenceType::Cell) return result;

    Array<RobustPath*> array = cell->get_robustpaths(depth);
    uint32_t factor = rows * columns;
    result.ensure_slots(array.size * factor);
    result.size = array.size * factor;

    RobustPath** src = array.items;
    RobustPath** dst = result.items;
    Vec2 translation;
    for (int64_t i = 0; i < array.size; i++, src++) {
        for (int64_t r = rows - 1; r >= 0; r--) {
            translation.y = r * spacing.y;
            for (int64_t c = columns - 1; c >= 0; c--) {
                translation.x = c * spacing.x;
                if (r == 0 && c == 0)
                    *dst = *src;
                else {
                    *dst = (RobustPath*)calloc(1, sizeof(RobustPath));
                    (*dst)->copy_from(**src);
                }
                (*dst)->transform(magnification, translation, x_reflection, rotation, origin);
                dst++;
            }
        }
    }
    array.clear();

    return result;
}

Array<Label*> Reference::labels(int64_t depth) const {
    Array<Label*> result = {0};
    if (type != ReferenceType::Cell) return result;

    Array<Label*> array = cell->get_labels(depth);
    uint32_t factor = rows * columns;
    result.ensure_slots(array.size * factor);
    result.size = array.size * factor;

    Label** src = array.items;
    Label** dst = result.items;
    Vec2 translation;
    for (int64_t i = 0; i < array.size; i++, src++) {
        for (int64_t r = rows - 1; r >= 0; r--) {
            translation.y = r * spacing.y;
            for (int64_t c = columns - 1; c >= 0; c--) {
                translation.x = c * spacing.x;
                if (r == 0 && c == 0)
                    *dst = *src;
                else {
                    *dst = (Label*)calloc(1, sizeof(Label));
                    (*dst)->copy_from(**src);
                }
                (*dst)->transform(magnification, translation, x_reflection, rotation, origin);
                dst++;
            }
        }
    }
    array.clear();

    return result;
}

void Reference::to_gds(FILE* out, double scaling) const {
    bool array = (columns > 1 || rows > 1);
    double x2 = 0;
    double y2 = 0;
    double x3 = 0;
    double y3 = 0;

    const char* ref_name = type == ReferenceType::Cell
                               ? cell->name
                               : (type == ReferenceType::RawCell ? rawcell->name : name);
    int64_t len = strlen(ref_name);
    if (len % 2) len++;
    uint16_t buffer[] = {4, 0x0A00, (uint16_t)(4 + len), 0x1206};
    if (array) buffer[1] = 0x0B00;
    swap16(buffer, COUNT(buffer));
    fwrite(buffer, sizeof(uint16_t), COUNT(buffer), out);
    fwrite(ref_name, sizeof(char), len, out);

    if (array) {
        x2 = origin.x + columns * spacing.x;
        y2 = origin.y;
        x3 = origin.x;
        y3 = origin.y + rows * spacing.y;
    }

    if (rotation != 0 || magnification != 1 || x_reflection) {
        uint16_t buffer_flags[] = {6, 0x1A01, 0};
        if (x_reflection) {
            buffer_flags[2] |= 0x8000;
            if (array) y3 = 2 * origin.y - y3;
        }
        if (magnification != 1) {
            // Unsuported flag
            // if("absolute magnification") buffer_flags[2] |= 0x0004;
        }
        if (rotation != 0) {
            // Unsuported flag
            // if("absolute rotation") buffer_flags[2] |= 0x0002;
            if (array) {
                double sa = sin(rotation);
                double ca = cos(rotation);
                double dx = x2 - origin.x;
                double dy = y2 - origin.y;
                x2 = dx * ca - dy * sa + origin.x;
                y2 = dx * sa + dy * ca + origin.y;
                dx = x3 - origin.x;
                dy = y3 - origin.y;
                x3 = dx * ca - dy * sa + origin.x;
                y3 = dx * sa + dy * ca + origin.y;
            }
        }
        swap16(buffer_flags, COUNT(buffer_flags));
        fwrite(buffer_flags, sizeof(uint16_t), COUNT(buffer_flags), out);
        if (magnification != 1) {
            uint16_t buffer_mag[] = {12, 0x1B05};
            swap16(buffer_mag, COUNT(buffer_mag));
            fwrite(buffer_mag, sizeof(uint16_t), COUNT(buffer_mag), out);
            uint64_t mag_real = gdsii_real_from_double(magnification);
            swap64(&mag_real, 1);
            fwrite(&mag_real, sizeof(uint64_t), 1, out);
        }
        if (rotation != 0) {
            uint16_t buffer_rot[] = {12, 0x1C05};
            swap16(buffer_rot, COUNT(buffer_rot));
            fwrite(buffer_rot, sizeof(uint16_t), COUNT(buffer_rot), out);
            uint64_t rot_real = gdsii_real_from_double(rotation * (180.0 / M_PI));
            swap64(&rot_real, 1);
            fwrite(&rot_real, sizeof(uint64_t), 1, out);
        }
    }

    if (array) {
        uint16_t buffer_array[] = {8, 0x1302, columns, rows, 28, 0x1003};
        int32_t buffer_coord[] = {
            (int32_t)(lround(origin.x * scaling)), (int32_t)(lround(origin.y * scaling)),
            (int32_t)(lround(x2 * scaling)),       (int32_t)(lround(y2 * scaling)),
            (int32_t)(lround(x3 * scaling)),       (int32_t)(lround(y3 * scaling))};
        swap16(buffer_array, COUNT(buffer_array));
        swap32((uint32_t*)buffer_coord, COUNT(buffer_coord));
        fwrite(buffer_array, sizeof(uint16_t), COUNT(buffer_array), out);
        fwrite(buffer_coord, sizeof(int32_t), COUNT(buffer_coord), out);
    } else {
        uint16_t buffer_single[] = {12, 0x1003};
        int32_t buffer_coord[] = {(int32_t)(lround(origin.x * scaling)),
                                  (int32_t)(lround(origin.y * scaling))};
        swap16(buffer_single, COUNT(buffer_single));
        swap32((uint32_t*)buffer_coord, COUNT(buffer_coord));
        fwrite(buffer_single, sizeof(uint16_t), COUNT(buffer_single), out);
        fwrite(buffer_coord, sizeof(int32_t), COUNT(buffer_coord), out);
    }

    properties_to_gds(properties, out);

    uint16_t buffer_end[] = {4, 0x1100};
    swap16(buffer_end, COUNT(buffer_end));
    fwrite(buffer_end, sizeof(int16_t), COUNT(buffer_end), out);
}

void Reference::to_svg(FILE* out, double scaling) const {
    const char* src_name = type == ReferenceType::Cell
                               ? cell->name
                               : (type == ReferenceType::RawCell ? rawcell->name : name);
    char* ref_name = (char*)malloc(sizeof(char) * (strlen(src_name) + 1));
    // NOTE: Here be dragons if name is not ASCII.  The GDSII specification imposes ASCII-only for
    // strings, but who knows…
    char* d = ref_name;
    for (const char* c = src_name; *c != 0; c++, d++) *d = *c == '#' ? '_' : *c;
    *d = 0;

    double px = scaling * origin.x;
    double py = scaling * origin.y;
    double dx = scaling * spacing.x;
    double dy = scaling * spacing.y;
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < columns; c++) {
            fprintf(out, "<use transform=\"translate(%lf %lf)", px, py);
            if (rotation != 0) fprintf(out, " rotate(%lf)", rotation * (180.0 / M_PI));
            if (x_reflection) fputs(" scale(1 -1)", out);
            if (c > 0 || r > 0) fprintf(out, " translate(%lf %lf)", dx * c, dy * r);
            if (magnification != 1) fprintf(out, " scale(%lf)", magnification);
            fprintf(out, "\" xlink:href=\"#%s\"/>\n", ref_name);
        }
    }
    free(ref_name);
}

}  // namespace gdstk
