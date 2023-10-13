/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <gdstk/allocator.hpp>
#include <gdstk/cell.hpp>
#include <gdstk/gdsii.hpp>
#include <gdstk/rawcell.hpp>
#include <gdstk/reference.hpp>
#include <gdstk/utils.hpp>

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
    printf(", at (%lg, %lg), %lg rad, mag %lg,%s reflected, properties <%p>, owner <%p>\n",
           origin.x, origin.y, rotation, magnification, x_reflection ? "" : " not", properties,
           owner);
    properties_print(properties);
    repetition.print();
}

void Reference::clear() {
    if (type == ReferenceType::Name) {
        free_allocation(name);
        name = NULL;
    }
    repetition.clear();
    properties_clear(properties);
}

void Reference::copy_from(const Reference& reference) {
    type = reference.type;
    if (reference.type == ReferenceType::Name) {
        name = copy_string(reference.name, NULL);
    } else {
        cell = reference.cell;
    }
    origin = reference.origin;
    rotation = reference.rotation;
    magnification = reference.magnification;
    x_reflection = reference.x_reflection;
    repetition.copy_from(reference.repetition);
    properties = properties_copy(reference.properties);
}

void Reference::repeat_and_transform(Array<Vec2>& point_array) const {
    const uint64_t num_points = point_array.count;
    if (num_points == 0) return;

    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {};

    if (repetition.type != RepetitionType::None) {
        repetition.get_extrema(offsets);
        point_array.ensure_slots((offsets.count - 1) * num_points);
        point_array.count *= offsets.count;
    } else {
        offsets.count = 1;
        offsets.items = &zero;
    }

    double ca = cos(rotation);
    double sa = sin(rotation);

    Vec2* dst = point_array.items + point_array.count - num_points;
    Vec2* off = offsets.items;
    for (uint64_t offset_count = offsets.count; offset_count > 0; offset_count--) {
        if (offset_count != 1) {
            memcpy(dst, point_array.items, num_points * sizeof(Vec2));
        }

        Vec2* p = dst;
        for (uint64_t num = num_points; num > 0; num--, p++) {
            Vec2 q = *p * magnification;
            if (x_reflection) q.y = -q.y;
            p->x = q.x * ca - q.y * sa + origin.x + off->x;
            p->y = q.x * sa + q.y * ca + origin.y + off->y;
        }

        off++;
        dst -= num_points;
    }

    if (repetition.type != RepetitionType::None) offsets.clear();
}

void Reference::bounding_box(Vec2& min, Vec2& max) const {
    Map<GeometryInfo> cache = {};
    bounding_box(min, max, cache);
    for (MapItem<GeometryInfo>* item = cache.next(NULL); item; item = cache.next(item)) {
        item->value.clear();
    }
    cache.clear();
}

void Reference::bounding_box(Vec2& min, Vec2& max, Map<GeometryInfo>& cache) const {
    min.x = min.y = DBL_MAX;
    max.x = max.y = -DBL_MAX;
    if (type != ReferenceType::Cell) return;

    GeometryInfo info = cache.get(cell->name);
    int64_t m;
    Array<Vec2> point_array = {};

    if (is_multiple_of_pi_over_2(rotation, m)) {
        Vec2 cmin, cmax;
        if (!info.bounding_box_valid) {
            info = cell->bounding_box(cache);
        }
        cmin = info.bounding_box_min;
        cmax = info.bounding_box_max;
        if (cmin.x <= cmax.x) {
            point_array.ensure_slots(4);
            point_array.append_unsafe(cmin);
            point_array.append_unsafe(cmax);
            point_array.append_unsafe(Vec2{cmin.x, cmax.y});
            point_array.append_unsafe(Vec2{cmax.x, cmin.y});
        }
    } else {
        if (!info.convex_hull_valid) {
            info = cell->convex_hull(cache);
        }
        point_array.extend(info.convex_hull);
    }

    repeat_and_transform(point_array);

    Vec2* point = point_array.items;
    for (uint64_t i = point_array.count; i > 0; i--, point++) {
        if (point->x < min.x) min.x = point->x;
        if (point->y < min.y) min.y = point->y;
        if (point->x > max.x) max.x = point->x;
        if (point->y > max.y) max.y = point->y;
    }
    point_array.clear();
}

void Reference::convex_hull(Array<Vec2>& result) const {
    if (type != ReferenceType::Cell) return;
    Map<GeometryInfo> cache = {};
    convex_hull(result, cache);
    for (MapItem<GeometryInfo>* item = cache.next(NULL); item; item = cache.next(item)) {
        item->value.clear();
    }
    cache.clear();
}

void Reference::convex_hull(Array<Vec2>& result, Map<GeometryInfo>& cache) const {
    if (type != ReferenceType::Cell) return;
    GeometryInfo info = cache.get(cell->name);
    if (!info.convex_hull_valid) {
        info = cell->convex_hull(cache);
    }
    Array<Vec2> point_array = {};
    point_array.extend(info.convex_hull);
    repeat_and_transform(point_array);
    gdstk::convex_hull(point_array, result);
    point_array.clear();
}

void Reference::transform(double mag, bool x_refl, double rot, const Vec2 orig) {
    const int r1 = x_refl ? -1 : 1;
    const double crot = cos(rot);
    const double srot = sin(rot);
    const double x = origin.x;
    const double y = origin.y;
    origin.x = orig.x + mag * (x * crot - r1 * y * srot);
    origin.y = orig.y + mag * (x * srot + r1 * y * crot);
    rotation = r1 * rotation + rot;
    magnification *= mag;
    x_reflection ^= x_refl;
}

void Reference::apply_repetition(Array<Reference*>& result) {
    if (repetition.type == RepetitionType::None) return;

    Array<Vec2> offsets = {};
    repetition.get_offsets(offsets);
    repetition.clear();

    // Skip first offset (0, 0)
    double* offset_p = (double*)(offsets.items + 1);
    result.ensure_slots(offsets.count - 1);
    for (uint64_t offset_count = offsets.count - 1; offset_count > 0; offset_count--) {
        Reference* reference = (Reference*)allocate_clear(sizeof(Reference));
        reference->copy_from(*this);
        reference->origin.x += *offset_p++;
        reference->origin.y += *offset_p++;
        result.append_unsafe(reference);
    }

    offsets.clear();
    return;
}

// Depth is passed as-is to Cell::get_polygons, where it is inspected and applied.
void Reference::get_polygons(bool apply_repetitions, bool include_paths, int64_t depth, bool filter,
                             Tag tag, Array<Polygon*>& result) const {
    if (type != ReferenceType::Cell) return;

    Array<Polygon*> array = {};
    cell->get_polygons(apply_repetitions, include_paths, depth, filter, tag, array);

    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {};
    if (repetition.type != RepetitionType::None) {
        repetition.get_offsets(offsets);
    } else {
        offsets.count = 1;
        offsets.items = &zero;
    }
    result.ensure_slots(array.count * offsets.count);

    Polygon** a_item = array.items;
    for (uint64_t i = 0; i < array.count; i++) {
        Polygon* src = *a_item++;
        Vec2* offset_p = offsets.items;
        for (uint64_t offset_count = offsets.count; offset_count > 0; offset_count--) {
            Polygon* dst;
            // Avoid an extra allocation by moving the last polygon.
            if (offset_count == 1) {
                dst = src;
            } else {
                dst = (Polygon*)allocate_clear(sizeof(Polygon));
                dst->copy_from(*src);
            }
            dst->transform(magnification, x_reflection, rotation, origin + *offset_p++);
            result.append_unsafe(dst);
        }
    }
    array.clear();
    if (repetition.type != RepetitionType::None) offsets.clear();
}

void Reference::get_flexpaths(bool apply_repetitions, int64_t depth, bool filter, Tag tag,
                              Array<FlexPath*>& result) const {
    if (type != ReferenceType::Cell) return;

    Array<FlexPath*> array = {};
    cell->get_flexpaths(apply_repetitions, depth, filter, tag, array);

    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {};
    if (repetition.type != RepetitionType::None) {
        repetition.get_offsets(offsets);
    } else {
        offsets.count = 1;
        offsets.items = &zero;
    }
    result.ensure_slots(array.count * offsets.count);

    FlexPath** a_item = array.items;
    for (uint64_t i = 0; i < array.count; i++) {
        FlexPath* src = *a_item++;
        Vec2* offset_p = offsets.items;
        for (uint64_t offset_count = offsets.count; offset_count > 0; offset_count--) {
            FlexPath* dst;
            if (offset_count == 1) {
                dst = src;
            } else {
                dst = (FlexPath*)allocate_clear(sizeof(FlexPath));
                dst->copy_from(*src);
            }
            dst->transform(magnification, x_reflection, rotation, origin + *offset_p++);
            result.append_unsafe(dst);
        }
    }
    array.clear();
    if (repetition.type != RepetitionType::None) offsets.clear();
}

void Reference::get_robustpaths(bool apply_repetitions, int64_t depth, bool filter, Tag tag,
                                Array<RobustPath*>& result) const {
    if (type != ReferenceType::Cell) return;

    Array<RobustPath*> array = {};
    cell->get_robustpaths(apply_repetitions, depth, filter, tag, array);

    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {};
    if (repetition.type != RepetitionType::None) {
        repetition.get_offsets(offsets);
    } else {
        offsets.count = 1;
        offsets.items = &zero;
    }
    result.ensure_slots(array.count * offsets.count);

    RobustPath** a_item = array.items;
    for (uint64_t i = 0; i < array.count; i++) {
        RobustPath* src = *a_item++;
        Vec2* offset_p = offsets.items;
        for (uint64_t offset_count = offsets.count; offset_count > 0; offset_count--) {
            RobustPath* dst;
            if (offset_count == 1) {
                dst = src;
            } else {
                dst = (RobustPath*)allocate_clear(sizeof(RobustPath));
                dst->copy_from(*src);
            }
            dst->transform(magnification, x_reflection, rotation, origin + *offset_p++);
            result.append_unsafe(dst);
        }
    }
    array.clear();
    if (repetition.type != RepetitionType::None) offsets.clear();
}

void Reference::get_labels(bool apply_repetitions, int64_t depth, bool filter, Tag tag,
                           Array<Label*>& result) const {
    if (type != ReferenceType::Cell) return;

    Array<Label*> array = {};
    cell->get_labels(apply_repetitions, depth, filter, tag, array);

    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {};
    if (repetition.type != RepetitionType::None) {
        repetition.get_offsets(offsets);
    } else {
        offsets.count = 1;
        offsets.items = &zero;
    }
    result.ensure_slots(array.count * offsets.count);

    Label** a_item = array.items;
    for (uint64_t i = 0; i < array.count; i++) {
        Label* src = *a_item++;
        Vec2* offset_p = offsets.items;
        for (uint64_t offset_count = offsets.count; offset_count > 0; offset_count--) {
            Label* dst;
            if (offset_count == 1) {
                dst = src;
            } else {
                dst = (Label*)allocate_clear(sizeof(Label));
                dst->copy_from(*src);
            }
            dst->transform(magnification, x_reflection, rotation, origin + *offset_p++);
            result.append_unsafe(dst);
        }
    }
    array.clear();
    if (repetition.type != RepetitionType::None) offsets.clear();
}

#define GDSTK_REFERENCE_REPETITION_TOLERANCE 1e-12
ErrorCode Reference::to_gds(FILE* out, double scaling) const {
    ErrorCode error_code = ErrorCode::NoError;
    bool array = false;
    double x2, y2, x3, y3;
    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {};
    offsets.count = 1;
    offsets.items = &zero;

    uint16_t buffer_array[] = {8, 0x1302, 0, 0, 28, 0x1003};
    int32_t buffer_coord[6];
    uint16_t buffer_single[] = {12, 0x1003};
    big_endian_swap16(buffer_single, COUNT(buffer_single));

    uint64_t columns = repetition.columns;
    uint64_t rows = repetition.rows;
    if (repetition.type != RepetitionType::None) {
        int64_t m = 0;
        if (repetition.type == RepetitionType::Regular ||
            (repetition.type == RepetitionType::Rectangular &&
             is_multiple_of_pi_over_2(rotation, m))) {
            Vec2 v1, v2;
            if (repetition.type == RepetitionType::Rectangular) {
                v1.x = repetition.spacing.x;
                v1.y = 0;
                v2.x = 0;
                v2.y = repetition.spacing.y;
            } else {
                v1 = repetition.v1;
                v2 = repetition.v2;
            }
            Vec2 u1 = v1;
            Vec2 u2 = v2;
            double len1 = u1.normalize();
            double len2 = u2.normalize();
            double sa = sin(rotation);
            double ca = cos(rotation);
            double p1 = u1.inner(Vec2{ca, sa});
            double p2 = u2.inner(Vec2{-sa, ca});
            double p3 = u1.inner(Vec2{-sa, ca});
            double p4 = u2.inner(Vec2{ca, sa});
            if ((len1 == 0 || fabs(fabs(p1) - 1.0) < GDSTK_REFERENCE_REPETITION_TOLERANCE) &&
                (len2 == 0 || fabs(fabs(p2) - 1.0) < GDSTK_REFERENCE_REPETITION_TOLERANCE)) {
                array = true;
                x2 = origin.x + columns * v1.x;
                y2 = origin.y + columns * v1.y;
                x3 = origin.x + rows * v2.x;
                y3 = origin.y + rows * v2.y;
            } else if ((len1 == 0 || fabs(fabs(p3) - 1.0) < GDSTK_REFERENCE_REPETITION_TOLERANCE) &&
                       (len2 == 0 || fabs(fabs(p4) - 1.0) < GDSTK_REFERENCE_REPETITION_TOLERANCE)) {
                array = true;
                columns = repetition.rows;
                rows = repetition.columns;
                x2 = origin.x + columns * v2.x;
                y2 = origin.y + columns * v2.y;
                x3 = origin.x + rows * v1.x;
                y3 = origin.y + rows * v1.y;
            }
        }

        if (array) {
            if (repetition.columns > UINT16_MAX || repetition.rows > UINT16_MAX) {
                if (error_logger)
                    fputs(
                        "[GDSTK] Repetition with more than 65535 columns or rows cannot be saved to a GDSII file.\n",
                        error_logger);
                error_code = ErrorCode::InvalidRepetition;
                buffer_array[2] = UINT16_MAX;
                buffer_array[3] = UINT16_MAX;
            } else {
                buffer_array[2] = (uint16_t)columns;
                buffer_array[3] = (uint16_t)rows;
            }
            big_endian_swap16(buffer_array, COUNT(buffer_array));
            buffer_coord[0] = (int32_t)(lround(origin.x * scaling));
            buffer_coord[1] = (int32_t)(lround(origin.y * scaling));
            buffer_coord[2] = (int32_t)(lround(x2 * scaling));
            buffer_coord[3] = (int32_t)(lround(y2 * scaling));
            buffer_coord[4] = (int32_t)(lround(x3 * scaling));
            buffer_coord[5] = (int32_t)(lround(y3 * scaling));
            big_endian_swap32((uint32_t*)buffer_coord, COUNT(buffer_coord));
        } else {
            offsets.count = 0;
            offsets.items = NULL;
            repetition.get_offsets(offsets);
        }
    }

    const char* ref_name = type == ReferenceType::Cell
                               ? cell->name
                               : (type == ReferenceType::RawCell ? rawcell->name : name);
    uint64_t len = strlen(ref_name);
    if (len % 2) len++;
    uint16_t buffer_start[] = {4, 0x0A00, (uint16_t)(4 + len), 0x1206};
    if (array) buffer_start[1] = 0x0B00;
    big_endian_swap16(buffer_start, COUNT(buffer_start));

    uint16_t buffer_end[] = {4, 0x1100};
    big_endian_swap16(buffer_end, COUNT(buffer_end));

    bool transform_ = rotation != 0 || magnification != 1 || x_reflection;
    uint16_t buffer_flags[] = {6, 0x1A01, 0};
    uint16_t buffer_mag[] = {12, 0x1B05};
    uint16_t buffer_rot[] = {12, 0x1C05};
    uint64_t mag_real, rot_real;
    if (transform_) {
        if (x_reflection) {
            buffer_flags[2] |= 0x8000;
        }
        if (magnification != 1) {
            // if("absolute magnification") buffer_flags[2] |= 0x0004; UNSUPPORTED
            big_endian_swap16(buffer_mag, COUNT(buffer_mag));
            mag_real = gdsii_real_from_double(magnification);
            big_endian_swap64(&mag_real, 1);
        }
        if (rotation != 0) {
            // if("absolute rotation") buffer_flags[2] |= 0x0002; UNSUPPORTED
            big_endian_swap16(buffer_rot, COUNT(buffer_rot));
            rot_real = gdsii_real_from_double(rotation * (180.0 / M_PI));
            big_endian_swap64(&rot_real, 1);
        }
        big_endian_swap16(buffer_flags, COUNT(buffer_flags));
    }

    Vec2* offset_p = offsets.items;
    for (uint64_t offset_count = offsets.count; offset_count > 0; offset_count--, offset_p++) {
        fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);
        fwrite(ref_name, 1, len, out);

        if (transform_) {
            fwrite(buffer_flags, sizeof(uint16_t), COUNT(buffer_flags), out);
            if (magnification != 1) {
                fwrite(buffer_mag, sizeof(uint16_t), COUNT(buffer_mag), out);
                fwrite(&mag_real, sizeof(uint64_t), 1, out);
            }
            if (rotation != 0) {
                fwrite(buffer_rot, sizeof(uint16_t), COUNT(buffer_rot), out);
                fwrite(&rot_real, sizeof(uint64_t), 1, out);
            }
        }

        if (array) {
            fwrite(buffer_array, sizeof(uint16_t), COUNT(buffer_array), out);
            fwrite(buffer_coord, sizeof(int32_t), COUNT(buffer_coord), out);
        } else {
            fwrite(buffer_single, sizeof(uint16_t), COUNT(buffer_single), out);
            int32_t buffer_single_coord[] = {(int32_t)(lround((origin.x + offset_p->x) * scaling)),
                                             (int32_t)(lround((origin.y + offset_p->y) * scaling))};
            big_endian_swap32((uint32_t*)buffer_single_coord, COUNT(buffer_single_coord));
            fwrite(buffer_single_coord, sizeof(int32_t), COUNT(buffer_single_coord), out);
        }

        ErrorCode err = properties_to_gds(properties, out);
        if (err != ErrorCode::NoError) error_code = err;
        fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
    }

    if (repetition.type != RepetitionType::None && !array) offsets.clear();
    return error_code;
}

ErrorCode Reference::to_svg(FILE* out, double scaling, uint32_t precision) const {
    const char* src_name = type == ReferenceType::Cell
                               ? cell->name
                               : (type == ReferenceType::RawCell ? rawcell->name : name);
    char* ref_name = (char*)allocate(strlen(src_name) + 1);
    // NOTE: Here be dragons if name is not ASCII.  The GDSII specification imposes ASCII-only
    // for strings, but who knows…
    char* d = ref_name;
    for (const char* c = src_name; *c != 0; c++, d++) *d = *c == '#' ? '_' : *c;
    *d = 0;

    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {};
    if (repetition.type != RepetitionType::None) {
        repetition.get_offsets(offsets);
    } else {
        offsets.count = 1;
        offsets.items = &zero;
    }

    char double_buffer[GDSTK_DOUBLE_BUFFER_COUNT];
    double* offset_p = (double*)offsets.items;
    for (uint64_t offset_count = offsets.count; offset_count > 0; offset_count--) {
        double offset_x = scaling * (origin.x + *offset_p++);
        double offset_y = scaling * (origin.y + *offset_p++);
        fputs("<use transform=\"translate(", out);
        fputs(double_print(offset_x, precision, double_buffer, COUNT(double_buffer)), out);
        fputc(' ', out);
        fputs(double_print(offset_y, precision, double_buffer, COUNT(double_buffer)), out);
        fputc(')', out);
        if (rotation != 0) {
            fputs(" rotate(", out);
            fputs(double_print(rotation * (180.0 / M_PI), precision, double_buffer,
                               COUNT(double_buffer)),
                  out);
            fputc(')', out);
        }
        if (x_reflection) {
            fputs(" scale(1 -1)", out);
        }
        if (magnification != 1) {
            fputs(" scale(", out);
            fputs(double_print(magnification, precision, double_buffer, COUNT(double_buffer)), out);
            fputc(')', out);
        }
        fprintf(out, "\" xlink:href=\"#%s\"/>\n", ref_name);
    }
    free_allocation(ref_name);
    if (repetition.type != RepetitionType::None) offsets.clear();
    return ErrorCode::NoError;
}

}  // namespace gdstk
