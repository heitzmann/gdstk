/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "reference.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "allocator.h"
#include "cell.h"
#include "gdsii.h"
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
    printf(", at (%lg, %lg), %lg rad, mag %lg, reflection %d, properties <%p>, owner <%p>\n",
           origin.x, origin.y, rotation, magnification, x_reflection, properties, owner);
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
        uint64_t len;
        name = copy_string(reference.name, len);
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

void Reference::bounding_box(Vec2& min, Vec2& max) const {
    int64_t m;
    min.x = min.y = DBL_MAX;
    max.x = max.y = -DBL_MAX;
    Array<Polygon*> array = {0};
    if (type == ReferenceType::Cell && is_multiple_of_pi_over_2(rotation, m)) {
        Vec2 cmin, cmax;
        cell->bounding_box(cmin, cmax);
        if (cmin.x <= cmax.x) {
            Polygon* src = (Polygon*)allocate_clear(sizeof(Polygon));
            *src = rectangle(cmin, cmax, 0, 0);

            Vec2 zero = {0, 0};
            Array<Vec2> offsets = {0};
            if (repetition.type != RepetitionType::None) {
                repetition.get_extrema(offsets);
            } else {
                offsets.count = 1;
                offsets.items = &zero;
            }
            array.ensure_slots(offsets.count);

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
                array.append_unsafe(dst);
            }
            if (repetition.type != RepetitionType::None) offsets.clear();
        }
    } else {
        polygons(true, true, -1, array);

        Array<Label*> lbl_array = {0};
        labels(true, -1, lbl_array);

        if (lbl_array.count > 0) {
            Polygon* lbl_poly = (Polygon*)allocate_clear(sizeof(Polygon));
            array.append(lbl_poly);

            Label** p_lbl = lbl_array.items;
            for (uint64_t i = 0; i < lbl_array.count; i++) {
                Label* lbl = *p_lbl++;
                lbl_poly->point_array.append(lbl->origin);
                lbl->clear();
                free_allocation(lbl);
            }
        }

        lbl_array.clear();
    }

    Polygon** p_item = array.items;
    for (uint64_t i = 0; i < array.count; i++) {
        Polygon* poly = *p_item++;
        Vec2 pmin, pmax;
        poly->bounding_box(pmin, pmax);
        if (pmin.x < min.x) min.x = pmin.x;
        if (pmin.y < min.y) min.y = pmin.y;
        if (pmax.x > max.x) max.x = pmax.x;
        if (pmax.y > max.y) max.y = pmax.y;
        poly->clear();
        free_allocation(poly);
    }
    array.clear();
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

    Array<Vec2> offsets = {0};
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
void Reference::polygons(bool apply_repetitions, bool include_paths, int64_t depth,
                         Array<Polygon*>& result) const {
    if (type != ReferenceType::Cell) return;

    Array<Polygon*> array = {0};
    cell->get_polygons(apply_repetitions, include_paths, depth, array);

    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {0};
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

void Reference::flexpaths(bool apply_repetitions, int64_t depth, Array<FlexPath*>& result) const {
    if (type != ReferenceType::Cell) return;

    Array<FlexPath*> array = {0};
    cell->get_flexpaths(apply_repetitions, depth, array);

    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {0};
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

void Reference::robustpaths(bool apply_repetitions, int64_t depth,
                            Array<RobustPath*>& result) const {
    if (type != ReferenceType::Cell) return;

    Array<RobustPath*> array = {0};
    cell->get_robustpaths(apply_repetitions, depth, array);

    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {0};
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

void Reference::labels(bool apply_repetitions, int64_t depth, Array<Label*>& result) const {
    if (type != ReferenceType::Cell) return;

    Array<Label*> array = {0};
    cell->get_labels(apply_repetitions, depth, array);

    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {0};
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

#define REFERENCE_REPETITION_TOLERANCE 1e-12
void Reference::to_gds(FILE* out, double scaling) const {
    bool array = false;
    double x2, y2, x3, y3;
    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {0};
    offsets.count = 1;
    offsets.items = &zero;

    uint16_t buffer_array[] = {8, 0x1302, 0, 0, 28, 0x1003};
    int32_t buffer_coord[6];
    uint16_t buffer_single[] = {12, 0x1003};
    big_endian_swap16(buffer_single, COUNT(buffer_single));

    if (repetition.type != RepetitionType::None) {
        if (repetition.type == RepetitionType::Rectangular && !x_reflection && rotation == 0) {
            // printf("AREF (simple): ");
            // print();
            array = true;
            x2 = origin.x + repetition.columns * repetition.spacing.x;
            y2 = origin.y;
            x3 = origin.x;
            y3 = origin.y + repetition.rows * repetition.spacing.y;
        } else if (repetition.type == RepetitionType::Regular) {
            Vec2 u1 = repetition.v1;
            Vec2 u2 = repetition.v2;
            u1.normalize();
            u2.normalize();
            if (x_reflection) u2 = -u2;
            double sa = sin(rotation);
            double ca = cos(rotation);
            if (fabs(u1.x - ca) < REFERENCE_REPETITION_TOLERANCE &&
                fabs(u1.y - sa) < REFERENCE_REPETITION_TOLERANCE &&
                fabs(u2.x + sa) < REFERENCE_REPETITION_TOLERANCE &&
                fabs(u2.y - ca) < REFERENCE_REPETITION_TOLERANCE) {
                // printf("AREF (complex): ");
                // print();
                array = true;
                x2 = origin.x + repetition.columns * repetition.v1.x;
                y2 = origin.y + repetition.columns * repetition.v1.y;
                x3 = origin.x + repetition.rows * repetition.v2.x;
                y3 = origin.y + repetition.rows * repetition.v2.y;
            }
        }

        if (array) {
            if (repetition.columns > UINT16_MAX || repetition.rows > UINT16_MAX) {
                fputs(
                    "[GDSTK] Repetition with more than 65535 columns or rows cannot be saved to a GDSII file.\n",
                    stderr);
            }
            buffer_array[2] = (uint16_t)repetition.columns;
            buffer_array[3] = (uint16_t)repetition.rows;
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
            // printf("Repeated SREF: ");
            // print();
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

    bool transform = rotation != 0 || magnification != 1 || x_reflection;
    uint16_t buffer_flags[] = {6, 0x1A01, 0};
    uint16_t buffer_mag[] = {12, 0x1B05};
    uint16_t buffer_rot[] = {12, 0x1C05};
    uint64_t mag_real, rot_real;
    if (transform) {
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

        if (transform) {
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

        properties_to_gds(properties, out);
        fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
    }

    if (repetition.type != RepetitionType::None && !array) offsets.clear();
}

void Reference::to_svg(FILE* out, double scaling) const {
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
    Array<Vec2> offsets = {0};
    if (repetition.type != RepetitionType::None) {
        repetition.get_offsets(offsets);
    } else {
        offsets.count = 1;
        offsets.items = &zero;
    }

    double* offset_p = (double*)offsets.items;
    for (uint64_t offset_count = offsets.count; offset_count > 0; offset_count--) {
        double offset_x = scaling * (origin.x + *offset_p++);
        double offset_y = scaling * (origin.y + *offset_p++);
        fprintf(out, "<use transform=\"translate(%lf %lf)", offset_x, offset_y);
        if (rotation != 0) fprintf(out, " rotate(%lf)", rotation * (180.0 / M_PI));
        if (x_reflection) fputs(" scale(1 -1)", out);
        if (magnification != 1) fprintf(out, " scale(%lf)", magnification);
        fprintf(out, "\" xlink:href=\"#%s\"/>\n", ref_name);
    }
    free_allocation(ref_name);
    if (repetition.type != RepetitionType::None) offsets.clear();
}

}  // namespace gdstk
