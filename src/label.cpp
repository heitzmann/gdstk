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

#include <gdstk/gdsii.hpp>
#include <gdstk/label.hpp>
#include <gdstk/utils.hpp>

namespace gdstk {

void Label::print() {
    printf("Label <%p> %s, at (%lg, %lg), %lg rad, mag %lg,%s reflected, layer %" PRIu32
           ", texttype %" PRIu32 ", properties <%p>, owner <%p>\n",
           this, text, origin.x, origin.y, rotation, magnification, x_reflection ? "" : " not",
           get_layer(tag), get_type(tag), properties, owner);
    properties_print(properties);
    repetition.print();
}

void Label::clear() {
    if (text) {
        free_allocation(text);
        text = NULL;
    }
    repetition.clear();
    properties_clear(properties);
}

void Label::copy_from(const Label& label) {
    tag = label.tag;
    text = copy_string(label.text, NULL);
    origin = label.origin;
    anchor = label.anchor;
    rotation = label.rotation;
    magnification = label.magnification;
    x_reflection = label.x_reflection;
    repetition.copy_from(label.repetition);
    properties = properties_copy(label.properties);
}

void Label::bounding_box(Vec2& min, Vec2& max) const {
    min = origin;
    max = origin;
    if (repetition.type != RepetitionType::None) {
        Array<Vec2> offsets = {};
        repetition.get_extrema(offsets);
        Vec2* off = offsets.items;
        Vec2 min0 = min;
        Vec2 max0 = max;
        for (uint64_t i = offsets.count; i > 0; i--, off++) {
            if (min0.x + off->x < min.x) min.x = min0.x + off->x;
            if (max0.x + off->x > max.x) max.x = max0.x + off->x;
            if (min0.y + off->y < min.y) min.y = min0.y + off->y;
            if (max0.y + off->y > max.y) max.y = max0.y + off->y;
        }
        offsets.clear();
    }
}

void Label::transform(double mag, bool x_refl, double rot, const Vec2 orig) {
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

void Label::apply_repetition(Array<Label*>& result) {
    if (repetition.type == RepetitionType::None) return;

    Array<Vec2> offsets = {};
    repetition.get_offsets(offsets);
    repetition.clear();

    // Skip first offset (0, 0)
    double* offset_p = (double*)(offsets.items + 1);
    result.ensure_slots(offsets.count - 1);
    for (uint64_t offset_count = offsets.count - 1; offset_count > 0; offset_count--) {
        Label* label = (Label*)allocate_clear(sizeof(Label));
        label->copy_from(*this);
        label->origin.x += *offset_p++;
        label->origin.y += *offset_p++;
        result.append_unsafe(label);
    }

    offsets.clear();
    return;
}

ErrorCode Label::to_gds(FILE* out, double scaling) const {
    ErrorCode error_code = ErrorCode::NoError;
    uint16_t buffer0[] = {4, 0x0C00};
    uint16_t buffer1[] = {6, 0x1701, (uint16_t)anchor};
    big_endian_swap16(buffer0, COUNT(buffer0));
    big_endian_swap16(buffer1, COUNT(buffer1));

    uint16_t buffer_end[] = {4, 0x1100};
    big_endian_swap16(buffer_end, COUNT(buffer_end));

    uint16_t buffer_xy[] = {12, 0x1003};
    big_endian_swap16(buffer_xy, COUNT(buffer_xy));

    uint64_t len = strlen(text);
    if (len % 2) len++;
    uint16_t buffer_text[] = {(uint16_t)(4 + len), 0x1906};
    big_endian_swap16(buffer_text, COUNT(buffer_text));

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
            // if ("absolute magnification") buffer_flags[2] |= 0x0004; // UNSUPPORTED
            big_endian_swap16(buffer_mag, COUNT(buffer_mag));
            mag_real = gdsii_real_from_double(magnification);
            big_endian_swap64(&mag_real, 1);
        }
        if (rotation != 0) {
            // if ("absolute rotation") buffer_flags[2] |= 0x0002; // UNSUPPORTED
            big_endian_swap16(buffer_rot, COUNT(buffer_rot));
            rot_real = gdsii_real_from_double(rotation * (180.0 / M_PI));
            big_endian_swap64(&rot_real, 1);
        }
        big_endian_swap16(buffer_flags, COUNT(buffer_flags));
    }

    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {};
    if (repetition.type != RepetitionType::None) {
        repetition.get_offsets(offsets);
    } else {
        offsets.count = 1;
        offsets.items = &zero;
    }

    Vec2* offset_p = offsets.items;
    for (uint64_t offset_count = offsets.count; offset_count > 0; offset_count--, offset_p++) {
        fwrite(buffer0, sizeof(uint16_t), COUNT(buffer0), out);
        tag_to_gds(out, tag, GdsiiRecord::TEXTTYPE);
        fwrite(buffer1, sizeof(uint16_t), COUNT(buffer1), out);

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

        int32_t buffer_pos[] = {(int32_t)(lround((origin.x + offset_p->x) * scaling)),
                                (int32_t)(lround((origin.y + offset_p->y) * scaling))};
        big_endian_swap32((uint32_t*)buffer_pos, COUNT(buffer_pos));

        fwrite(buffer_xy, sizeof(uint16_t), COUNT(buffer_xy), out);
        fwrite(buffer_pos, sizeof(uint32_t), COUNT(buffer_pos), out);

        fwrite(buffer_text, sizeof(uint16_t), COUNT(buffer_text), out);
        fwrite(text, 1, len, out);

        ErrorCode err = properties_to_gds(properties, out);
        if (err != ErrorCode::NoError) error_code = err;
        fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
    }
    if (repetition.type != RepetitionType::None) offsets.clear();
    return error_code;
}

ErrorCode Label::to_svg(FILE* out, double scaling, uint32_t precision) const {
    fprintf(out, "<text id=\"%p\" class=\"l%" PRIu32 "t%" PRIu32 "\"", this, get_layer(tag),
            get_type(tag));
    switch (anchor) {
        case Anchor::NW:
        case Anchor::W:
        case Anchor::SW:
            fputs(" text-anchor=\"start\"", out);
            break;
        case Anchor::N:
        case Anchor::O:
        case Anchor::S:
            fputs(" text-anchor=\"middle\"", out);
            break;
        case Anchor::NE:
        case Anchor::E:
        case Anchor::SE:
            fputs(" text-anchor=\"end\"", out);
            break;
    }
    switch (anchor) {
        case Anchor::NW:
        case Anchor::N:
        case Anchor::NE:
            fputs(" dominant-baseline=\"text-before-edge\"", out);
            break;
        case Anchor::W:
        case Anchor::O:
        case Anchor::E:
            fputs(" dominant-baseline=\"central\"", out);
            break;
        case Anchor::SW:
        case Anchor::S:
        case Anchor::SE:
            fputs(" dominant-baseline=\"text-after-edge\"", out);
            break;
    }

    char double_buffer[GDSTK_DOUBLE_BUFFER_COUNT];
    fputs(" transform=\"translate(", out);
    fputs(double_print(scaling * origin.x, precision, double_buffer, COUNT(double_buffer)), out);
    fputc(' ', out);
    fputs(double_print(scaling * origin.y, precision, double_buffer, COUNT(double_buffer)), out);
    fputc(')', out);

    if (rotation != 0) {
        fputs(" rotate(", out);
        fputs(
            double_print(rotation * (180.0 / M_PI), precision, double_buffer, COUNT(double_buffer)),
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

    // NOTE: Escape “<”, “>”, and “&” inside the SVG tag.  Here be dragons if the text is not ASCII.
    // The GDSII specification imposes ASCII-only for strings, but who knows…
    fputs(" scale(1 -1)\">", out);
    for (char* c = text; *c != 0; c++) {
        switch (*c) {
            case '<':
                fputs("&lt;", out);
                break;
            case '>':
                fputs("&gt;", out);
                break;
            case '&':
                fputs("&amp;", out);
                break;
            default:
                putc(*c, out);
        }
    }
    fputs("</text>\n", out);

    if (repetition.type != RepetitionType::None) {
        Array<Vec2> offsets = {};
        repetition.get_offsets(offsets);
        double* offset_p = (double*)(offsets.items + 1);
        for (uint64_t offset_count = offsets.count - 1; offset_count > 0; offset_count--) {
            double offset_x = *offset_p++;
            double offset_y = *offset_p++;
            fprintf(out, "<use href=\"#%p\" x=\"", this);
            fputs(double_print(offset_x * scaling, precision, double_buffer, COUNT(double_buffer)),
                  out);
            fputs("\" y=\"", out);
            fputs(double_print(offset_y * scaling, precision, double_buffer, COUNT(double_buffer)),
                  out);
            fputs("\"/>\n", out);
        }
        offsets.clear();
    }
    return ErrorCode::NoError;
}

}  // namespace gdstk
