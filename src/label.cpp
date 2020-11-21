/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "label.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "utils.h"

namespace gdstk {

void Label::print() {
    printf(
        "Label <%p> %s, at (%lg, %lg), %lg rad, mag %lg, reflection %d, layer %hd, texttype %hd, properties <%p>, owner <%p>\n",
        this, text, origin.x, origin.y, rotation, magnification, x_reflection, layer, texttype,
        properties, owner);
    repetition.print();
}

void Label::clear() {
    if (text) {
        free_allocation(text);
        text = NULL;
    }
    repetition.clear();
    properties_clear(properties);
    properties = NULL;
}

void Label::copy_from(const Label& label) {
    layer = label.layer;
    texttype = label.texttype;
    text = (char*)allocate((strlen(label.text) + 1) * sizeof(char));
    strcpy(text, label.text);
    origin = label.origin;
    anchor = label.anchor;
    rotation = label.rotation;
    magnification = label.magnification;
    x_reflection = label.x_reflection;
    repetition.copy_from(label.repetition);
    properties = properties_copy(label.properties);
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

    Array<Vec2> offsets = {0};
    repetition.get_offsets(offsets);
    repetition.clear();

    // Skip first offset (0, 0)
    double* offset_p = (double*)(offsets.items + 1);
    result.ensure_slots(offsets.size - 1);
    for (int64_t offset_count = offsets.size - 1; offset_count > 0; offset_count--) {
        Label* label = (Label*)allocate_clear(sizeof(Label));
        label->copy_from(*this);
        label->origin.x += *offset_p++;
        label->origin.y += *offset_p++;
        result.append_unsafe(label);
    }

    offsets.clear();
    return;
}

void Label::to_gds(FILE* out, double scaling) const {
    uint16_t buffer_start[] = {
        4,      0x0C00,          6, 0x0D02, (uint16_t)layer, 6, 0x1602, (uint16_t)texttype, 6,
        0x1701, (uint16_t)anchor};
    swap16(buffer_start, COUNT(buffer_start));

    uint16_t buffer_end[] = {4, 0x1100};
    swap16(buffer_end, COUNT(buffer_end));

    uint16_t buffer_xy[] = {12, 0x1003};
    swap16(buffer_xy, COUNT(buffer_xy));

    int64_t len = strlen(text);
    if (len % 2) len++;
    uint16_t buffer_text[] = {(uint16_t)(4 + len), 0x1906};
    swap16(buffer_text, COUNT(buffer_text));

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
            // if ("absolute magnification") buffer_flags[2] |= 0x0004; // UNSUPPORTED
            swap16(buffer_mag, COUNT(buffer_mag));
            mag_real = gdsii_real_from_double(magnification);
            swap64(&mag_real, 1);
        }
        if (rotation != 0) {
            // if ("absolute rotation") buffer_flags[2] |= 0x0002; // UNSUPPORTED
            swap16(buffer_rot, COUNT(buffer_rot));
            rot_real = gdsii_real_from_double(rotation * (180.0 / M_PI));
            swap64(&rot_real, 1);
        }
        swap16(buffer_flags, COUNT(buffer_flags));
    }

    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {0};
    if (repetition.type != RepetitionType::None) {
        repetition.get_offsets(offsets);
    } else {
        offsets.size = 1;
        offsets.items = &zero;
    }

    Vec2* offset_p = offsets.items;
    for (int64_t offset_count = offsets.size; offset_count > 0; offset_count--, offset_p++) {
        fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);

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

        int32_t buffer_pos[] = {(int32_t)(lround((origin.x + offset_p->x) * scaling)),
                                (int32_t)(lround((origin.y + offset_p->y) * scaling))};
        swap32((uint32_t*)buffer_pos, COUNT(buffer_pos));

        fwrite(buffer_xy, sizeof(uint16_t), COUNT(buffer_xy), out);
        fwrite(buffer_pos, sizeof(uint32_t), COUNT(buffer_pos), out);

        fwrite(buffer_text, sizeof(uint16_t), COUNT(buffer_text), out);
        fwrite(text, sizeof(char), len, out);

        properties_to_gds(properties, out);
        fwrite(buffer_end, sizeof(int16_t), COUNT(buffer_end), out);
    }
    if (repetition.type != RepetitionType::None) offsets.clear();
}

void Label::to_svg(FILE* out, double scaling) const {
    fprintf(out, "<text id=\"%p\" class=\"l%dt%d\"", this, layer, texttype);
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

    fprintf(out, " transform=\"scale(1 -1) translate(%lf %lf)", scaling * origin.x,
            -scaling * origin.y);
    // Negative sign to correct for the default coordinate system with y-down
    if (rotation != 0) fprintf(out, " rotate(%lf)", rotation * (-180.0 / M_PI));
    if (x_reflection) fputs(" scale(1 -1)", out);
    if (magnification != 1) fprintf(out, " scale(%lf)", magnification);

    // NOTE: Escape “<”, “>”, and “&” inside the SVG tag.  Here be dragons if the text is not ASCII.
    // The GDSII specification imposes ASCII-only for strings, but who knows…
    fputs("\">", out);
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
        Array<Vec2> offsets = {0};
        repetition.get_offsets(offsets);
        double* offset_p = (double*)(offsets.items + 1);
        for (int64_t offset_count = offsets.size - 1; offset_count > 0; offset_count--) {
            double offset_x = *offset_p++;
            double offset_y = *offset_p++;
            fprintf(out, "<use href=\"#%p\" x=\"%lf\" y=\"%lf\"/>\n", this, offset_x * scaling,
                    offset_y * scaling);
        }
        offsets.clear();
    }
}

}  // namespace gdstk
