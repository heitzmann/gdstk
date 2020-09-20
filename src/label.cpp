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
}

void Label::to_gds(FILE* out, double scaling) const {
    uint16_t buffer_start[] = {
        4,      0x0C00,          6, 0x0D02, (uint16_t)layer, 6, 0x1602, (uint16_t)texttype, 6,
        0x1701, (uint16_t)anchor};
    swap16(buffer_start, COUNT(buffer_start));
    fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);

    if (rotation != 0 || magnification != 1 || x_reflection) {
        uint16_t buffer_flags[] = {6, 0x1A01, 0};
        if (x_reflection) buffer_flags[2] |= 0x8000;
        // Unsuported flags
        // if (magnification != 1 && "absolute magnification") buffer_flags[2] |= 0x0004;
        // if (rotation != 0 && "absolute rotation") buffer_flags[2] |= 0x0002;
        swap16(buffer_flags, COUNT(buffer_flags));
        fwrite(buffer_flags, sizeof(uint16_t), COUNT(buffer_flags), out);
        if (magnification != 1) {
            uint16_t buffer_mag[] = {12, 0x1B05};
            swap16(buffer_mag, COUNT(buffer_mag));
            fwrite(buffer_mag, sizeof(uint16_t), COUNT(buffer_mag), out);
            uint64_t real_mag = gdsii_real_from_double(magnification);
            swap64(&real_mag, 1);
            fwrite(&real_mag, sizeof(uint64_t), 1, out);
        }
        if (rotation != 0) {
            uint16_t buffer_rot[] = {12, 0x1C05};
            swap16(buffer_rot, COUNT(buffer_rot));
            fwrite(buffer_rot, sizeof(uint16_t), COUNT(buffer_rot), out);
            uint64_t real_rot = gdsii_real_from_double(rotation * (180.0 / M_PI));
            swap64(&real_rot, 1);
            fwrite(&real_rot, sizeof(uint64_t), 1, out);
        }
    }

    uint16_t buffer[] = {12, 0x1003};
    int32_t buffer_pos[] = {(int32_t)(lround(origin.x * scaling)),
                            (int32_t)(lround(origin.y * scaling))};
    swap16(buffer, COUNT(buffer));
    fwrite(buffer, sizeof(uint16_t), COUNT(buffer), out);
    swap32((uint32_t*)buffer_pos, COUNT(buffer_pos));
    fwrite(buffer_pos, sizeof(uint32_t), COUNT(buffer_pos), out);

    int64_t len = strlen(text);
    if (len % 2) len++;
    buffer[0] = (uint16_t)(4 + len);
    buffer[1] = 0x1906;
    swap16(buffer, COUNT(buffer));
    fwrite(buffer, sizeof(uint16_t), COUNT(buffer), out);
    fwrite(text, sizeof(char), len, out);

    properties_to_gds(properties, out);

    buffer[0] = 4;
    buffer[1] = 0x1100;
    swap16(buffer, COUNT(buffer));
    fwrite(buffer, sizeof(int16_t), COUNT(buffer), out);
}

void Label::to_svg(FILE* out, double scaling) const {
    fprintf(out, "<text class=\"l%dt%d\"", layer, texttype);
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
}

}  // namespace gdstk
