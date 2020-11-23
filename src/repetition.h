/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __REPETITION_H__
#define __REPETITION_H__

#include <cstdint>
#include <cstdio>

#include "array.h"
#include "vec.h"

namespace gdstk {

enum struct RepetitionType {
    None = 0,
    Rectangular,
    Regular,
    Explicit,
    ExplicitX,
    ExplicitY,
};

struct Repetition {
    RepetitionType type;
    union {
        struct {              // Rectangular (Oasis 1, 2, 3) or Regular (Oasis 8, 9)
            int64_t columns;  // Along x or v1
            int64_t rows;     // Along y or v2
            union {
                Vec2 spacing;
                Vec2 v1;
            };
            Vec2 v2;
        };
        Array<Vec2> offsets;   // Explicit (10, 11)
        Array<double> coords;  // ExplicitX, ExplicitY (Oasis 4, 5, 6, 7)
    };

    void print() const {
        const int64_t n = 12;
        switch (type) {
            case RepetitionType::Rectangular:
                printf("Rectangular repetition <%p>, %" PRId64 " columns, %" PRId64
                       " rows, spacing (%lg, %lg)\n",
                       this, columns, rows, spacing.x, spacing.y);
                break;
            case RepetitionType::Regular:
                printf("Regular repetition <%p>, %" PRId64 " x %" PRId64
                       " elements along (%lg, %lg) and (%lg, %lg)\n",
                       this, columns, rows, v1.x, v1.y, v2.x, v2.y);
                break;
            case RepetitionType::Explicit:
                printf("Explicit repetition <%p>: ", this);
                offsets.print(true);
                break;
            case RepetitionType::ExplicitX:
            case RepetitionType::ExplicitY:
                printf("Explicit %c repetition <%p>:",
                       type == RepetitionType::ExplicitX ? 'X' : 'Y', this);
                for (int64_t i = 0; i < coords.size; i += n) {
                    for (int64_t j = 0; j < n && i + j < coords.size; j++) {
                        printf(" %lg", coords[i + j]);
                    }
                    putchar('\n');
                }
                break;
            case RepetitionType::None:
                return;
        }
    }

    void clear() {
        if (type == RepetitionType::Explicit) {
            offsets.clear();
        } else if (type == RepetitionType::ExplicitX || type == RepetitionType::ExplicitY) {
            coords.clear();
        }
        memset(this, 0, sizeof(Repetition));
    }

    void copy_from(const Repetition repetition) {
        type = repetition.type;
        switch (type) {
            case RepetitionType::Rectangular:
                columns = repetition.columns;
                rows = repetition.rows;
                spacing = repetition.spacing;
                break;
            case RepetitionType::Regular:
                columns = repetition.columns;
                rows = repetition.rows;
                v1 = repetition.v1;
                v2 = repetition.v2;
                break;
            case RepetitionType::Explicit:
                offsets.copy_from(repetition.offsets);
                break;
            case RepetitionType::ExplicitX:
            case RepetitionType::ExplicitY:
                coords.copy_from(repetition.coords);
                break;
            case RepetitionType::None:
                return;
        }
    }

    int64_t get_size() const {
        switch (type) {
            case RepetitionType::Rectangular:
            case RepetitionType::Regular:
                return columns * rows;
            case RepetitionType::Explicit:
                return offsets.size + 1;  // Assume (0, 0) is not included.
            case RepetitionType::ExplicitX:
            case RepetitionType::ExplicitY:
                return coords.size + 1;  // Assume 0 is not included.
            case RepetitionType::None:
                return 0;
        }
        return 0;
    }

    // NOTE: the coordinates for the original (0, 0) are includded as 1st element
    void get_offsets(Array<Vec2>& result) const {
        if (type == RepetitionType::None) return;
        int64_t size = get_size();
        result.ensure_slots(size);
        double* c_item;
        double* c = (double*)(result.items + result.size);
        switch (type) {
            case RepetitionType::Rectangular:
                for (int64_t i = 0; i < columns; i++) {
                    double cx = i * spacing.x;
                    for (int64_t j = 0; j < rows; j++) {
                        *c++ = cx;
                        *c++ = j * spacing.y;
                    }
                }
                result.size += size;
                break;
            case RepetitionType::Regular:
                for (int64_t i = 0; i < columns; i++) {
                    Vec2 vi = (double)i * v1;
                    for (int64_t j = 0; j < rows; j++) {
                        *c++ = vi.x + j * v2.x;
                        *c++ = vi.y + j * v2.y;
                    }
                }
                result.size += size;
                break;
            case RepetitionType::ExplicitX:
                *c++ = 0;
                *c++ = 0;
                c_item = coords.items;
                for (int64_t j = 1; j < size; j++) {
                    *c++ = *c_item++;
                    *c++ = 0;
                }
                result.size += size;
                break;
            case RepetitionType::ExplicitY:
                *c++ = 0;
                *c++ = 0;
                c_item = coords.items;
                for (int64_t j = 1; j < size; j++) {
                    *c++ = 0;
                    *c++ = *c_item++;
                }
                result.size += size;
                break;
            case RepetitionType::Explicit:
                result.append_unsafe(Vec2{0, 0});
                result.extend(offsets);
                break;
            case RepetitionType::None:
                return;
        }
    }

    void transform(double magnification, bool x_reflection, double rotation) {
        if (type == RepetitionType::None) return;
        switch (type) {
            case RepetitionType::Rectangular: {
                if (magnification != 1) spacing *= magnification;
                if (x_reflection || rotation != 0) {
                    Vec2 v = spacing;
                    if (x_reflection) v.y = -v.y;
                    double ca = cos(rotation);
                    double sa = sin(rotation);
                    type = RepetitionType::Regular;
                    v1.x = v.x * ca;
                    v1.y = v.x * sa;
                    v2.x = -v.y * sa;
                    v2.y = v.y * ca;
                }
            } break;
            case RepetitionType::Regular: {
                if (magnification != 1) {
                    v1 *= magnification;
                    v2 *= magnification;
                }
                if (x_reflection) {
                    v1.y = -v1.y;
                    v2.y = -v2.y;
                }
                if (rotation != 0) {
                    Vec2 r = {cos(rotation), sin(rotation)};
                    v1 = cplx_mul(v1, r);
                    v2 = cplx_mul(v2, r);
                }
            } break;
            case RepetitionType::ExplicitX: {
                if (rotation != 0) {
                    double ca = magnification * cos(rotation);
                    double sa = magnification * sin(rotation);
                    Array<Vec2> temp = {0};
                    temp.ensure_slots(coords.size);
                    temp.size = coords.size;
                    Vec2* v = temp.items;
                    double* c = coords.items;
                    for (uint64_t i = coords.size; i > 0; i--, c++, v++) {
                        v->x = *c * ca;
                        v->y = *c * sa;
                    }
                    coords.clear();
                    type = RepetitionType::Explicit;
                    offsets = temp;
                } else if (magnification != 1) {
                    double* c = coords.items;
                    for (uint64_t i = coords.size; i > 0; i--) {
                        *c++ *= magnification;
                    }
                }
            } break;
            case RepetitionType::ExplicitY: {
                if (rotation != 0) {
                    double ca = magnification * cos(rotation);
                    double sa = -magnification * sin(rotation);
                    if (x_reflection) {
                        ca = -ca;
                        sa = -sa;
                    }
                    Array<Vec2> temp = {0};
                    temp.ensure_slots(coords.size);
                    temp.size = coords.size;
                    Vec2* v = temp.items;
                    double* c = coords.items;
                    for (uint64_t i = coords.size; i > 0; i--, c++, v++) {
                        v->x = *c * sa;
                        v->y = *c * ca;
                    }
                    coords.clear();
                    type = RepetitionType::Explicit;
                    offsets = temp;
                } else if (x_reflection || magnification != 1) {
                    if (x_reflection) magnification = -magnification;
                    double* c = coords.items;
                    for (uint64_t i = coords.size; i > 0; i--) {
                        *c++ *= magnification;
                    }
                }
            } break;
            case RepetitionType::Explicit: {
                Vec2* v = offsets.items;
                if (rotation != 0) {
                    Vec2 r = {magnification * cos(rotation), magnification * sin(rotation)};
                    if (x_reflection) {
                        for (uint64_t i = offsets.size; i > 0; i--, v++) {
                            *v = cplx_mul(cplx_conj(*v), r);
                        }
                    } else {
                        for (uint64_t i = offsets.size; i > 0; i--, v++) {
                            *v = cplx_mul(*v, r);
                        }
                    }
                } else if (x_reflection && magnification != 1) {
                    for (uint64_t i = offsets.size; i > 0; i--, v++) {
                        v->x *= magnification;
                        v->y *= -magnification;
                    }
                } else if (x_reflection) {
                    for (uint64_t i = offsets.size; i > 0; i--, v++) {
                        v->y = -v->y;
                    }
                } else if (magnification != 1) {
                    for (uint64_t i = offsets.size; i > 0; i--, v++) {
                        *v *= magnification;
                    }
                }
            } break;
            default:
                return;
        }
    }
};

}  // namespace gdstk

#endif
