/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>

#include <gdstk/array.hpp>
#include <gdstk/repetition.hpp>
#include <gdstk/vec.hpp>

namespace gdstk {

void Repetition::print() const {
    switch (type) {
        case RepetitionType::Rectangular:
            printf("Rectangular repetition <%p>, %" PRIu64 " columns, %" PRIu64
                   " rows, spacing (%lg, %lg)\n",
                   this, columns, rows, spacing.x, spacing.y);
            break;
        case RepetitionType::Regular:
            printf("Regular repetition <%p>, %" PRIu64 " x %" PRIu64
                   " elements along (%lg, %lg) and (%lg, %lg)\n",
                   this, columns, rows, v1.x, v1.y, v2.x, v2.y);
            break;
        case RepetitionType::Explicit:
            printf("Explicit repetition <%p>: ", this);
            offsets.print(true);
            break;
        case RepetitionType::ExplicitX:
        case RepetitionType::ExplicitY:
            printf("Explicit %c repetition <%p>: ", type == RepetitionType::ExplicitX ? 'X' : 'Y',
                   this);
            coords.print(true);
            break;
        case RepetitionType::None:
            return;
    }
}

void Repetition::clear() {
    if (type == RepetitionType::Explicit) {
        offsets.clear();
    } else if (type == RepetitionType::ExplicitX || type == RepetitionType::ExplicitY) {
        coords.clear();
    }
    memset(this, 0, sizeof(Repetition));
}

void Repetition::copy_from(const Repetition repetition) {
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

uint64_t Repetition::get_count() const {
    switch (type) {
        case RepetitionType::Rectangular:
        case RepetitionType::Regular:
            return columns * rows;
        case RepetitionType::Explicit:
            return offsets.count + 1;  // Assume (0, 0) is not included.
        case RepetitionType::ExplicitX:
        case RepetitionType::ExplicitY:
            return coords.count + 1;  // Assume 0 is not included.
        case RepetitionType::None:
            return 0;
    }
    return 0;
}

void Repetition::get_offsets(Array<Vec2>& result) const {
    uint64_t count = get_count();
    result.ensure_slots(count);
    double* c_item;
    double* c = (double*)(result.items + result.count);
    switch (type) {
        case RepetitionType::Rectangular:
            for (uint64_t i = 0; i < columns; i++) {
                double cx = i * spacing.x;
                for (uint64_t j = 0; j < rows; j++) {
                    *c++ = cx;
                    *c++ = j * spacing.y;
                }
            }
            result.count += count;
            break;
        case RepetitionType::Regular:
            for (uint64_t i = 0; i < columns; i++) {
                Vec2 vi = (double)i * v1;
                for (uint64_t j = 0; j < rows; j++) {
                    *c++ = vi.x + j * v2.x;
                    *c++ = vi.y + j * v2.y;
                }
            }
            result.count += count;
            break;
        case RepetitionType::ExplicitX:
            *c++ = 0;
            *c++ = 0;
            c_item = coords.items;
            for (uint64_t j = 1; j < count; j++) {
                *c++ = *c_item++;
                *c++ = 0;
            }
            result.count += count;
            break;
        case RepetitionType::ExplicitY:
            *c++ = 0;
            *c++ = 0;
            c_item = coords.items;
            for (uint64_t j = 1; j < count; j++) {
                *c++ = 0;
                *c++ = *c_item++;
            }
            result.count += count;
            break;
        case RepetitionType::Explicit:
            result.append_unsafe(Vec2{0, 0});
            result.extend(offsets);
            break;
        case RepetitionType::None:
            return;
    }
}

void Repetition::get_extrema(Array<Vec2>& result) const {
    switch (type) {
        case RepetitionType::Rectangular:
            if (columns == 0 || rows == 0) return;
            if (columns == 1) {
                if (rows == 1) {
                    result.append(Vec2{0, 0});
                } else {
                    result.ensure_slots(2);
                    result.append_unsafe(Vec2{0, 0});
                    result.append_unsafe(Vec2{0, (rows - 1) * spacing.y});
                }
            } else {
                if (rows == 1) {
                    result.ensure_slots(2);
                    result.append_unsafe(Vec2{0, 0});
                    result.append_unsafe(Vec2{(columns - 1) * spacing.x, 0});
                } else {
                    result.ensure_slots(4);
                    result.append_unsafe(Vec2{0, 0});
                    result.append_unsafe(Vec2{0, (rows - 1) * spacing.y});
                    result.append_unsafe(Vec2{(columns - 1) * spacing.x, 0});
                    result.append_unsafe(Vec2{(columns - 1) * spacing.x, (rows - 1) * spacing.y});
                }
            }
            break;
        case RepetitionType::Regular:
            if (columns == 0 || rows == 0) return;
            if (columns == 1) {
                if (rows == 1) {
                    result.append(Vec2{0, 0});
                } else {
                    result.ensure_slots(2);
                    result.append_unsafe(Vec2{0, 0});
                    result.append_unsafe((double)(rows - 1) * v2);
                }
            } else {
                if (rows == 1) {
                    result.ensure_slots(2);
                    result.append_unsafe(Vec2{0, 0});
                    result.append_unsafe((double)(columns - 1) * v1);
                } else {
                    result.ensure_slots(4);
                    Vec2 vi = (double)(columns - 1) * v1;
                    Vec2 vj = (double)(rows - 1) * v2;
                    result.append_unsafe(Vec2{0, 0});
                    result.append_unsafe(vi);
                    result.append_unsafe(vj);
                    result.append_unsafe(vi + vj);
                }
            }
            break;
        case RepetitionType::ExplicitX: {
            if (coords.count == 0) return;
            double xmin = 0;
            double xmax = 0;
            double* c = coords.items;
            for (uint64_t i = coords.count; i > 0; i--, c++) {
                if (*c < xmin) {
                    xmin = *c;
                } else if (*c > xmax) {
                    xmax = *c;
                }
            }
            if (xmin != xmax) {
                result.ensure_slots(2);
                result.append_unsafe(Vec2{xmin, 0});
                result.append_unsafe(Vec2{xmax, 0});
            } else {
                result.append(Vec2{xmin, 0});
            }
        } break;
        case RepetitionType::ExplicitY: {
            if (coords.count == 0) return;
            double ymin = 0;
            double ymax = 0;
            double* c = coords.items;
            for (uint64_t i = coords.count; i > 0; i--, c++) {
                if (*c < ymin) {
                    ymin = *c;
                } else if (*c > ymax) {
                    ymax = *c;
                }
            }
            if (ymin != ymax) {
                result.ensure_slots(2);
                result.append_unsafe(Vec2{0, ymin});
                result.append_unsafe(Vec2{0, ymax});
            } else {
                result.append(Vec2{0, ymin});
            }
        } break;
        case RepetitionType::Explicit: {
            if (offsets.count == 0) return;
            Vec2 vxmin = {0, 0};
            Vec2 vxmax = {0, 0};
            Vec2 vymin = {0, 0};
            Vec2 vymax = {0, 0};
            Vec2* v = offsets.items;
            for (uint64_t i = offsets.count; i > 0; i--, v++) {
                if (v->x < vxmin.x) {
                    vxmin = *v;
                } else if (v->x > vxmax.x) {
                    vxmax = *v;
                }
                if (v->y < vymin.y) {
                    vymin = *v;
                } else if (v->y > vymax.y) {
                    vymax = *v;
                }
            }
            result.ensure_slots(4);
            result.append_unsafe(vxmin);
            result.append_unsafe(vxmax);
            result.append_unsafe(vymin);
            result.append_unsafe(vymax);
        } break;
        case RepetitionType::None:
            return;
    }
}

void Repetition::transform(double magnification, bool x_reflection, double rotation) {
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
                Array<Vec2> temp = {};
                temp.ensure_slots(coords.count);
                temp.count = coords.count;
                Vec2* v = temp.items;
                double* c = coords.items;
                for (uint64_t i = coords.count; i > 0; i--, c++, v++) {
                    v->x = *c * ca;
                    v->y = *c * sa;
                }
                coords.clear();
                type = RepetitionType::Explicit;
                offsets = temp;
            } else if (magnification != 1) {
                double* c = coords.items;
                for (uint64_t i = coords.count; i > 0; i--) {
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
                Array<Vec2> temp = {};
                temp.ensure_slots(coords.count);
                temp.count = coords.count;
                Vec2* v = temp.items;
                double* c = coords.items;
                for (uint64_t i = coords.count; i > 0; i--, c++, v++) {
                    v->x = *c * sa;
                    v->y = *c * ca;
                }
                coords.clear();
                type = RepetitionType::Explicit;
                offsets = temp;
            } else if (x_reflection || magnification != 1) {
                if (x_reflection) magnification = -magnification;
                double* c = coords.items;
                for (uint64_t i = coords.count; i > 0; i--) {
                    *c++ *= magnification;
                }
            }
        } break;
        case RepetitionType::Explicit: {
            Vec2* v = offsets.items;
            if (rotation != 0) {
                Vec2 r = {magnification * cos(rotation), magnification * sin(rotation)};
                if (x_reflection) {
                    for (uint64_t i = offsets.count; i > 0; i--, v++) {
                        *v = cplx_mul(cplx_conj(*v), r);
                    }
                } else {
                    for (uint64_t i = offsets.count; i > 0; i--, v++) {
                        *v = cplx_mul(*v, r);
                    }
                }
            } else if (x_reflection && magnification != 1) {
                for (uint64_t i = offsets.count; i > 0; i--, v++) {
                    v->x *= magnification;
                    v->y *= -magnification;
                }
            } else if (x_reflection) {
                for (uint64_t i = offsets.count; i > 0; i--, v++) {
                    v->y = -v->y;
                }
            } else if (magnification != 1) {
                for (uint64_t i = offsets.count; i > 0; i--, v++) {
                    *v *= magnification;
                }
            }
        } break;
        default:
            return;
    }
}

}  // namespace gdstk
