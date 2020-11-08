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

// TODO:
// - Respect deep_copy
// - copy_from/clear: follow properties
// - No changes with transformations
// - Polygon->fracture -> copy
// - Play nice with bounding_box and area
// - Fix all transforms
// - Cell.get_polygons(true, …) and Reference.polygons(true, …) -> copy from paths
// - Cell.flatten (apply reference repetitions, but not resulting geometry's)
// - path.to_polygon -> copy
// - create python object and all interfaces

enum struct RepetitionType {
    Rectangular = 0,
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
        Array<Vec2> positions;  // Explicit (10, 11)
        Array<double> coords;   // ExplicitX, ExplicitY (Oasis 4, 5, 6, 7)
    };
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void print() const {
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
                positions.print(true);
                break;
            default:
                const int64_t n = 12;
                printf("Explicit %c repetition <%p>:",
                       type == RepetitionType::ExplicitX ? 'X' : 'Y', this);
                for (int64_t i = 0; i < coords.size; i += n) {
                    for (int64_t j = 0; j < n && i + j < coords.size; j++) {
                        printf(" %lg", coords[i + j]);
                    }
                    putchar('\n');
                }
        }
    }

    void clear() {
        if (type == RepetitionType::Explicit) {
            positions.clear();
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
                positions.copy_from(repetition.positions);
                break;
            default:
                coords.copy_from(repetition.coords);
        }
    }

    int64_t get_size() const {
        switch (type) {
            case RepetitionType::Rectangular:
            case RepetitionType::Regular:
                return columns * rows;
            case RepetitionType::Explicit:
                return positions.size;
            default:
                return coords.size;
        }
        return 0;
    }

    // NOTE: the coordinates for the original (0, 0) are includded as 1st element
    void get_offsets(Array<Vec2>& result) const {
        int64_t size = get_size();
        result.ensure_slots(size + 1);
        double* c_item;
        double* c = (double*)(result.items + result.size);
        *c++ = 0;
        *c++ = 0;
        result.size++;
        switch (type) {
            case RepetitionType::Rectangular:
                for (int64_t j = 1; j < rows; j++) {
                    *c++ = 0;
                    *c++ = j * spacing.y;
                }
                for (int64_t i = 1; i < columns; i++) {
                    double cx = i * spacing.x;
                    for (int64_t j = 0; j < rows; j++) {
                        *c++ = cx;
                        *c++ = j * spacing.y;
                    }
                }
                result.size += size;
                break;
            case RepetitionType::Regular:
                for (int64_t j = 1; j < rows; j++) {
                    *c++ = j * v2.x;
                    *c++ = j * v2.y;
                }
                for (int64_t i = 1; i < columns; i++) {
                    Vec2 vi = i * v1;
                    for (int64_t j = 0; j < rows; j++) {
                        *c++ = vi.x + j * v2.x;
                        *c++ = vi.y + j * v2.y;
                    }
                }
                result.size += size;
                break;
            case RepetitionType::ExplicitX:
                c_item = coords.items;
                for (int64_t j = 0; j < size; j++) {
                    *c++ = *c_item++;
                    *c++ = 0;
                }
                result.size += size;
                break;
            case RepetitionType::ExplicitY:
                c_item = coords.items;
                for (int64_t j = 0; j < size; j++) {
                    *c++ = 0;
                    *c++ = *c_item++;
                }
                result.size += size;
                break;
            default:
                result.extend(positions);
        }
    }
};

inline Repetition* repetition_copy(const Repetition* repetition) {
    if (repetition == NULL) return NULL;
    Repetition* result = (Repetition*)allocate_clear(sizeof(Repetition));
    result->copy_from(*repetition);
    return result;
};

}  // namespace gdstk

#endif
