/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_ARRAY
#define GDSTK_HEADER_ARRAY

#define _USE_MATH_DEFINES
#define __STDC_FORMAT_MACROS

#define ARRAY_GROWTH_FACTOR 2
#define INITIAL_ARRAY_CAPACITY 4

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allocator.h"
#include "vec.h"

namespace gdstk {

template <class T>
struct Array {
    uint64_t capacity;  // allocated capacity
    uint64_t count;     // number of slots used
    T* items;           // slots

    T& operator[](uint64_t idx) { return items[idx]; }
    const T& operator[](uint64_t idx) const { return items[idx]; }

    void print(bool all) const {
        const uint8_t n = 6;
        printf("Array <%p>, count %" PRIu64 "/%" PRIu64 "\n", this, count, capacity);
        if (all) {
            for (uint64_t i = 0; i < count; i += n) {
                for (uint64_t j = 0; j < n && i + j < count; j++) {
                    if (j > 0) printf(" ");
                    printf("<%p>", items[i + j]);
                }
                putchar('\n');
            }
        }
    }

    void clear() {
        if (items) free_allocation(items);
        items = NULL;
        capacity = 0;
        count = 0;
    }

    bool contains(const T item) const {
        T* it = items;
        for (uint64_t j = 0; j < count; j++)
            if (*(it++) == item) return true;
        return false;
    }

    uint64_t index(const T item) const {
        T* it = items;
        for (uint64_t j = 0; j < count; j++)
            if (*(it++) == item) return j;
        return count;
    }

    void append(T item) {
        if (count == capacity) {
            capacity = capacity >= INITIAL_ARRAY_CAPACITY ? capacity * ARRAY_GROWTH_FACTOR
                                                          : INITIAL_ARRAY_CAPACITY;
            items = (T*)reallocate(items, sizeof(T) * capacity);
        }
        items[count++] = item;
    }

    void append_unsafe(T item) { items[count++] = item; }

    void insert(uint64_t index, T item) {
        if (index >= count) {
            append(item);
        } else {
            if (count == capacity) {
                capacity = capacity >= INITIAL_ARRAY_CAPACITY ? capacity * ARRAY_GROWTH_FACTOR
                                                              : INITIAL_ARRAY_CAPACITY;
                items = (T*)reallocate(items, sizeof(T) * capacity);
            }
            memmove(items + index + 1, items + index, sizeof(T) * (count - index));
            items[index] = item;
        }
    }

    void remove_unordered(uint64_t index) { items[index] = items[--count]; }

    void remove(uint64_t index) {
        memmove(items + index, items + index + 1, sizeof(T) * ((--count) - index));
    }

    bool remove_item(const T item) {
        uint64_t i = index(item);
        if (i == count) return false;
        remove(i);
        return true;
    }

    void ensure_slots(uint64_t free_slots) {
        if (capacity < count + free_slots) {
            capacity = count + free_slots;
            items = (T*)reallocate(items, sizeof(T) * capacity);
        }
    }

    void extend(const Array<T>& src) {
        ensure_slots(src.count);
        memcpy(items + count, src.items, sizeof(T) * src.count);
        count += src.count;
    }

    void copy_from(const Array<T>& src) {
        capacity = src.count;
        count = src.count;
        if (count > 0) {
            items = (T*)allocate(sizeof(T) * capacity);
            memcpy(items, src.items, sizeof(T) * count);
        } else {
            items = NULL;
        }
    }
};

template <>
inline void Array<Vec2>::print(bool all) const {
    const uint8_t n = 6;
    printf("Array <%p>, count %" PRIu64 "/%" PRIu64 "\n", this, count, capacity);
    if (all) {
        for (uint64_t i = 0; i < count; i += n) {
            for (uint64_t j = 0; j < n && i + j < count; j++) {
                if (j > 0) printf(" ");
                printf("(%lg, %lg)", items[i + j].x, items[i + j].y);
            }
            putchar('\n');
        }
    }
}

template <>
inline void Array<IntVec2>::print(bool all) const {
    const uint8_t n = 6;
    printf("Array <%p>, count %" PRIu64 "/%" PRIu64 "\n", this, count, capacity);
    if (all) {
        for (uint64_t i = 0; i < count; i += n) {
            for (uint64_t j = 0; j < n && i + j < count; j++) {
                if (j > 0) printf(" ");
                printf("(%" PRId64 ", %" PRId64 ")", items[i + j].x, items[i + j].y);
            }
            putchar('\n');
        }
    }
}

template <>
inline void Array<double>::print(bool all) const {
    const uint8_t n = 12;
    printf("Array <%p>, count %" PRIu64 "/%" PRIu64 "\n", this, count, capacity);
    if (all) {
        for (uint64_t i = 0; i < count; i += n) {
            for (uint64_t j = 0; j < n && i + j < count; j++) {
                if (j > 0) printf(" ");
                printf("%lg", items[i + j]);
            }
            putchar('\n');
        }
    }
}

}  // namespace gdstk

#endif
