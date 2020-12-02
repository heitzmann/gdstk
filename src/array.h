/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __ARRAY_H__
#define __ARRAY_H__

#define ARRAY_GROWTH_FACTOR 2
#define INITIAL_ARRAY_CAPACITY 4

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "allocator.h"
#include "vec.h"

namespace gdstk {

template <class T>
struct Array {
    uint64_t capacity;  // allocated capacity
    uint64_t size;      // number of slots used
    T* items;           // slots

    T& operator[](uint64_t idx) { return items[idx]; }
    const T& operator[](uint64_t idx) const { return items[idx]; }

    void print(bool all) const {
        const uint8_t n = 6;
        printf("Array <%p>, size %" PRIu64 "/%" PRIu64 "\n", this, size, capacity);
        if (all) {
            for (uint64_t i = 0; i < size; i += n) {
                for (uint64_t j = 0; j < n && i + j < size; j++) {
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
        size = 0;
    }

    bool contains(const T item) const {
        T* it = items;
        for (uint64_t j = 0; j < size; j++)
            if (*(it++) == item) return true;
        return false;
    }

    uint64_t index(const T item) const {
        T* it = items;
        for (uint64_t j = 0; j < size; j++)
            if (*(it++) == item) return j;
        return size;
    }

    void append(T item) {
        if (size == capacity) {
            capacity = capacity >= INITIAL_ARRAY_CAPACITY ? capacity * ARRAY_GROWTH_FACTOR
                                                          : INITIAL_ARRAY_CAPACITY;
            items = (T*)reallocate(items, sizeof(T) * capacity);
        }
        items[size++] = item;
    }

    void append_unsafe(T item) { items[size++] = item; }

    void remove_unordered(uint64_t index) { items[index] = items[--size]; }

    void remove(uint64_t index) {
        memmove(items + index, items + index + 1, sizeof(T) * ((--size) - index));
    }

    bool remove_item(const T item) {
        uint64_t i = index(item);
        if (i == size) return false;
        remove(i);
        return true;
    }

    void ensure_slots(uint64_t free_slots) {
        if (capacity < size + free_slots) {
            capacity = size + free_slots;
            items = (T*)reallocate(items, sizeof(T) * capacity);
        }
    }

    void extend(const Array<T>& src) {
        ensure_slots(src.size);
        memcpy(items + size, src.items, sizeof(T) * src.size);
        size += src.size;
    }

    void copy_from(const Array<T>& src) {
        capacity = src.size;
        size = src.size;
        if (size > 0) {
            items = (T*)allocate(sizeof(T) * capacity);
            memcpy(items, src.items, sizeof(T) * size);
        } else {
            items = NULL;
        }
    }
};

template <>
inline void Array<Vec2>::print(bool all) const {
    const uint8_t n = 6;
    printf("Array <%p>, size %" PRIu64 "/%" PRIu64 "\n", this, size, capacity);
    if (all) {
        for (uint64_t i = 0; i < size; i += n) {
            for (uint64_t j = 0; j < n && i + j < size; j++) {
                if (j > 0) printf(" ");
                printf("(%lg, %lg)", items[i + j].x, items[i + j].y);
            }
            putchar('\n');
        }
    }
}

template <>
inline void Array<double>::print(bool all) const {
    const uint8_t n = 12;
    printf("Array <%p>, size %" PRIu64 "/%" PRIu64 "\n", this, size, capacity);
    if (all) {
        for (uint64_t i = 0; i < size; i += n) {
            for (uint64_t j = 0; j < n && i + j < size; j++) {
                if (j > 0) printf(" ");
                printf("%lg", items[i + j]);
            }
            putchar('\n');
        }
    }
}

}  // namespace gdstk

#endif
