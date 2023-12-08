/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_ARRAY
#define GDSTK_HEADER_ARRAY

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#define ARRAY_GROWTH_FACTOR 2
#define INITIAL_ARRAY_CAPACITY 4

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allocator.hpp"
#include "vec.hpp"

namespace gdstk {

template <class T>
struct Array {
    uint64_t capacity;  // allocated capacity
    uint64_t count;     // number of slots used
    T* items;           // slots

    T& operator[](uint64_t idx) { return items[idx]; }
    const T& operator[](uint64_t idx) const { return items[idx]; }

    void print(bool all) const {
        printf("Array <%p>, count %" PRIu64 "/%" PRIu64 "\n", this, count, capacity);
        if (all && count > 0) {
            printf("<%p>", (void*)items[0]);
            for (uint64_t i = 0; i < count; ++i) {
                printf(" <%p>", (void*)items[i]);
            }
            putchar('\n');
        }
    }

    void clear() {
        if (items) free_allocation(items);
        items = NULL;
        capacity = 0;
        count = 0;
    }

    bool contains(const T& item) const { return index(item) < count; }

    // Return the index of an array item.  If the item is not found, return the
    // array count.
    uint64_t index(const T& item) const {
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

    // Does NOT check capacity. To be used after ensure_slots, for example.
    void append_unsafe(T item) { items[count++] = item; }

    // Insert item at specified index, pushing the remaining forward
    void insert(uint64_t index_, T item) {
        if (count == capacity) {
            capacity = capacity >= INITIAL_ARRAY_CAPACITY ? capacity * ARRAY_GROWTH_FACTOR
                                                          : INITIAL_ARRAY_CAPACITY;
            items = (T*)reallocate(items, sizeof(T) * capacity);
        }
        if (index_ >= count) {
            append_unsafe(item);
        } else {
            memmove(items + index_ + 1, items + index_, sizeof(T) * (count - index_));
            items[index_] = item;
            count++;
        }
    }

    // Remove the item at index by substituting it with the last item in the
    // array.
    void remove_unordered(uint64_t index_) { items[index_] = items[--count]; }

    // Remove the item at index and pull the remaining to fill the gap.
    void remove(uint64_t index_) {
        memmove(items + index_, items + index_ + 1, sizeof(T) * ((--count) - index_));
    }

    // Remove (ordered) the first occurrence of a specific item in the array.
    // Return false if the item cannot be found.
    bool remove_item(const T& item) {
        uint64_t i = index(item);
        if (i == count) return false;
        remove(i);
        return true;
    }

    // Ensure at least the specified number of free slots at the end
    void ensure_slots(uint64_t free_slots) {
        if (capacity < count + free_slots) {
            capacity = count + free_slots;
            items = (T*)reallocate(items, sizeof(T) * capacity);
        }
    }

    // Extend the array by appending all elements from src (in order)
    void extend(const Array<T>& src) {
        ensure_slots(src.count);
        memcpy(items + count, src.items, sizeof(T) * src.count);
        count += src.count;
    }

    // The instance should be zeroed before copy_from
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
    printf("Array <%p>, count %" PRIu64 "/%" PRIu64 "\n", this, count, capacity);
    if (all && count > 0) {
        printf("(%lg, %lg)", items[0].x, items[0].y);
        for (uint64_t i = 1; i < count; ++i) {
            printf(" (%lg, %lg)", items[i].x, items[i].y);
        }
        putchar('\n');
    }
}

template <>
inline void Array<IntVec2>::print(bool all) const {
    printf("Array <%p>, count %" PRIu64 "/%" PRIu64 "\n", this, count, capacity);
    if (all && count > 0) {
        printf(" (%" PRId64 ", %" PRId64 ")", items[0].x, items[0].y);
        for (uint64_t i = 1; i < count; ++i) {
            printf(" (%" PRId64 ", %" PRId64 ")", items[i].x, items[i].y);
        }
        putchar('\n');
    }
}

template <>
inline void Array<double>::print(bool all) const {
    printf("Array <%p>, count %" PRIu64 "/%" PRIu64 "\n", this, count, capacity);
    if (all && count > 0) {
        printf(" %lg", items[0]);
        for (uint64_t i = 1; i < count; ++i) {
            printf(" %lg", items[i]);
        }
        putchar('\n');
    }
}

}  // namespace gdstk

#endif
