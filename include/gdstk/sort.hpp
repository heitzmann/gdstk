/*
Copyright 2021 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_SORT
#define GDSTK_HEADER_SORT

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <inttypes.h>

#include "array.hpp"

namespace gdstk {

// Insertion sort

template <class T>
void insertion_sort(T* items, int64_t count, bool (*sorted)(const T&, const T&)) {
    for (int64_t i = 1; i < count; i++) {
        T store = items[i];
        int64_t j = i - 1;
        while (j >= 0 && sorted(store, items[j])) {
            items[j + 1] = items[j];
            j--;
        }
        items[j + 1] = store;
    }
}

// Heap sort

#define GDSTK_HEAP_PARENT(n) (((n)-1) >> 1)
#define GDSTK_HEAP_LEFT(n) ((n)*2 + 1)
#define GDSTK_HEAP_RIGHT(n) ((n)*2 + 2)

template <class T>
inline void swap_values(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

template <class T>
int64_t leaf_search(T* items, int64_t j, int64_t end, bool (*sorted)(const T&, const T&)) {
    int64_t jr = GDSTK_HEAP_RIGHT(j);
    while (jr <= end) {
        int64_t jl = GDSTK_HEAP_LEFT(j);
        if (sorted(items[jl], items[jr])) {
            j = jr;
        } else {
            j = jl;
        }
        jr = GDSTK_HEAP_RIGHT(j);
    }
    int64_t jl = GDSTK_HEAP_LEFT(j);
    if (jl <= end) {
        j = jl;
    }
    return j;
}

template <class T>
void sift_down(T* items, int64_t start, int64_t end, bool (*sorted)(const T&, const T&)) {
    int64_t j = leaf_search(items, start, end, sorted);
    while (sorted(items[j], items[start])) {
        j = GDSTK_HEAP_PARENT(j);
    }
    T store = items[j];
    items[j] = items[start];
    while (j > start) {
        int64_t parent = GDSTK_HEAP_PARENT(j);
        swap_values(store, items[parent]);
        j = parent;
    }
}

template <class T>
void heap_sort(T* items, int64_t count, bool (*sorted)(const T&, const T&)) {
    // Build heap
    for (int64_t start = GDSTK_HEAP_PARENT(count - 1); start >= 0; start--) {
        sift_down(items, start, count - 1, sorted);
    }
    // Sort
    int64_t end = count - 1;
    while (end > 0) {
        swap_values(items[0], items[end]);
        end--;
        sift_down(items, 0, end, sorted);
    }
}

// Intro sort

template <class T>
int64_t partition(T* items, int64_t count, bool (*sorted)(const T&, const T&)) {
    const int64_t hi = count - 1;
    const int64_t mid = hi >> 2;
    if (sorted(items[hi], items[0])) swap_values(items[0], items[hi]);
    if (sorted(items[mid], items[0])) swap_values(items[0], items[mid]);
    if (sorted(items[hi], items[mid])) swap_values(items[mid], items[hi]);
    const T pivot = items[mid];
    int64_t i = -1;
    int64_t j = count;
    while (true) {
        do {
            i++;
        } while (sorted(items[i], pivot));
        do {
            j--;
        } while (sorted(pivot, items[j]));
        if (i >= j) {
            return j + 1;
        }
        swap_values(items[i], items[j]);
    }
}

template <class T>
void intro_sort(T* items, int64_t count, int64_t max_depth, bool (*sorted)(const T&, const T&)) {
    if (count <= 1) {
        return;
    } else if (count == 2) {
        if (sorted(items[1], items[0])) {
            swap_values(items[0], items[1]);
        }
        return;
    } else if (count <= 16) {
        insertion_sort(items, count, sorted);
    } else if (max_depth == 0) {
        heap_sort(items, count, sorted);
    } else {
        int64_t p = partition(items, count, sorted);
        intro_sort(items, p, max_depth - 1, sorted);
        intro_sort(items + p, count - p, max_depth - 1, sorted);
    }
}

template <class T>
inline bool default_sorted(const T& a, const T& b) {
    return a < b;
}

template <class T>
void sort(T* items, int64_t count, bool (*sorted)(const T&, const T&)) {
    // max_depth = 2 floor(log₂(count))
    int64_t max_depth = 0;
    for (int64_t i = count; i > 0; i >>= 1) max_depth++;
    max_depth = 2 * (max_depth - 1);
    intro_sort(items, count, max_depth, sorted);
}

template <class T>
void sort(T* items, int64_t count) {
    sort(items, count, default_sorted<T>);
}

template <class T>
inline void sort(Array<T>& array, bool (*sorted)(const T&, const T&)) {
    sort(array.items, array.count, sorted);
};

template <class T>
inline void sort(Array<T>& array) {
    sort(array.items, array.count);
};

}  // namespace gdstk

#endif
