/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_ALLOCATOR
#define GDSTK_HEADER_ALLOCATOR

#define __STDC_FORMAT_MACROS 1

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace gdstk {

#ifdef GDSTK_CUSTOM_ALLOCATOR

void* allocate(uint64_t size);

void* reallocate(void* ptr, uint64_t size);

void* allocate_clear(uint64_t size);

void free_allocation(void* ptr);

#else  // GDSTK_CUSTOM_ALLOCATOR

inline void* allocate(uint64_t size) { return malloc(size); };

inline void* reallocate(void* ptr, uint64_t size) { return realloc(ptr, size); };

inline void* allocate_clear(uint64_t size) { return calloc(1, size); };

inline void free_allocation(void* ptr) { free(ptr); };

#endif  // GDSTK_CUSTOM_ALLOCATOR

}  // namespace gdstk

#endif
