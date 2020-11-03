/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef __ALLOCATOR_H__
#define __ALLOCATOR_H__

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define USE_MALLOC

namespace gdstk {

#ifdef USE_MALLOC

inline void* allocate(uint64_t size) { return malloc(size); };

inline void* reallocate(void* ptr, uint64_t size) { return realloc(ptr, size); };

inline void* allocate_clear(uint64_t size) { return calloc(1, size); };

inline void free_allocation(void* ptr) { free(ptr); };

inline void gdstk_finalize(){};

#else  // USE_MALLOC

void* allocate(uint64_t size);

void* reallocate(void* ptr, uint64_t size);

void* allocate_clear(uint64_t size);

void free_allocation(void* ptr);

void gdstk_finalize();

#endif  // USE_MALLOC

}  // namespace gdstk

#endif
