/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_GDSTK
#define GDSTK_HEADER_GDSTK

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#define GDSTK_VERSION "0.9.42"

// If GDSTK_CUSTOM_ALLOCATOR is defined, the user must supply implementations
// for the following dynamic memory management functions:
// void* allocate(uint64_t size);
// void* reallocate(void* ptr, uint64_t size);
// void* allocate_clear(uint64_t size);
// void free_allocation(void* ptr);
// They will be used throughout the library instead of malloc, realloc, calloc
// and free.
//
// #define GDSTK_CUSTOM_ALLOCATOR

// After installation, this should be the only header required to be included
// by the user.  All other headers are included below.

#include <gdstk/array.hpp>
#include <gdstk/cell.hpp>
#include <gdstk/clipper_tools.hpp>
#include <gdstk/curve.hpp>
#include <gdstk/flexpath.hpp>
#include <gdstk/gdsii.hpp>
#include <gdstk/gdswriter.hpp>
#include <gdstk/label.hpp>
#include <gdstk/library.hpp>
#include <gdstk/map.hpp>
#include <gdstk/oasis.hpp>
#include <gdstk/pathcommon.hpp>
#include <gdstk/polygon.hpp>
#include <gdstk/rawcell.hpp>
#include <gdstk/reference.hpp>
#include <gdstk/repetition.hpp>
#include <gdstk/robustpath.hpp>
#include <gdstk/set.hpp>
#include <gdstk/sort.hpp>
#include <gdstk/style.hpp>
#include <gdstk/utils.hpp>
#include <gdstk/vec.hpp>

#endif
