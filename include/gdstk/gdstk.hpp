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

#define GDSTK_VERSION "0.9.61"

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

#include "array.hpp"
#include "cell.hpp"
#include "clipper_tools.hpp"
#include "curve.hpp"
#include "flexpath.hpp"
#include "gdsii.hpp"
#include "gdswriter.hpp"
#include "label.hpp"
#include "library.hpp"
#include "map.hpp"
#include "oasis.hpp"
#include "pathcommon.hpp"
#include "polygon.hpp"
#include "raithdata.hpp"
#include "rawcell.hpp"
#include "reference.hpp"
#include "repetition.hpp"
#include "robustpath.hpp"
#include "set.hpp"
#include "sort.hpp"
#include "style.hpp"
#include "utils.hpp"
#include "vec.hpp"

#endif
