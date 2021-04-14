/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_GDSTK
#define GDSTK_HEADER_GDSTK

#define __STDC_FORMAT_MACROS
#define _USE_MATH_DEFINES

#define GDSTK_VERSION "0.4.0"

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

#include "gdstk/array.h"
#include "gdstk/cell.h"
#include "gdstk/clipper_tools.h"
#include "gdstk/curve.h"
#include "gdstk/flexpath.h"
#include "gdstk/gdsii.h"
#include "gdstk/gdswriter.h"
#include "gdstk/label.h"
#include "gdstk/library.h"
#include "gdstk/map.h"
#include "gdstk/oasis.h"
#include "gdstk/pathcommon.h"
#include "gdstk/polygon.h"
#include "gdstk/rawcell.h"
#include "gdstk/reference.h"
#include "gdstk/repetition.h"
#include "gdstk/robustpath.h"
#include "gdstk/style.h"
#include "gdstk/utils.h"
#include "gdstk/vec.h"

#endif
