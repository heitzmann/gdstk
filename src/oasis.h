/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __OASIS_H__
#define __OASIS_H__

#define _USE_MATH_DEFINES

#include <cstdint>
#include <cstdio>

namespace gdstk {

uint64_t oasis_read_uint(FILE* in);

void oasis_write_uint(FILE* out, uint64_t value);

int64_t oasis_read_int(FILE* in);

void oasis_write_int(FILE* out, int64_t value);

double oasis_read_real(FILE* in);

void oasis_write_real(FILE* out, double value);

}  // namespace gdstk

#endif

