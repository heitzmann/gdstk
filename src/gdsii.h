/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __GDSII_H__
#define __GDSII_H__

#define _USE_MATH_DEFINES

#include <cstdint>
#include <cstdio>

namespace gdstk {

uint64_t gdsii_real_from_double(double value);

double gdsii_real_to_double(uint64_t real);

// Read record and make necessary swaps
uint32_t read_record(FILE* in, uint8_t* buffer);

}  // namespace gdstk

#endif
