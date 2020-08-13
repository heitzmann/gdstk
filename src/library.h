/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __LIBRARY_H__
#define __LIBRARY_H__

#include <cstdio>
#include <ctime>

#include "array.h"
#include "cell.h"

namespace gdstk {

struct Library {
    char* name;
    double unit;
    double precision;
    Array<Cell*> cell_array;
    Array<RawCell*> rawcell_array;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void print(bool all) const;

    void clear() {
        if (name) free(name);
        name = NULL;
        cell_array.clear();
        rawcell_array.clear();
    }

    void copy_from(const Library& library, bool deep_copy);
    void top_level(Array<Cell*>& top_cells, Array<RawCell*>& top_rawcells) const;

    void write_gds(FILE* out, int64_t max_points, std::tm* timestamp) const;
};

Library read_gds(FILE* in, double unit);

int gds_units(FILE* in, double& unit, double& precision);

}  // namespace gdstk

#endif
