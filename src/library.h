/*
Copyright 2020 Lucas Heitzmann Gabrielli.
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
    Property* properties;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void print(bool all) const;

    void clear() {
        if (name) free_allocation(name);
        name = NULL;
        cell_array.clear();
        rawcell_array.clear();
        properties_clear(properties);
    }

    void copy_from(const Library& library, bool deep_copy);
    void top_level(Array<Cell*>& top_cells, Array<RawCell*>& top_rawcells) const;

    void write_gds(const char* filename, uint64_t max_points, std::tm* timestamp) const;
};

Library read_gds(const char* filename, double unit, double tolerance);

Library read_oas(const char* filename, double unit, double tolerance);

int gds_units(const char* filename, double& unit, double& precision);

int oas_precision(const char* filename, double& precision);

// TODO: add function to perform validation
// bool oas_validate(const char filename);

}  // namespace gdstk

#endif
