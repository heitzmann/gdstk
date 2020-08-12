/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef __PATHCOMMON_H__
#define __PATHCOMMON_H__

namespace gdstk {

enum struct EndType {
    Flush = 0,
    Extended,  // Use end_extensions
    Round,
    Smooth,    // Becomes Round if simple_path
    Function,  // Uses end_function(...)
};

enum struct JoinType {
    Natural = 0,
    Miter,
    Bevel,
    Round,
    Smooth,    // Becomes Round if simple_path
    Function,  // Uses join_function(...)
};

enum struct BendType {
    None = 0,
    Circular,  // Uses bend_radius
    Function,  // Uses bend_function(...)
};

}  // namespace gdstk

#endif
