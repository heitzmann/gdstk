/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_PATHCOMMON
#define GDSTK_HEADER_PATHCOMMON

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

namespace gdstk {

enum struct EndType {
    Flush = 0,
    Round,
    HalfWidth,
    Extended,  // Use end_extensions
    Smooth,    // Becomes Round if simple_path
    Function,  // Use end_function(…)
};

enum struct JoinType {
    Natural = 0,  // Only bevel acute joins
    Miter,
    Bevel,
    Round,
    Smooth,    // Becomes Round if simple_path
    Function,  // Use join_function(…)
};

enum struct BendType {
    None = 0,
    Circular,  // Use bend_radius
    Function,  // Use bend_function(…)
};

inline const char* end_type_name(EndType end_type) {
    switch (end_type) {
        case EndType::Flush:
            return "flush";
        case EndType::Round:
            return "round";
        case EndType::HalfWidth:
            return "half-width";
        case EndType::Extended:
            return "extended";
        case EndType::Smooth:
            return "smooth";
        case EndType::Function:
            return "function";
    }
    return "unknown";
}

inline const char* join_type_name(JoinType join_type) {
    switch (join_type) {
        case JoinType::Natural:
            return "natural";
        case JoinType::Miter:
            return "miter";
        case JoinType::Bevel:
            return "bevel";
        case JoinType::Round:
            return "round";
        case JoinType::Smooth:
            return "smooth";
        case JoinType::Function:
            return "function";
    }
    return "unknown";
}

inline const char* bend_type_name(BendType bend_type) {
    switch (bend_type) {
        case BendType::None:
            return "none";
        case BendType::Circular:
            return "circular";
        case BendType::Function:
            return "function";
    }
    return "unknown";
}

}  // namespace gdstk

#endif
