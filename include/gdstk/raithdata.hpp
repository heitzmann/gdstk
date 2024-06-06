#ifndef GDSTK_HEADER_RAITHDATA
#define GDSTK_HEADER_RAITHDATA

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdio.h>

namespace gdstk {

struct RaithData {
    uint8_t dwelltime_selection;
    double pitch_parallel_to_path;
    double pitch_perpendicular_to_path;
    double pitch_scale;
    int32_t periods;
    int32_t grating_type;
    int32_t dots_per_cycle;

    void* owner;

    void copy_from(const RaithData& raith_data);
    void clear();
};

}  // namespace gdstk

#endif