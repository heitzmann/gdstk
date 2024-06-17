#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>

#include <gdstk/array.hpp>
#include <gdstk/raithdata.hpp>

namespace gdstk {

void RaithData::copy_from(const RaithData& raith_data) {
    dwelltime_selection = raith_data.dwelltime_selection;
    pitch_parallel_to_path = raith_data.pitch_parallel_to_path;
    pitch_perpendicular_to_path = raith_data.pitch_perpendicular_to_path;
    pitch_scale = raith_data.pitch_scale;
    periods = raith_data.periods;
    grating_type = raith_data.grating_type;
    dots_per_cycle = raith_data.dots_per_cycle;
}

}  // namespace gdstk