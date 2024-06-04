#include "raithdata.hpp"

namespace gdstk {

void RaithData::copy_from(const RaithData raith_data) {
    dwelltime_selection = raith_data.dwelltime_selection;
    pitch_parallel_to_path = raith_data.pitch_parallel_to_path;
    pitch_perpendicular_to_path = raith_data.pitch_perpendicular_to_path;
    pitch_scale = raith_data.pitch_scale;
    periods = raith_data.periods;
    grating_type = raith_data.grating_type;
    dots_per_cycle = raith_data.dots_per_cycle;
}

void RaithData::clear() {
    dwelltime_selection = 0;
    pitch_parallel_to_path = 0;
    pitch_perpendicular_to_path = 0;
    pitch_scale = 0;
    periods = 0;
    grating_type = 0;
    dots_per_cycle = 0;
}

}  // namespace gdstk