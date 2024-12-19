#include <stddef.h>
#include <string.h>

#include <gdstk/raithdata.hpp>
#include <gdstk/utils.hpp>

namespace gdstk {

void PXXData::little_endian_swap() {
    little_endian_swap16((uint16_t*)(this + offsetof(PXXData, unused)), 1);
    little_endian_swap64((uint64_t*)(this + offsetof(PXXData, pitch_parallel_to_path)), 3);
    little_endian_swap32((uint32_t*)(this + offsetof(PXXData, periods)), 5);
    little_endian_swap64((uint64_t*)(this + offsetof(PXXData, ret_stage_speed)), 2);
    little_endian_swap16((uint16_t*)(this + offsetof(PXXData, revision)), 1);
}

void RaithData::clear() {
    if (base_cell_name) {
        free_allocation(base_cell_name);
        base_cell_name = NULL;
    }
}

void RaithData::copy_from(const RaithData& raith_data) {
    pitch_parallel_to_path = raith_data.pitch_parallel_to_path;
    pitch_perpendicular_to_path = raith_data.pitch_perpendicular_to_path;
    pitch_scale = raith_data.pitch_scale;
    periods = raith_data.periods;
    grating_type = raith_data.grating_type;
    dots_per_cycle = raith_data.dots_per_cycle;
    dwelltime_selection = raith_data.dwelltime_selection;
    if (base_cell_name) free_allocation(base_cell_name);
    base_cell_name = NULL;
    if (raith_data.base_cell_name) base_cell_name = copy_string(raith_data.base_cell_name, NULL);
}

PXXData RaithData::to_pxxdata(double scaling) const {
    PXXData pd{};
    pd.dwelltime_selection = dwelltime_selection;
    pd.pitch_parallel_to_path = pitch_parallel_to_path * scaling;
    pd.pitch_perpendicular_to_path = pitch_perpendicular_to_path * scaling;
    pd.pitch_scale = pitch_scale * scaling;
    pd.periods = periods;
    pd.grating_type = grating_type;
    pd.dots_per_cycle = dots_per_cycle;
    memset(pd.free, 0, sizeof(pd.free));
    pd.revision = 1;
    return pd;
}

void RaithData::from_pxxdata(PXXData const& pxxdata) {
    pitch_parallel_to_path = pxxdata.pitch_parallel_to_path;
    pitch_perpendicular_to_path = pxxdata.pitch_perpendicular_to_path;
    pitch_scale = pxxdata.pitch_scale;
    periods = pxxdata.periods;
    grating_type = pxxdata.grating_type;
    dots_per_cycle = pxxdata.dots_per_cycle;
    dwelltime_selection = pxxdata.dwelltime_selection;
}

}  // namespace gdstk
