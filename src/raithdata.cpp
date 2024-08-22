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
    if (raith_data.base_cell_name) base_cell_name = copy_string(raith_data.base_cell_name, NULL);
}

PXXData RaithData::to_pxxdata(double scaling) {
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

ErrorCode RaithData::to_gds(FILE* out, double scaling) {
    ErrorCode error_code = ErrorCode::NoError;

    uint64_t len = base_cell_name ? strlen(base_cell_name) : 0;
    if (len % 2) len++;
    uint16_t sname_start[] = {(uint16_t)(4 + len), 0x1206};
    big_endian_swap16(sname_start, COUNT(sname_start));
    fwrite(sname_start, sizeof(uint16_t), COUNT(sname_start), out);

    fwrite(base_cell_name, 1, len, out);

    PXXData pxxdata = to_pxxdata(scaling);
    pxxdata.little_endian_swap();

    uint16_t buffer_pxx[] = {(uint16_t)(4 + sizeof(PXXData)), 0x6206};
    big_endian_swap16(buffer_pxx, COUNT(buffer_pxx));
    fwrite(buffer_pxx, sizeof(uint16_t), COUNT(buffer_pxx), out);
    fwrite(&pxxdata, 1, sizeof(PXXData), out);

    return error_code;
}

RaithData RaithData::from_pxxdata(PXXData const& pxxdata) {
    RaithData data;
    data.pitch_parallel_to_path = pxxdata.pitch_parallel_to_path;
    data.pitch_perpendicular_to_path = pxxdata.pitch_perpendicular_to_path;
    data.pitch_scale = pxxdata.pitch_scale;
    data.periods = pxxdata.periods;
    data.grating_type = pxxdata.grating_type;
    data.dots_per_cycle = pxxdata.dots_per_cycle;
    data.dwelltime_selection = pxxdata.dwelltime_selection;
    return data;
}

}  // namespace gdstk
