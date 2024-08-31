#ifndef GDSTK_HEADER_RAITHDATA
#define GDSTK_HEADER_RAITHDATA

#include <cstdint>

namespace gdstk {

#pragma pack(push, 1)
struct PXXData {
    uint8_t calc_only;
    uint8_t dwelltime_selection;
    uint16_t unused;
    double pitch_parallel_to_path;
    double pitch_perpendicular_to_path;
    double pitch_scale;
    int32_t periods;
    int32_t grating_type;
    int32_t dots_per_cycle;
    int32_t ret_base_pixel_count;
    int32_t ret_pixel_count;
    double ret_stage_speed;
    double ret_dwell_time;
    uint8_t free[190];
    uint16_t revision;

    void little_endian_swap();
};
#pragma pack(pop)

struct RaithData {
    double pitch_parallel_to_path;
    double pitch_perpendicular_to_path;
    double pitch_scale;
    int32_t periods;
    int32_t grating_type;
    int32_t dots_per_cycle;
    uint8_t dwelltime_selection;
    char* base_cell_name;

    void* owner;

    void clear();

    void copy_from(const RaithData& raith_data);

    PXXData to_pxxdata(double scaling) const;
    void from_pxxdata(PXXData const& pxxdata);
};

}  // namespace gdstk

#endif
