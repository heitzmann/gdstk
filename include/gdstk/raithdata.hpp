#ifndef GDSTK_HEADER_RAITHDATA
#define GDSTK_HEADER_RAITHDATA

namespace gdstk {

struct RaithData {
    int dwelltime_selection;
    double pitch_parallel_to_path;
    double pitch_perpendicular_to_path;
    double pitch_scale;
    int periods;
    int grating_type;
    int dots_per_cycle;

    void copy_from(const RaithData raith_data);
    void clear();
};

}  // namespace gdstk

#endif