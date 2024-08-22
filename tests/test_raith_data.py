import gdstk


def test_init():
    raith_data = gdstk.RaithData("CELL", 1, 2, 3, 4, 5, 6, 7)
    assert raith_data.dwelltime_selection == 1
    assert raith_data.pitch_parallel_to_path == 2
    assert raith_data.pitch_perpendicular_to_path == 3
    assert raith_data.pitch_scale == 4
    assert raith_data.periods == 5
    assert raith_data.grating_type == 6
    assert raith_data.dots_per_cycle == 7
