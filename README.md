# GDSTK README

[![Boost Software License - Version 1.0](https://img.shields.io/github/license/heitzmann/gdstk.svg)](https://www.boost.org/LICENSE_1_0.txt)
[![Tests Runner](https://github.com/heitzmann/gdstk/workflows/Tests%20Runner/badge.svg)](https://github.com/heitzmann/gdstk/actions?query=workflow%3A%22Tests+Runner%22)
[![Publish Docs](https://github.com/heitzmann/gdstk/workflows/Publish%20Docs/badge.svg)](https://github.com/heitzmann/gdstk/actions?query=workflow%3A%22Publish+Docs%22)
[![Downloads](https://img.shields.io/github/downloads/heitzmann/gdstk/total.svg)](https://github.com/heitzmann/gdstk/releases)

Gdstk (GDSII Tool Kit) is a C++ library for creation and manipulation of GDSII stream files.
It is also available as a Python module meant to be a successor to [Gdspy](https://github.com/heitzmann/gdspy).

Key features for the creation of complex CAD layouts are included:

* Boolean operations on polygons (AND, OR, NOT, XOR) based on clipping algorithm
* Polygon offset (inward and outward rescaling of polygons)
* Efficient point-in-polygon solutions for large array sets

Typical applications of Gdstk are in the fields of electronic chip design, planar lightwave circuit design, and mechanical engineering.


## Documentation

The complete documentation is available [here](http://heitzmann.github.io/gdstk).

The source files can be found in the _docs_ directory.


## Installation

The C++ library is meant to be used straight from source.
The only requirement is that it must be linked against LAPACK.
The included CMakeLists.txt file can be used as a guide.

### Dependencies for the Python wrapper

* LAPACK
* [CMake](https://cmake.org/)
* [Python](https://www.python.org/)
* [Numpy](https://numpy.org/)
* [Sphinx](https://www.sphinx-doc.org/) and [rtd theme](https://sphinx-rtd-theme.readthedocs.io/) (to build the [documentation](http://heitzmann.github.io/gdstk))

Installation from source should follow the usual method:

```sh
python setup.py install
```

Windows users have the option to use a pre-compiled binary from the [releases](https://github.com/heitzmann/gdstk/releases) page.


## Support

Help support Gdstk development by [donating via PayPal](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=JD2EUE2WPPBQQ)


## Benchmarks

The _benchmarks_ directory contains a few tests to compare the performance gain of the Python interface versus Gdspy.
They are only for reference; the real improvement is heavily dependent on the type of layout and features used.
If maximal performance is important, the library should be used directly from C++, without the Python interface.

These results were obtained on an Intel Core i7-3820 with 8 cores and 64 GB of RAM.
They represent the best average time to run each function out of 32 sets of 32 runs each.

| Benchmark       |    gdspy    |    gdstk    |   Gain  |
| :-------------- | :---------: | :---------: | :-----: |
| boolean-offset  |    411 μs   |   48.2 μs   |   8.53  |
| bounding_box    |    633 μs   |   7.95 μs   |   79.6  |
| curves          |   2.98 ms   |   65.2 μs   |   45.7  |
| flatten         |     1 ms    |   10.5 μs   |   95.6  |
| flexpath-param  |   7.29 ms   |    1.4 ms   |   5.22  |
| flexpath        |   5.54 ms   |   24.5 μs   |   227   |
| fracture        |    1.7 ms   |    850 μs   |    2    |
| inside          |   38.1 μs   |   8.55 μs   |   4.46  |
| read_gds        |   4.24 ms   |   71.6 μs   |   59.2  |
| read_rawcells   |    339 μs   |   52.7 μs   |   6.44  |
| robustpath      |    377 μs   |   12.8 μs   |   29.4  |
