# GDSTK README

[![Boost Software License - Version 1.0](https://img.shields.io/github/license/heitzmann/gdstk.svg)](https://www.boost.org/LICENSE_1_0.txt)
[![Tests Runner](https://github.com/heitzmann/gdstk/workflows/Tests%20Runner/badge.svg)](https://github.com/heitzmann/gdstk/actions)
[![Publish Docs](https://github.com/heitzmann/gdstk/workflows/Publish%20Docs/badge.svg)](http://heitzmann.github.io/gdstk)
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

Timing results were obtained with Python 3.8 on an Intel Core i7-3820 with 8 cores and 64 GB of RAM.
They represent the best average time to run each function out of 32 sets of 8 runs each.

| Benchmark        |    Gdspy 1.6     |    Gdstk 0.1     |   Gain   |
| :--------------- | :--------------: | :--------------: | :------: |
| 10k_rectangles   |      221 ms      |     6.42 ms      |   34.4   |
| 1k_circles       |      567 ms      |      359 ms      |   1.58   |
| boolean-offset   |      413 μs      |     49.7 μs      |   8.31   |
| bounding_box     |      634 μs      |     8.43 μs      |   75.2   |
| curves           |     2.97 ms      |     66.1 μs      |   44.9   |
| flatten          |     1.01 ms      |     10.1 μs      |   100    |
| flexpath-param   |     7.26 ms      |     1.42 ms      |   5.1    |
| flexpath         |     5.48 ms      |     25.2 μs      |   217    |
| fracture         |     1.71 ms      |      857 μs      |   1.99   |
| inside           |     39.4 μs      |     8.41 μs      |   4.68   |
| read_gds         |     7.08 ms      |      108 μs      |   65.6   |
| read_rawcells    |      615 μs      |      74 μs       |   8.31   |
| robustpath       |      375 μs      |     13.6 μs      |   27.6   |

Memory usage per object for 100.000 objects using Python 3.8:

| Object               |    Gdspy 1.6     |    Gdstk 0.1     | Reduction |
| :------------------- | :--------------: | :--------------: | :-------: |
| Rectangle            |      606 B       |      184 B       |    70%    |
| Circle (r = 10)      |     1.67 kB      |     1.23 kB      |    27%    |
| FlexPath segment     |     1.42 kB      |      392 B       |    73%    |
| FlexPath arc         |     2.23 kB      |     1.44 kB      |    35%    |
| RobustPath segment   |     2.79 kB      |      872 B       |    69%    |
| RobustPath arc       |     2.58 kB      |      872 B       |    67%    |
| Label                |      397 B       |      170 B       |    57%    |
| Reference            |      160 B       |      131 B       |    18%    |
| Reference (array)    |      187 B       |      138 B       |    26%    |
| Cell                 |      431 B       |      214 B       |    50%    |
