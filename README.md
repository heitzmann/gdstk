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
They represent the best average time to run each function out of 32 sets of 32 runs each.

| Benchmark        |    Gdspy 1.6     |  Gdstk 0.1.dev0  |   Gain   |
| :--------------- | :--------------: | :--------------: | :------: |
| 10k_rectangles   |      221 ms      |     6.44 ms      |   34.3   |
| boolean-offset   |      410 μs      |     49.6 μs      |   8.27   |
| bounding_box     |      628 μs      |     8.56 μs      |   73.4   |
| curves           |     2.94 ms      |     65.6 μs      |   44.8   |
| flatten          |      995 μs      |     9.85 μs      |   101    |
| flexpath-param   |      7.2 ms      |     1.42 ms      |   5.06   |
| flexpath         |     5.45 ms      |     24.9 μs      |   219    |
| fracture         |     1.69 ms      |      854 μs      |   1.97   |
| inside           |     38.8 μs      |     8.92 μs      |   4.35   |
| read_gds         |     4.21 ms      |     73.5 μs      |   57.2   |
| read_rawcells    |      339 μs      |      53 μs       |   6.39   |
| robustpath       |      369 μs      |     13.4 μs      |   27.5   |

Memory usage per object for 100.000 objects using Python 3.8:

| Object               |    Gdspy 1.6     |  Gdstk 0.1.dev0  | Reduction |
| :------------------- | :--------------: | :--------------: | :-------: |
| Rectangle            |      592 B       |      184 B       |    69%    |
| Circle (r = 10)      |     1.67 kB      |     1.23 kB      |    27%    |
| FlexPath segment     |     1.42 kB      |      394 B       |    73%    |
| FlexPath arc         |     2.23 kB      |     1.45 kB      |    35%    |
| RobustPath segment   |     2.78 kB      |      875 B       |    69%    |
| RobustPath arc       |     2.58 kB      |      874 B       |    67%    |
| Label                |      397 B       |      167 B       |    58%    |
| Reference            |      160 B       |      133 B       |    17%    |
| Reference (array)    |      187 B       |      135 B       |    28%    |
| Cell                 |      434 B       |      214 B       |    51%    |
