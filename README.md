# GDSTK

[![Boost Software License - Version 1.0](https://img.shields.io/github/license/heitzmann/gdstk.svg)](https://www.boost.org/LICENSE_1_0.txt)
[![Tests Runner](https://github.com/heitzmann/gdstk/actions/workflows/run-tests.yml/badge.svg)](https://github.com/heitzmann/gdstk/actions/workflows/run-tests.yml)
[![Publish Docs](https://github.com/heitzmann/gdstk/actions/workflows/publish-docs.yml/badge.svg)](https://github.com/heitzmann/gdstk/actions/workflows/publish-docs.yml)
[![Package Builder](https://github.com/heitzmann/gdstk/actions/workflows/publish-packages.yml/badge.svg)](https://github.com/heitzmann/gdstk/actions/workflows/publish-packages.yml)
[![Downloads](https://img.shields.io/github/downloads/heitzmann/gdstk/total.svg)](https://github.com/heitzmann/gdstk/releases)

Gdstk (GDSII Tool Kit) is a C++ library for creation and manipulation of GDSII and OASIS files.
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

### C++ library only

The C++ library is meant to be used by including it in your own source code.

If you prefer to install a static library, the included _CMakeLists.txt_ should be a good starting option (use `-DCMAKE_INSTALL_PREFIX=path` to control the installation path):

```sh
cmake -S . -B build
cmake --build build --target install
```

The library depends on [zlib](https://zlib.net/) and [qhull](http://www.qhull.org/)

### Python wrapper

The Python module can be installed via pip, Conda or compiled directly from source.
It depends on:

* [zlib](https://zlib.net/)
* [qhull](http://www.qhull.org/)
* [CMake](https://cmake.org/)
* [Python](https://www.python.org/)
* [Numpy](https://numpy.org/)
* [Sphinx](https://www.sphinx-doc.org/), [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/), and [Sphinx Inline Tabs](https://sphinx-inline-tabs.readthedocs.io/) (to build the [documentation](http://heitzmann.github.io/gdstk))

#### From PyPI

Simply run the following to install the package for the current user:

```sh
pip install --user gdstk
```

Or download and install the available wheels manually.

#### From source

Installation from source requires the `build` module (plus CMake and Ninja, for faster compilation):

```sh
pip install --user build
```

With that, simply build the wheel package using:

```sh
python -m build -w
```

This will create a _dist_ directory containing the compiled _.whl_ package that can be installed with ``pip``.

## Support

Help support Gdstk development by [donating via PayPal](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=JD2EUE2WPPBQQ) or [sponsoring me on GitHub](https://github.com/sponsors/heitzmann).


## Benchmarks

The _benchmarks_ directory contains a few tests to compare the performance gain of the Python interface versus Gdspy.
They are only for reference; the real improvement is heavily dependent on the type of layout and features used.
If maximal performance is important, the library should be used directly from C++, without the Python interface.

Timing results were obtained with Python 3.11 on an Intel Core i7-9750H @ 2.60 GHz
They represent the best average time to run each function out of 16 sets of 8 runs each.

| Benchmark        |   Gdspy 1.6.13   |   Gdstk 0.9.41   |   Gain   |
| :--------------- | :--------------: | :--------------: | :------: |
| 10k_rectangles   |     80.2 ms      |     4.87 ms      |   16.5   |
| 1k_circles       |      312 ms      |      239 ms      |   1.3    |
| boolean-offset   |      187 μs      |     44.7 μs      |   4.19   |
| bounding_box     |     36.7 ms      |      170 μs      |   216    |
| curves           |     1.52 ms      |     30.9 μs      |   49.3   |
| flatten          |      465 μs      |     8.17 μs      |   56.9   |
| flexpath         |     2.88 ms      |     16.1 μs      |   178    |
| flexpath-param   |      2.8 ms      |      585 μs      |   4.78   |
| fracture         |      929 μs      |      616 μs      |   1.51   |
| inside           |      161 μs      |      33 μs       |   4.88   |
| read_gds         |     2.68 ms      |      94 μs       |   28.5   |
| read_rawcells    |      363 μs      |     52.4 μs      |   6.94   |
| robustpath       |      171 μs      |     8.68 μs      |   19.7   |

Memory usage per object for 100000 objects:

| Object               |   Gdspy 1.6.13   |   Gdstk 0.9.41   | Reduction |
| :------------------- | :--------------: | :--------------: | :-------: |
| Rectangle            |      601 B       |      232 B       |    61%    |
| Circle (r = 10)      |     1.68 kB      |     1.27 kB      |    24%    |
| FlexPath segment     |     1.48 kB      |      439 B       |    71%    |
| FlexPath arc         |     2.26 kB      |     1.49 kB      |    34%    |
| RobustPath segment   |     2.89 kB      |      920 B       |    69%    |
| RobustPath arc       |     2.66 kB      |      920 B       |    66%    |
| Label                |      407 B       |      215 B       |    47%    |
| Reference            |      160 B       |      179 B       |    -12%   |
| Reference (array)    |      189 B       |      181 B       |     4%    |
| Cell                 |      430 B       |      229 B       |    47%    |
