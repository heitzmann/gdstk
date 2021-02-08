# GDSTK README

[![Boost Software License - Version 1.0](https://img.shields.io/github/license/heitzmann/gdstk.svg)](https://www.boost.org/LICENSE_1_0.txt)
[![Tests Runner](https://github.com/heitzmann/gdstk/workflows/Tests%20Runner/badge.svg)](https://github.com/heitzmann/gdstk/actions)
[![Publish Docs](https://github.com/heitzmann/gdstk/workflows/Publish%20Docs/badge.svg)](http://heitzmann.github.io/gdstk)
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


## Instalation

### C++ library only

The C++ library is meant to be used by including it in your own source code.

If you prefer to install a static library, the included _CMakeLists.txt_ should be a good starting option (use `-DCMAKE_INSTALL_PREFIX=path` to control the installation path):

```sh
cmake -S . -B build
cmake --build build --target install
```

The library depends on LAPACK and [zlib](https://zlib.net/).

### Python wrapper

The Python module can be installed via Conda (recommended) or compiled directly from source.
It depends on:

* LAPACK
* [zlib](https://zlib.net/)
* [CMake](https://cmake.org/)
* [Python](https://www.python.org/)
* [Numpy](https://numpy.org/)
* [Sphinx](https://www.sphinx-doc.org/) and [rtd theme](https://sphinx-rtd-theme.readthedocs.io/) (to build the [documentation](http://heitzmann.github.io/gdstk))

#### Conda

Windows users are suggested to install via [Conda](https://www.anaconda.com/) using the available [conda-forge recipe](https://github.com/conda-forge/gdstk-feedstock).
The recipe works on MacOS and Linux as well.

#### From source

The module must be linked aginst LAPACK and zlib.
The included CMakeLists.txt file can be used as a guide.

Installation from source should follow the usual method (there is no need to compile the static library beforehand):

```sh
python setup.py install
```

## Support

Help support Gdstk development by [donating via PayPal](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=JD2EUE2WPPBQQ)


## Benchmarks

The _benchmarks_ directory contains a few tests to compare the performance gain of the Python interface versus Gdspy.
They are only for reference; the real improvement is heavily dependent on the type of layout and features used.
If maximal performance is important, the library should be used directly from C++, without the Python interface.

Timing results were obtained with Python 3.8 on an Intel Core i7-3820 with 8 cores and 64 GB of RAM.
They represent the best average time to run each function out of 16 sets of 8 runs each.

| Benchmark        |   Gdspy 1.6.1    |   Gdstk 0.2.0    |   Gain   |
| :--------------- | :--------------: | :--------------: | :------: |
| 10k_rectangles   |      222 ms      |     6.67 ms      |   33.3   |
| 1k_circles       |      572 ms      |      363 ms      |   1.57   |
| boolean-offset   |      419 μs      |     60.7 μs      |   6.91   |
| bounding_box     |      634 μs      |     11.2 μs      |   56.8   |
| curves           |     3.02 ms      |     65.4 μs      |   46.1   |
| flatten          |     1.01 ms      |     12.6 μs      |   79.8   |
| flexpath         |     5.59 ms      |     25.2 μs      |   222    |
| flexpath-param   |     7.34 ms      |     1.45 ms      |   5.08   |
| fracture         |     1.71 ms      |      911 μs      |   1.88   |
| inside           |     41.2 μs      |     11.6 μs      |   3.55   |
| read_gds         |     7.03 ms      |      119 μs      |    59    |
| read_rawcells    |      609 μs      |     58.5 μs      |   10.4   |
| robustpath       |      386 μs      |     15.9 μs      |   24.3   |

Memory usage per object for 100.000 objects using Python 3.8:

| Object               |   Gdspy 1.6.1    |   Gdstk 0.2.0    | Reduction |
| :------------------- | :--------------: | :--------------: | :-------: |
| Rectangle            |      594 B       |      233 B       |    61%    |
| Circle (r = 10)      |     1.67 kB      |     1.27 kB      |    24%    |
| FlexPath segment     |     1.42 kB      |      441 B       |    70%    |
| FlexPath arc         |     2.23 kB      |     1.49 kB      |    33%    |
| RobustPath segment   |     2.78 kB      |      922 B       |    68%    |
| RobustPath arc       |     2.58 kB      |      919 B       |    65%    |
| Label                |      398 B       |      218 B       |    45%    |
| Reference            |      159 B       |      180 B       |    -13%   |
| Reference (array)    |      191 B       |      182 B       |     5%    |
| Cell                 |      435 B       |      211 B       |    52%    |
