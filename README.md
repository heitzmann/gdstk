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

Timing results were obtained with Python 3.9 on an Intel Core i7-3930K with 6 cores and 16 GB of RAM at 3.2 GHz.
They represent the best average time to run each function out of 16 sets of 8 runs each.

| Benchmark        |   Gdspy 1.6.3    |   Gdstk 0.3.1    |   Gain   |
| :--------------- | :--------------: | :--------------: | :------: |
| 10k_rectangles   |      264 ms      |     7.53 ms      |   35.1   |
| 1k_circles       |      641 ms      |      376 ms      |   1.71   |
| boolean-offset   |      444 μs      |     78.1 μs      |   5.69   |
| bounding_box     |      752 μs      |     14.8 μs      |   50.8   |
| curves           |     3.49 ms      |     65.1 μs      |   53.6   |
| flatten          |     1.19 ms      |     15.8 μs      |   75.2   |
| flexpath         |     6.44 ms      |     30.6 μs      |   211    |
| flexpath-param   |     7.48 ms      |     1.45 ms      |   5.14   |
| fracture         |     1.81 ms      |     1.07 ms      |   1.69   |
| inside           |     42.7 μs      |     13.9 μs      |   3.07   |
| read_gds         |     7.71 ms      |      143 μs      |    54    |
| read_rawcells    |      742 μs      |     70.9 μs      |   10.5   |
| robustpath       |      456 μs      |     19.3 μs      |   23.7   |

Memory usage per object for 100.000 objects using Python 3.9:

| Object               |   Gdspy 1.6.3    |   Gdstk 0.3.1    | Reduction |
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
