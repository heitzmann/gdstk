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


## Installation

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
* [Sphinx](https://www.sphinx-doc.org/), [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/), and [Sphinx Inline Tabs](https://sphinx-inline-tabs.readthedocs.io/) (to build the [documentation](http://heitzmann.github.io/gdstk))

#### Conda

Windows users are suggested to install via [Conda](https://www.anaconda.com/) using the available [conda-forge recipe](https://github.com/conda-forge/gdstk-feedstock).
The recipe works on MacOS and Linux as well.

To install in a new Conda environment:

```sh
# Create a new conda environment named gdstk
conda create -n gdstk -c conda-forge --strict-channel-priority
# Activate the new environment
conda activate gdstk
# Install gdstk
conda install gdstk
```

To use an existing environment, make sure it is configured to prioritize the conda-forge channel:

```sh
# Configure the conda-forge channel
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
# Install gdstk
conda install gdstk
```

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

| Benchmark        |   Gdspy 1.6.6    |   Gdstk 0.5.0    |   Gain   |
| :--------------- | :--------------: | :--------------: | :------: |
| 10k_rectangles   |      215 ms      |     7.07 ms      |   30.4   |
| 1k_circles       |      575 ms      |      363 ms      |   1.59   |
| boolean-offset   |      376 μs      |     64.8 μs      |   5.81   |
| bounding_box     |     83.7 ms      |      207 μs      |   403    |
| curves           |      3.1 ms      |     55.1 μs      |   56.2   |
| flatten          |     1.03 ms      |     13.9 μs      |    74    |
| flexpath         |     5.68 ms      |     25.8 μs      |   220    |
| flexpath-param   |     6.42 ms      |     1.25 ms      |   5.15   |
| fracture         |      1.5 ms      |      884 μs      |   1.7    |
| inside           |     35.2 μs      |     12.9 μs      |   2.74   |
| read_gds         |     6.45 ms      |      122 μs      |   52.9   |
| read_rawcells    |      638 μs      |     59.9 μs      |   10.7   |
| robustpath       |      405 μs      |     16.5 μs      |   24.6   |

Memory usage per object for 100000 objects.

| Object               |   Gdspy 1.6.6    |   Gdstk 0.5.0    | Reduction |
| :------------------- | :--------------: | :--------------: | :-------: |
| Rectangle            |      608 B       |      232 B       |    62%    |
| Circle (r = 10)      |     1.69 kB      |     1.28 kB      |    24%    |
| FlexPath segment     |      1.5 kB      |      439 B       |    71%    |
| FlexPath arc         |     2.28 kB      |     1.49 kB      |    35%    |
| RobustPath segment   |     2.86 kB      |      922 B       |    69%    |
| RobustPath arc       |     2.62 kB      |      922 B       |    66%    |
| Label                |      412 B       |      216 B       |    48%    |
| Reference            |      157 B       |      183 B       |    -16%   |
| Reference (array)    |      191 B       |      179 B       |     6%    |
| Cell                 |      437 B       |      227 B       |    48%    |
