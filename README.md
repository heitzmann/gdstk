# GDSTK 0.9.40

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

The library depends on [zlib](https://zlib.net/).

### Python wrapper

The Python module can be installed via pip, Conda or compiled directly from source.
It depends on:

* [zlib](https://zlib.net/)
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

The module must be linked aginst zlib.
The included CMakeLists.txt file can be used as a guide.

Installation from source should follow the usual method (there is no need to compile the static library beforehand):

```sh
python setup.py install
```

## Support

Help support Gdstk development by [donating via PayPal](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=JD2EUE2WPPBQQ) or [sponsoring me on GitHub](https://github.com/sponsors/heitzmann).


## Benchmarks

The _benchmarks_ directory contains a few tests to compare the performance gain of the Python interface versus Gdspy.
They are only for reference; the real improvement is heavily dependent on the type of layout and features used.
If maximal performance is important, the library should be used directly from C++, without the Python interface.

Timing results were obtained with Python 3.10 on an Intel Core i7-3820.
They represent the best average time to run each function out of 16 sets of 8 runs each.

| Benchmark        |   Gdspy 1.6.12   |   Gdstk 0.9.0    |   Gain   |
| :--------------- | :--------------: | :--------------: | :------: |
| 10k_rectangles   |     84.3 ms      |     5.03 ms      |   16.7   |
| 1k_circles       |      309 ms      |      233 ms      |   1.33   |
| boolean-offset   |      223 μs      |     43.4 μs      |   5.13   |
| bounding_box     |     35.4 ms      |      171 μs      |   207    |
| curves           |     1.66 ms      |     30.3 μs      |   54.8   |
| flatten          |      566 μs      |      8.7 μs      |   65.1   |
| flexpath         |     2.98 ms      |     15.8 μs      |   188    |
| flexpath-param   |     2.79 ms      |      626 μs      |   4.45   |
| fracture         |      965 μs      |      611 μs      |   1.58   |
| inside           |      162 μs      |     31.6 μs      |   5.14   |
| read_gds         |     3.04 ms      |      95 μs       |    32    |
| read_rawcells    |      445 μs      |     60.1 μs      |   7.4    |
| robustpath       |      202 μs      |     9.05 μs      |   22.3   |

Memory usage per object for 100000 objects:

| Object               |   Gdspy 1.6.12   |   Gdstk 0.9.0    | Reduction |
| :------------------- | :--------------: | :--------------: | :-------: |
| Rectangle            |      521 B       |      232 B       |    56%    |
| Circle (r = 10)      |     1.69 kB      |     1.27 kB      |    25%    |
| FlexPath segment     |      1.5 kB      |      439 B       |    71%    |
| FlexPath arc         |     2.27 kB      |     1.49 kB      |    34%    |
| RobustPath segment   |     2.87 kB      |      919 B       |    69%    |
| RobustPath arc       |     2.63 kB      |      919 B       |    66%    |
| Label                |      419 B       |      215 B       |    49%    |
| Reference            |      156 B       |      182 B       |    -16%   |
| Reference (array)    |      186 B       |      184 B       |     1%    |
| Cell                 |      437 B       |      231 B       |    47%    |
