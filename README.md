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

The library depends on [zlib](https://zlib.net/).

### Python wrapper

The Python module can be installed via Conda (recommended) or compiled directly from source.
It depends on:

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

The module must be linked aginst zlib.
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

Timing results were obtained with Python 3.10 on an Intel Core i7-3820.
They represent the best average time to run each function out of 16 sets of 8 runs each.

| Benchmark        |   Gdspy 1.6.10   |   Gdstk 0.8.1    |   Gain   |
| :--------------- | :--------------: | :--------------: | :------: |
| 10k_rectangles   |      202 ms      |      6.4 ms      |   31.5   |
| 1k_circles       |      516 ms      |      311 ms      |   1.66   |
| boolean-offset   |      365 μs      |     63.1 μs      |   5.79   |
| bounding_box     |     79.3 ms      |      229 μs      |   347    |
| curves           |     3.06 ms      |     56.3 μs      |   54.3   |
| flatten          |      963 μs      |     13.2 μs      |   72.9   |
| flexpath         |     5.69 ms      |     25.3 μs      |   225    |
| flexpath-param   |     6.56 ms      |     1.33 ms      |   4.92   |
| fracture         |     1.51 ms      |      834 μs      |   1.81   |
| inside           |      216 μs      |     40.6 μs      |   5.31   |
| read_gds         |     6.26 ms      |      120 μs      |   52.1   |
| read_rawcells    |      635 μs      |     63.1 μs      |   10.1   |
| robustpath       |      357 μs      |     16.4 μs      |   21.8   |

Memory usage per object for 100000 objects:

| Object               |   Gdspy 1.6.10   |   Gdstk 0.8.1    | Reduction |
| :------------------- | :--------------: | :--------------: | :-------: |
| Rectangle            |      461 B       |      31.5 B      |    93%    |
| Circle (r = 10)      |      775 B       |     1.27 kB      |    -68%   |
| FlexPath segment     |      1.5 kB      |      449 B       |    71%    |
| FlexPath arc         |     2.27 kB      |     1.49 kB      |    34%    |
| RobustPath segment   |     2.87 kB      |      920 B       |    69%    |
| RobustPath arc       |     2.63 kB      |      919 B       |    66%    |
| Label                |      418 B       |      216 B       |    48%    |
| Reference            |      157 B       |      180 B       |    -15%   |
| Reference (array)    |      191 B       |      179 B       |     6%    |
| Cell                 |      442 B       |      228 B       |    49%    |
