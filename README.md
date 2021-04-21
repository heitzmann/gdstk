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
* [Sphinx](https://www.sphinx-doc.org/) and [rtd theme](https://sphinx-rtd-theme.readthedocs.io/) (to build the [documentation](http://heitzmann.github.io/gdstk))

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

| Benchmark        |   Gdspy 1.6.3    |   Gdstk 0.4.0    |   Gain   |
| :--------------- | :--------------: | :--------------: | :------: |
| 10k_rectangles   |      273 ms      |     7.92 ms      |   34.5   |
| 1k_circles       |      695 ms      |      432 ms      |   1.61   |
| boolean-offset   |      454 μs      |     78.7 μs      |   5.77   |
| bounding_box     |      102 ms      |      248 μs      |   411    |
| curves           |     3.63 ms      |     64.6 μs      |   56.2   |
| flatten          |     1.22 ms      |     16.6 μs      |   73.6   |
| flexpath         |     6.78 ms      |     32.6 μs      |   208    |
| flexpath-param   |     7.81 ms      |     1.42 ms      |   5.51   |
| fracture         |     1.82 ms      |     1.06 ms      |   1.72   |
| inside           |     42.7 μs      |     13.8 μs      |   3.1    |
| read_gds         |     7.84 ms      |      143 μs      |   54.9   |
| read_rawcells    |      731 μs      |     70.4 μs      |   10.4   |
| robustpath       |      471 μs      |     19.4 μs      |   24.3   |

Memory usage per object for 100.000 objects using Python 3.9:

| Object               |   Gdspy 1.6.3    |   Gdstk 0.4.0    | Reduction |
| :------------------- | :--------------: | :--------------: | :-------: |
| Rectangle            |      611 B       |      232 B       |    62%    |
| Circle (r = 10)      |     1.69 kB      |     1.28 kB      |    24%    |
| FlexPath segment     |      1.5 kB      |      439 B       |    71%    |
| FlexPath arc         |     2.28 kB      |     1.49 kB      |    35%    |
| RobustPath segment   |     2.86 kB      |      922 B       |    69%    |
| RobustPath arc       |     2.62 kB      |      922 B       |    66%    |
| Label                |      412 B       |      216 B       |    48%    |
| Reference            |      157 B       |      183 B       |    -16%   |
| Reference (array)    |      191 B       |      179 B       |     6%    |
| Cell                 |      437 B       |      228 B       |    48%    |
