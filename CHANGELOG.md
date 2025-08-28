# Changelog

## 0.9.61 - 2025-08-28
### Added
- Support for non-standard repetition vectors in GDSII (#293, #299, thanks svollenweider, WesYu).
### Fixed
- Bug in OASIS output when using explicit repetitions (#307, thanks RedFalsh).
- Time stamp format in GDSII (#308, thanks albachten).

## 0.9.60 - 2025-04-15
### Fixed
- Added support to 32-bit layers and datatypes.

## 0.9.59 - 2025-02-11
### Fixed
- Treat string properties as binary byte arrays in OASIS.

## 0.9.58 - 2024-11-25
### Changed
- Empty paths now give a warning when being converted to polygons or stored in GDSII/OASIS.
### Fixed
- Missing paths when vertices were separated exactly by the tolerance (#277)

## 0.9.57 - 2024-11-07
### Fixed
- Bug when removing GDSII properties (#276, thanks jatoben).

## 0.9.56 - 2024-10-28
### Added
- Support for Python 3.13.
### Fixed
- Copy Raith data in `Cell::get_flexpaths`.

## 0.9.55 - 2024-08-31
### Fixed
- Memory bug fix for Raith data

## 0.9.54 - 2024-08-31
### Changed
- Dropped unnecessary dependencies
### Fixed
- Trapezoid loading bug in OASIS format

## 0.9.53 - 2024-07-04
### Added
- Support for Raith MBMS path data (thanks Matthew Mckee).
- Support for numpy 2.0
### Changed
- Dropped support for python 3.8
### Fixed
- Qhull maximal number of points.

## 0.9.52 - 2024-04-18
### Fixed
- Infinite loop in `Cell::remap_tags` (#246, thanks dtzitkas!)
- Install headers when targeting the C++ library (#245)

## 0.9.51 - 2024-04-17
### Changed
- Use scikit-build-core for building, which enables support for Python 3.12 on Windows.

## 0.9.50 - 2024-02-07
### Added
- `Polygon.perimeter`.

## 0.9.49 - 2023-12-29
### Fixed
- Type annotation for `Cell.write_svg`.

## 0.9.48 - 2023-12-21
### Changed
- `Cell.dependencies` accepts keyword arguments.
### Fixed
- Fracturing polygons with few points is more robust.
- Compilation improvements.

## 0.9.45 - 2023-10-12
### Changed
- Use Qhull as an external dependecy instead of installing it ourselves to avoid conflicts.

## 0.9.43 - 2023-10-08
### Added
- `Library.remap` and `Cell.remap` to remap layer and data/text types
- Add typing stub
- Add deepcopy support
### Changed
- Raise an error if not both layer and datatype are specified in `Cell.get_polygons` and `Reference.get_polygons`.
- Correct ordering of path ends in `Library::read_oas()`
### Fixed
- Sort `slice` positions when converting from python because the internal implementation expects the coordinates to be sorted

## 0.9.42 - 2023-06-14
### Fixed
- `racetrack` bug in inner radius

## 0.9.41 - 2023-05-24
### Added
- Dictionary-like access to library cells by name
- `len(Library)` returns the number of cells in the library
- `Reference.cell_name` to directly access a referenced cell's name
### Fixed
- `RobustPath` accepts width of 0 at the path end

## 0.9.37 - 2023-02-12
### Changed
- Build system changes for lower numpy version requirements

## 0.9.36 - 2023-02-10
### Changed
- Downgrade the zlib version dependency to support manylinux2014
- Minor documentation improvements

## 0.9.35 - 2022-12-16
### Fixed
- Segfaults caused by class inheritance in Python
- Segfaults caused by cleanup of incomplete initialization of Python instances

## 0.9.1 - 2022-10-12
### Fixed
- Reading polygons with extremely large number of vertices
- Integer overflow in boolean operations
- `GdsWriter` C++ API fix
- Properly read zlib path from environment during build
- Ensure polygons are closed when loading GDSII files
- Reference counting in `Reference.apply_repetition` and `Cell.flatten`

## 0.9.0 - 2022-08-20
### Fixed
- Bug when saving OASIS files with missing references.
### Added
- `Reference.get_polygons`, `Reference.get_paths`, `Reference.get_labels`
- `Library.rename_cell`
- `Library::rename_cell` and `Library::replace_cell` in the C++ API
### Changed
- `Cell.filter` arguments modified to match `read_gds`.
- Changed default tolerance for `read_gds` and `read_oas` to be the library's rounding size.
- `Reference::polygons` renamed to `Reference::get_polygons` in the C++ API.
- `Reference::flexpaths` renamed to `Reference::get_flexpaths` in the C++ API.
- `Reference::robustpaths` renamed to `Reference::get_robustpaths` in the C++ API.
- `Reference::labels` renamed to `Reference::get_labels` in the C++ API.
- Removed magnification argument from `Reference::init` and `Label::init` in the C++ API.

## 0.8.3 - 2022-06-02
### Fixed
- References from raw cells are kept from garbage collection while in use
- Allow assigning to `Reference.x_reflection` from Python
- Errors in detection of arrays when exporting GDSII files

## 0.8.2 - 2022-02-26
### Fixed
- `Cell.get_paths` not returning all FlexPaths when RobustPaths were present in the cell (thanks @jatoben)
- Reference array is no longer transformed into multiple references when element displacement is zero
- Memory leaks in the python wrapper (thanks @jatoben)
### Changed
- Set separation instead of offset in `FlexPath::init` and `RobustPath::init`

## 0.8.1 - 2022-01-04
### Fixed
- Missing flag in OASIS bounding box property
- Bug in bounding box when using explicit repetitions
- Segfault when loading GDSII files with missing cells
### Changed
- GdsWriter C++ API
- Safer initializers C++ API

## 0.8.0 - 2021-10-08
### Added
- `Cell.get_polygons`, `Cell.get_paths`, and `Cell.get_labels` return a copy of a cell’s polygons, paths or labels, including references, with the possibility of filtering by layer and type
- `Library.layers_and_datatypes` and `Library.layers_and_texttypes` return tuples with the layer and data/text types found in the library.
- `gdstk.gds_info` provide information about a GDSII file without fully loading it
- Several `FlexPath` and `RobustPath` attributes.
### Fixed
- Label transforms in SVG output
- Label styling in SVG output
- Default label magnification when loading a GDSII file
- Bugs when loading some OASIS files
- Bug in OASIS output for some Manhattan geometry.
- Bug fix in Map::del
- Bounding box calculations take all repetitions into account
- Memory leaks
### Changed
- Removed LAPACK dependency
- The implementation of layer and data/text type for shapes and labels use the type `Tag` in the C++ API
- Style arguments renamed in `Cell.write_svg`

## 0.7.0 - 2021-08-02
### Added
- `contour` function
- `Polygon.transform` to apply a general transformation to the polygon vertices
- `Polygon.contain` tests whether single points are inside the polygon
- `Polygon.contain_all` and `Polygon.contain_any` test multiple points with short circuit
- `all_inside` and `any_inside` test multiple points against multiple polygons with short circuit
- Alternative function interfaces in the C++ API
### Fixed
- Holes in boolean results could lead to incorrect geometry in specific cases
- Bug in boolean operations resulting in self-intersecting polygons
- Bug in boolean operations with clockwise-oriented polygons
- Unsupported records found when loading a library generate a warning, not an error
### Changed
- `inside` has changed to use the better interfaces: grouping has been removed, scaling is not necessary, and short-circuit is implemented in `all_inside` and `any_inside`

## 0.6.1 - 2021-07-03
### Fixed
- Bug in `gdstk::read_oas` and `Library.read_oas`

## 0.6.0 - 2021-06-29
### Added
- `Library.replace`, used when adding cells with substitution of duplicate cell names
- Added pyproject.toml (thanks Ben Gollmer for the fix)
### Changed
- `Reference.cell` is now writable
### Fixed
- Bug in Array::insert not increasing the array count

## 0.5.0 - 2021-06-11
### Added
- Argument `precision` in `Cell.write_svg` controls the maximum number of digits of coordinates in the SVG
- Function `gds_timestamp` can be used to query or set the timestamp in a GDSII file
- Better error handling in the C++ API and argument validation for the Python wrapper
### Changed
- `oas_validate` returns `None` if the file has no checksum information
- `Library.write_gds` and `GdsWriter` accept a timestamp setting
### Fixed
- Bend calculation for `FlexPath` correctly accounts for bending angle to make sure the bend fits
- Missing files in the source distribution

## 0.4.0 - 2021-04-11
### Added
- `Cell.filter` to remove elements by layer and data/text type
- `Cell.convex_hull` and `Reference.convex_hull`
- `FlexPath.path_spines()` and `RobustPath.path_spines()`
- `Library.unit` and `Library.precision`
- Shapes can be sorted in `Cell.write_svg` (sorting works within each cell, references remain on top)
### Changed
- Bounding box calculations use convex hull for efficiency
- Bounding box and convex hull calculations cache intermediate results for efficiency
### Fixed
- `Robustpath.parametric` docstring.
- Accept `None` as a possible value for arguments in several `RobustPath` methods

## 0.3.2 - 2021-02-15
### Fixed
- Build system fixes for conda recipe

## 0.3.1 - 2021-02-12
### Fixed
- Missing constant definition

## 0.3.0 - 2021-02-12
### Added
- Support for OASIS files
- Repetition property for geometric objects, labels and references
- Library and cells can have properties
### Changed
- Use cmake to properly install library
- More efficient bounding box calculation for rotations multiple of 90°
- Labels are now included in bounding box calculations
- Properties can be general or GDSII-specific. Only the latter are stored in gds files.
- Attribute `gdsii_path` renamed to `simple_path` in `FlexPath` and `RobustPath`.

## 0.2.0 - 2020-11-23
### Added
- How-tos and C++ examples
- `Cell.add` and `Library.add` accept iterators
- `Map<T>.has_key`
- `Repetition` class
### Changed
- `RawCell` doesn't copy the contents of its input file unless needed
- Dependencies for `Cell` and `RawCell` are stored in maps, instead of arrays in the C++ API
- The `translate` method of polygons and paths accept a sequence of 2 coordinates or a complex
### Fixed
- Incorrect scaling for `FlexPath` when `scale_width = False`
- Typo in default SVG background specification
- Negative path extensions are correctly implemented

## 0.1.1 - 2020-11-13
### Fixed
- Add missing source files to MANIFEST.in
- Remove directory from CMakeLists.txt
- Remove unnecessary dependency from conda

## 0.1.0 - 2020-10-03
### Added
- Initial release

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
