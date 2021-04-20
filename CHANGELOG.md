# Change Log

## Unreleased
### Fixed
- Bend calculation for `FlexPath` correctly accounts for bending angle to make sure the bend fits

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
