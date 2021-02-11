# Change Log

## Unreleased
### Added
- Support for OASIS files
- Repetition property for geometric objects, labels and references
- Library, cells and repetitions can have properties

### Changed
- Use cmake to properly intall library
- More efficient bounding box calculation for rotations multiple of 90°
- Labels are now included in bounding box calculations
- Properties can be general or GDSII-specific. Only the latter are stored in gds files.

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
- Remove unecessary dependency from conda

## 0.1.0 - 2020-10-03
### Added
- Initial release

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
