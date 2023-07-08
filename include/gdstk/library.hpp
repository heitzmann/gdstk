/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_LIBRARY
#define GDSTK_HEADER_LIBRARY

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <time.h>

#include "array.hpp"
#include "cell.hpp"

namespace gdstk {

struct Library {
    // NULL-terminated string with library name.  The GDSII specification
    // allows only ASCII-encoded strings.  Gdstk does NOT enforce either rule.
    // The name is not used in OASIS files.
    char* name;

    // Unit and precision used to define the library geometry.  Please see the
    // discussion about these values in the accompanying HTML documentation or
    // at https://heitzmann.github.io/gdstk/gettingstarted.html
    double unit;
    double precision;

    // Cells should be added to (or removed from) the library using these
    // arrays.  Each cell must have a unique name within the library, but Gdstk
    // does NOT enforce it.
    Array<Cell*> cell_array;
    Array<RawCell*> rawcell_array;

    Property* properties;

    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void init(const char* name_, double unit_, double precision_) {
        name = copy_string(name_, NULL);
        unit = unit_;
        precision = precision_;
    }

    void print(bool all) const;

    void clear() {
        if (name) free_allocation(name);
        name = NULL;
        cell_array.clear();
        rawcell_array.clear();
        properties_clear(properties);
    }

    // Clear and free the memory of the whole library (this should be used with
    // caution, it assumes all the memory is allocated dynamically and frees
    // it).  It does not touch rawcells in rawcell_array, but it frees all cell
    // contents from cell_array.
    void free_all() {
        for (uint64_t i = 0; i < cell_array.count; i++) {
            cell_array[i]->free_all();
            free_allocation(cell_array[i]);
        }
        clear();
    }

    // This library instance must be zeroed before copy_from.
    // If deep_copy == true, new cells are allocated and deep copied from the
    // source.  Otherwise, the same cell pointers are used.
    void copy_from(const Library& library, bool deep_copy);

    // Append all polygons/paths or labels tags found in this library's cells
    // to result (rawcells are not included).
    void get_shape_tags(Set<Tag>& result) const;
    void get_label_tags(Set<Tag>& result) const;

    // Append the top level cells (and raw cells) to the respective arrays.
    // Top level cells are those that do not appear as dependencies of other
    // cells in the library.
    void top_level(Array<Cell*>& top_cells, Array<RawCell*>& top_rawcells) const;

    // Find cell or rawcell by name. Return NULL if not found.
    Cell* get_cell(const char* name) const;
    RawCell* get_rawcell(const char* name) const;

    // Rename a cell in the library, updating any references that use the old
    // name with the new one.  Note: these assume cell names are dynamically
    // allocated in cells and references throughout the library.
    void rename_cell(const char* old_name, const char* new_name);
    void rename_cell(Cell* cell, const char* new_name);

    // Replace a library cell updating references to the old cell with
    // references to the new one.  If old_cell is not present in the library,
    // new_cell is not inserted either, but references are updated anyway.
    void replace_cell(Cell* old_cell, Cell* new_cell);
    void replace_cell(RawCell* old_cell, Cell* new_cell);
    void replace_cell(Cell* old_cell, RawCell* new_cell);
    void replace_cell(RawCell* old_cell, RawCell* new_cell);

    // Change the tags of all elements in this library.  Map keys are the
    // current element tags and map values are the desired new tags.
    void remap_tags(const TagMap& map) {
        for (uint64_t i = 0; i < cell_array.count; i++) cell_array[i]->remap_tags(map);
    };

    // Output this library to a GDSII file.  All polygons are fractured to
    // max_points before saving (but the originals are kept) if max_points > 4.
    // GDSII files include a timestamp, which can be specified by the caller or
    // left NULL, in which case the current time will be used.
    ErrorCode write_gds(const char* filename, uint64_t max_points, tm* timestamp) const;

    // Output this library to an OASIS file.  The OASIS specification includes
    // support for a few special shapes, which can significantly decrease the
    // file size.  Circle detection is enabled by setting circle_tolerance > 0.
    // Rectangles and trapezoids are enabled via config_flags.  Further size
    // reduction can be achieved by setting deflate_level > 0 (up to 9, for
    // maximal compression).  Finally, config_flags is a bit-field value
    // obtained by or-ing OASIS_CONFIG_* constants, defined in oasis.h
    ErrorCode write_oas(const char* filename, double circle_tolerance, uint8_t deflate_level,
                        uint16_t config_flags);
};

// Struct used to get information from a library file without loading the
// complete library.
struct LibraryInfo {
    Array<char*> cell_names;
    Set<Tag> shape_tags;
    Set<Tag> label_tags;
    uint64_t num_polygons;
    uint64_t num_paths;
    uint64_t num_references;
    uint64_t num_labels;
    double unit;
    double precision;

    void clear() {
        for (uint64_t i = 0; i < cell_names.count; i++) {
            free_allocation(cell_names[i]);
            cell_names[i] = NULL;
        }
        cell_names.clear();
        shape_tags.clear();
        label_tags.clear();
        num_polygons = 0;
        num_paths = 0;
        num_references = 0;
        num_labels = 0;
        unit = 0;
        precision = 0;
    }
};

// Read the contents of a GDSII file into a new library.  If unit is not zero,
// the units in the file are converted (all elements are properly scaled to the
// desired unit).  The value of tolerance is used as the default tolerance for
// paths in the library.  If shape_tags is not empty, only shapes in those
// tags will be imported.  If not NULL, any errors will be reported through
// error_code.
Library read_gds(const char* filename, double unit, double tolerance, const Set<Tag>* shape_tags,
                 ErrorCode* error_code);

// Read the contents of an OASIS file into a new library.  If unit is not zero,
// the units in the file are converted (all elements are properly scaled to the
// desired unit).  The value of tolerance is used as the default tolerance for
// paths in the library and for the creation of circles.  If shape_tags is not
// empty, only shapes in those tags will be imported.  If not NULL, any errors
// will be reported through error_code.
Library read_oas(const char* filename, double unit,
                 double tolerance,  // TODO: const Set<Tag>* shape_tags,
                 ErrorCode* error_code);

// Read the unit and precision of a GDSII file and return in the respective
// arguments.
ErrorCode gds_units(const char* filename, double& unit, double& precision);

// Get/set the timestamps in the GDSII file.  If timestamp is not NULL, set the
// timestamps of the file.  The main library timestamp before any modification
// is returned.
tm gds_timestamp(const char* filename, const tm* new_timestamp, ErrorCode* error_code);

// Gather information about the GDSII file. Return argument info must be
// properly initialized.
ErrorCode gds_info(const char* filename, LibraryInfo& info);

// Read the precision of an OASIS file (unit is always 1e-6) and return in the
// precision argument.
ErrorCode oas_precision(const char* filename, double& precision);

// Return true if the file signature checks or if the file has no validation
// data.  If signature is provided, the calculated signature is stored there.
// If not NULL, any errors will be reported through error_code.  If the file
// has no checksum data, signature will be set to zero and error_code to
// ErrorCode::ChecksumError if they are not NULL.
bool oas_validate(const char* filename, uint32_t* signature, ErrorCode* error_code);

// TODO: Gather information about file
// ErrorCode oas_info(const char* filename, LibraryInfo& info);

}  // namespace gdstk

#endif
