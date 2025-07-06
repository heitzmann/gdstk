/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <zlib.h>

#include <gdstk/allocator.hpp>
#include <gdstk/cell.hpp>
#include <gdstk/flexpath.hpp>
#include <gdstk/gdsii.hpp>
#include <gdstk/label.hpp>
#include <gdstk/library.hpp>
#include <gdstk/map.hpp>
#include <gdstk/oasis.hpp>
#include <gdstk/polygon.hpp>
#include <gdstk/rawcell.hpp>
#include <gdstk/reference.hpp>
#include <gdstk/utils.hpp>
#include <gdstk/vec.hpp>

namespace gdstk {

struct ByteArray {
    uint64_t count;
    uint8_t* bytes;
    Property* properties;
};  // namespace gdstk

void Library::print(bool all) const {
    printf("Library <%p> %s, unit %lg, precision %lg, %" PRIu64 " cells, %" PRIu64
           " raw cells, owner <%p>\n",
           this, name, unit, precision, cell_array.count, rawcell_array.count, owner);
    if (all) {
        printf("Cell array (count %" PRIu64 "/%" PRIu64 "):\n", cell_array.count,
               cell_array.capacity);
        for (uint64_t i = 0; i < cell_array.count; i++) {
            printf("Cell %" PRIu64 ": ", i);
            cell_array[i]->print(true);
        }
        printf("RawCell array (count %" PRIu64 "/%" PRIu64 "):\n", rawcell_array.count,
               rawcell_array.capacity);
        for (uint64_t i = 0; i < rawcell_array.count; i++) {
            printf("RawCell %" PRIu64 ": ", i);
            rawcell_array[i]->print(true);
        }
    }
    properties_print(properties);
}

void Library::copy_from(const Library& library, bool deep_copy) {
    name = copy_string(library.name, NULL);
    unit = library.unit;
    precision = library.precision;
    if (deep_copy) {
        cell_array.capacity = library.cell_array.capacity;
        cell_array.count = library.cell_array.count;
        cell_array.items = (Cell**)allocate(sizeof(Cell*) * cell_array.capacity);
        Cell** src = library.cell_array.items;
        Cell** dst = cell_array.items;
        for (uint64_t i = 0; i < library.cell_array.count; i++, src++, dst++) {
            *dst = (Cell*)allocate_clear(sizeof(Cell));
            (*dst)->copy_from(**src, NULL, true);
        }
    } else {
        cell_array.copy_from(library.cell_array);
    }
    // raw cells should be immutable, so there's no need to perform a deep copy
    rawcell_array.copy_from(library.rawcell_array);
}

void Library::get_shape_tags(Set<Tag>& result) const {
    for (uint64_t i = 0; i < cell_array.count; i++) {
        cell_array[i]->get_shape_tags(result);
    }
}

void Library::get_label_tags(Set<Tag>& result) const {
    for (uint64_t i = 0; i < cell_array.count; i++) {
        cell_array[i]->get_label_tags(result);
    }
}

void Library::top_level(Array<Cell*>& top_cells, Array<RawCell*>& top_rawcells) const {
    Map<Cell*> cell_deps = {};
    Map<RawCell*> rawcell_deps = {};
    cell_deps.resize(cell_array.count * 2);
    rawcell_deps.resize(rawcell_array.count * 2);

    Cell** c_item = cell_array.items;
    for (uint64_t i = 0; i < cell_array.count; i++, c_item++) {
        Cell* cell = *c_item;
        cell->get_dependencies(false, cell_deps);
        cell->get_raw_dependencies(false, rawcell_deps);
    }

    RawCell** r_item = rawcell_array.items;
    for (uint64_t i = 0; i < rawcell_array.count; i++) {
        (*r_item++)->get_dependencies(false, rawcell_deps);
    }

    c_item = cell_array.items;
    for (uint64_t i = 0; i < cell_array.count; i++) {
        Cell* cell = *c_item++;
        if (cell_deps.get(cell->name) != cell) top_cells.append(cell);
    }

    r_item = rawcell_array.items;
    for (uint64_t i = 0; i < rawcell_array.count; i++) {
        RawCell* rawcell = *r_item++;
        if (rawcell_deps.get(rawcell->name) != rawcell) top_rawcells.append(rawcell);
    }

    cell_deps.clear();
    rawcell_deps.clear();
}

Cell* Library::get_cell(const char* cell_name) const {
    Cell** p = cell_array.items;
    for (uint64_t i = cell_array.count; i > 0; i--) {
        Cell* cell = *p++;
        if (strcmp(cell->name, cell_name) == 0) return cell;
    }
    return NULL;
}

RawCell* Library::get_rawcell(const char* rawcell_name) const {
    RawCell** p = rawcell_array.items;
    for (uint64_t i = rawcell_array.count; i > 0; i--) {
        RawCell* rawcell = *p++;
        if (strcmp(rawcell->name, rawcell_name) == 0) return rawcell;
    }
    return NULL;
}

void Library::rename_cell(const char* old_name, const char* new_name) {
    Cell* cell = get_cell(old_name);
    if (cell) {
        rename_cell(cell, new_name);
    }
}

void Library::rename_cell(Cell* cell, const char* new_name) {
    const char* old_name = cell->name;
    uint64_t size = 1 + strlen(new_name);
    for (uint64_t i = 0; i < cell_array.count; ++i) {
        Array<Reference*> ref_array = cell_array[i]->reference_array;
        for (uint64_t j = 0; j < ref_array.count; ++j) {
            Reference* ref = ref_array[j];
            if (ref->type == ReferenceType::Name && strcmp(ref->name, old_name) == 0) {
                ref->name = (char*)reallocate(ref->name, size);
                memcpy(ref->name, new_name, size);
            }
        }
    }
    cell->name = (char*)reallocate(cell->name, size);
    memcpy(cell->name, new_name, size);
}

void Library::replace_cell(Cell* old_cell, Cell* new_cell) {
    uint64_t index = cell_array.index(old_cell);
    if (index < cell_array.count) {
        cell_array.items[index] = new_cell;
    }

    const char* old_name = old_cell->name;
    const char* new_name = new_cell->name;
    uint64_t size = 1 + strlen(new_name);
    bool rename = strcmp(old_name, new_name) != 0;
    for (uint64_t i = 0; i < cell_array.count; ++i) {
        Array<Reference*> ref_array = cell_array[i]->reference_array;
        for (uint64_t j = 0; j < ref_array.count; ++j) {
            Reference* ref = ref_array[j];
            switch (ref->type) {
                case ReferenceType::Cell:
                    if (ref->cell == old_cell) {
                        ref->cell = new_cell;
                    }
                    break;
                case ReferenceType::RawCell:
                    if (strcmp(ref->rawcell->name, old_name) == 0) {
                        ref->type = ReferenceType::Cell;
                        ref->cell = new_cell;
                    }
                    break;
                case ReferenceType::Name:
                    if (rename && (strcmp(ref->name, old_name) == 0)) {
                        ref->name = (char*)reallocate(ref->name, size);
                        memcpy(ref->name, new_name, size);
                    }
                    break;
            }
        }
    }
}

void Library::replace_cell(RawCell* old_cell, Cell* new_cell) {
    uint64_t index = rawcell_array.index(old_cell);
    if (index < rawcell_array.count) {
        rawcell_array.remove_unordered(index);
        cell_array.append(new_cell);
    }

    const char* old_name = old_cell->name;
    const char* new_name = new_cell->name;
    uint64_t size = 1 + strlen(new_name);
    bool rename = strcmp(old_name, new_name) != 0;
    for (uint64_t i = 0; i < cell_array.count; ++i) {
        Array<Reference*> ref_array = cell_array[i]->reference_array;
        for (uint64_t j = 0; j < ref_array.count; ++j) {
            Reference* ref = ref_array[j];
            switch (ref->type) {
                case ReferenceType::Cell:
                    if (strcmp(ref->cell->name, old_name) == 0) {
                        ref->cell = new_cell;
                    }
                    break;
                case ReferenceType::RawCell:
                    if (ref->rawcell == old_cell) {
                        ref->type = ReferenceType::Cell;
                        ref->cell = new_cell;
                    }
                    break;
                case ReferenceType::Name:
                    if (rename && (strcmp(ref->name, old_name) == 0)) {
                        ref->name = (char*)reallocate(ref->name, size);
                        memcpy(ref->name, new_name, size);
                    }
                    break;
            }
        }
    }
}

void Library::replace_cell(Cell* old_cell, RawCell* new_cell) {
    uint64_t index = cell_array.index(old_cell);
    if (index < cell_array.count) {
        cell_array.remove_unordered(index);
        rawcell_array.append(new_cell);
    }

    const char* old_name = old_cell->name;
    const char* new_name = new_cell->name;
    uint64_t size = 1 + strlen(new_name);
    bool rename = strcmp(old_name, new_name) != 0;
    for (uint64_t i = 0; i < cell_array.count; ++i) {
        Array<Reference*> ref_array = cell_array[i]->reference_array;
        for (uint64_t j = 0; j < ref_array.count; ++j) {
            Reference* ref = ref_array[j];
            switch (ref->type) {
                case ReferenceType::Cell:
                    if (ref->cell == old_cell) {
                        ref->type = ReferenceType::RawCell;
                        ref->rawcell = new_cell;
                    }
                    break;
                case ReferenceType::RawCell:
                    if (strcmp(ref->rawcell->name, old_name) == 0) {
                        ref->rawcell = new_cell;
                    }
                    break;
                case ReferenceType::Name:
                    if (rename && (strcmp(ref->name, old_name) == 0)) {
                        ref->name = (char*)reallocate(ref->name, size);
                        memcpy(ref->name, new_name, size);
                    }
                    break;
            }
        }
    }
}

void Library::replace_cell(RawCell* old_cell, RawCell* new_cell) {
    uint64_t index = rawcell_array.index(old_cell);
    if (index < rawcell_array.count) {
        rawcell_array.items[index] = new_cell;
    }

    const char* old_name = old_cell->name;
    const char* new_name = new_cell->name;
    uint64_t size = 1 + strlen(new_name);
    bool rename = strcmp(old_name, new_name) != 0;
    for (uint64_t i = 0; i < cell_array.count; ++i) {
        Array<Reference*> ref_array = cell_array[i]->reference_array;
        for (uint64_t j = 0; j < ref_array.count; ++j) {
            Reference* ref = ref_array[j];
            switch (ref->type) {
                case ReferenceType::Cell:
                    if (strcmp(ref->cell->name, old_name) == 0) {
                        ref->type = ReferenceType::RawCell;
                        ref->rawcell = new_cell;
                    }
                    break;
                case ReferenceType::RawCell:
                    if (ref->rawcell == old_cell) {
                        ref->rawcell = new_cell;
                    }
                    break;
                case ReferenceType::Name:
                    if (rename && (strcmp(ref->name, old_name) == 0)) {
                        ref->name = (char*)reallocate(ref->name, size);
                        memcpy(ref->name, new_name, size);
                    }
                    break;
            }
        }
    }
}

ErrorCode Library::write_gds(const char* filename, uint64_t max_points, tm* timestamp) const {
    ErrorCode error_code = ErrorCode::NoError;
    FILE* out = fopen(filename, "wb");
    if (out == NULL) {
        if (error_logger) fputs("[GDSTK] Unable to open GDSII file for output.\n", error_logger);
        return ErrorCode::OutputFileOpenError;
    }

    tm now = {};
    if (!timestamp) timestamp = get_now(now);

    uint64_t len = strlen(name);
    if (len % 2) len++;

    uint16_t buffer_start[] = {6,
                               0x0002,
                               0x0258,
                               28,
                               0x0102,
                               (uint16_t)(timestamp->tm_year),
                               (uint16_t)(timestamp->tm_mon + 1),
                               (uint16_t)timestamp->tm_mday,
                               (uint16_t)timestamp->tm_hour,
                               (uint16_t)timestamp->tm_min,
                               (uint16_t)timestamp->tm_sec,
                               (uint16_t)(timestamp->tm_year),
                               (uint16_t)(timestamp->tm_mon + 1),
                               (uint16_t)timestamp->tm_mday,
                               (uint16_t)timestamp->tm_hour,
                               (uint16_t)timestamp->tm_min,
                               (uint16_t)timestamp->tm_sec,
                               (uint16_t)(4 + len),
                               0x0206};
    big_endian_swap16(buffer_start, COUNT(buffer_start));
    fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);
    fwrite(name, 1, len, out);

    uint16_t buffer_units[] = {20, 0x0305};
    big_endian_swap16(buffer_units, COUNT(buffer_units));
    fwrite(buffer_units, sizeof(uint16_t), COUNT(buffer_units), out);
    uint64_t units[] = {gdsii_real_from_double(precision / unit),
                        gdsii_real_from_double(precision)};
    big_endian_swap64(units, COUNT(units));
    fwrite(units, sizeof(uint64_t), COUNT(units), out);

    double scaling = unit / precision;
    Cell** cell = cell_array.items;
    for (uint64_t i = 0; i < cell_array.count; i++, cell++) {
        ErrorCode err = (*cell)->to_gds(out, scaling, max_points, precision, timestamp);
        if (err != ErrorCode::NoError) error_code = err;
    }

    RawCell** rawcell = rawcell_array.items;
    for (uint64_t i = 0; i < rawcell_array.count; i++, rawcell++) {
        ErrorCode err = (*rawcell)->to_gds(out);
        if (err != ErrorCode::NoError) error_code = err;
    }

    uint16_t buffer_end[] = {4, 0x0400};
    big_endian_swap16(buffer_end, COUNT(buffer_end));
    fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);

    fclose(out);
    return error_code;
}

static uint64_t max_string_length(Property* property) {
    uint64_t result = 0;
    while (property) {
        uint64_t len = strlen(property->name);
        if (len > result) result = len;
        PropertyValue* value = property->value;
        while (value) {
            if (value->type == PropertyType::String) {
                len = value->count;
                if (len > result) result = len;
            }
            value = value->next;
        }
        property = property->next;
    }
    return result;
}

// zlib memory management
static void* zalloc(void*, uInt count, uInt size) { return allocate(count * size); }

static void zfree(void*, void* ptr) { free_allocation(ptr); }

ErrorCode Library::write_oas(const char* filename, double circle_tolerance,
                             uint8_t compression_level, uint16_t config_flags) {
    ErrorCode error_code = ErrorCode::NoError;
    const uint64_t c_size = cell_array.count;
    OasisState state = {};
    state.circle_tolerance = circle_tolerance;
    state.config_flags = config_flags;

    if (compression_level > 9) compression_level = 9;

    OasisStream out;
    out.file = fopen(filename, "wb");
    if (out.file == NULL) {
        if (error_logger) fputs("[GDSTK] Unable to open OASIS file for output.\n", error_logger);
        return ErrorCode::OutputFileOpenError;
    }
    out.data_size = 1024 * 1024;
    out.data = (uint8_t*)allocate(out.data_size);
    out.cursor = NULL;
    out.crc32 = state.config_flags & OASIS_CONFIG_INCLUDE_CRC32;
    out.checksum32 = state.config_flags & OASIS_CONFIG_INCLUDE_CHECKSUM32;
    if (out.crc32) {
        out.signature = crc32(0, NULL, 0);
    } else if (out.checksum32) {
        out.signature = 0;
    }
    out.error_code = ErrorCode::NoError;

    char header[] = {'%', 'S', 'E', 'M', 'I',  '-',  'O',
                     'A', 'S', 'I', 'S', '\r', '\n', (char)OasisRecord::START,
                     3,   '1', '.', '0'};
    oasis_write(header, 1, COUNT(header), out);

    state.scaling = unit / precision;
    oasis_write_real(out, 1e-6 / precision);
    oasis_putc(1, out);  // flag indicating that table-offsets will be stored in the END record

    if (state.config_flags & OASIS_CONFIG_PROPERTY_TOP_LEVEL) {
        remove_property(properties, s_top_level_property_name, true);
        Array<Cell*> top_cells = {};
        Array<RawCell*> top_rawcells = {};
        top_level(top_cells, top_rawcells);
        for (uint64_t i = 0; i < top_cells.count; i++) {
            set_property(properties, s_top_level_property_name, top_cells[i]->name, true);
        }
        top_cells.clear();
        top_rawcells.clear();
    }

    if (state.config_flags & OASIS_CONFIG_PROPERTY_BOUNDING_BOX) {
        remove_property(properties, s_bounding_box_available_property_name, true);
        set_property(properties, s_bounding_box_available_property_name, (uint64_t)2, true);
    }

    if (state.config_flags & OASIS_CONFIG_PROPERTY_MAX_COUNTS) {
        uint64_t string_max = strlen(s_max_uint_size_property_name);
        uint64_t polygon_max = 0;
        uint64_t path_max = 0;

        uint64_t len = max_string_length(properties);
        if (len > string_max) string_max = len;

        Cell** p_cell = cell_array.items;
        Array<Vec2> tmp_array = {};
        for (uint64_t i = 0; i < c_size; i++) {
            Cell* cell = *p_cell++;
            len = strlen(cell->name);
            if (len > string_max) string_max = len;
            len = max_string_length(cell->properties);
            if (len > string_max) string_max = len;

            Polygon** poly_p = cell->polygon_array.items;
            for (uint64_t j = cell->polygon_array.count; j > 0; j--) {
                Polygon* poly = *poly_p++;
                len = max_string_length(poly->properties);
                if (len > string_max) string_max = len;
                len = poly->point_array.count;
                if (len > polygon_max) polygon_max = len;
            }

            FlexPath** flexpath_p = cell->flexpath_array.items;
            for (uint64_t j = cell->flexpath_array.count; j > 0; j--) {
                FlexPath* path = *flexpath_p++;
                len = max_string_length(path->properties);
                if (len > string_max) string_max = len;
                if (path->simple_path) {
                    if (path->spine.point_array.count > 1) {
                        tmp_array.count = 0;
                        FlexPathElement* el = path->elements;
                        for (uint64_t ne = 0; ne < path->num_elements; ne++, el++) {
                            ErrorCode err = path->element_center(el, tmp_array);
                            if (err != ErrorCode::NoError) error_code = err;
                            len = tmp_array.count;
                            if (len > path_max) path_max = len;
                        }
                    }
                } else {
                    Array<Polygon*> array = {};
                    ErrorCode err = path->to_polygons(false, 0, array);
                    if (err != ErrorCode::NoError) error_code = err;
                    poly_p = array.items;
                    for (uint64_t k = array.count; k > 0; k--) {
                        Polygon* poly = *poly_p++;
                        len = poly->point_array.count;
                        if (len > polygon_max) polygon_max = len;
                        poly->clear();
                        free_allocation(poly);
                    }
                    array.clear();
                }
            }

            RobustPath** robustpath_p = cell->robustpath_array.items;
            for (uint64_t j = cell->robustpath_array.count; j > 0; j--) {
                RobustPath* path = *robustpath_p++;
                len = max_string_length(path->properties);
                if (len > string_max) string_max = len;
                if (path->simple_path) {
                    if (path->subpath_array.count > 0) {
                        tmp_array.count = 0;
                        RobustPathElement* el = path->elements;
                        for (uint64_t ne = 0; ne < path->num_elements; ne++, el++) {
                            ErrorCode err = path->element_center(el, tmp_array);
                            if (err != ErrorCode::NoError) error_code = err;
                            len = tmp_array.count;
                            if (len > path_max) path_max = len;
                        }
                    }
                } else {
                    Array<Polygon*> array = {};
                    ErrorCode err = path->to_polygons(false, 0, array);
                    if (err != ErrorCode::NoError) error_code = err;
                    poly_p = array.items;
                    for (uint64_t k = array.count; k > 0; k--) {
                        Polygon* poly = *poly_p++;
                        len = poly->point_array.count;
                        if (len > polygon_max) polygon_max = len;
                        poly->clear();
                        free_allocation(poly);
                    }
                    array.clear();
                }
            }

            Reference** ref_p = cell->reference_array.items;
            for (uint64_t j = cell->reference_array.count; j > 0; j--) {
                Reference* ref = *ref_p++;
                len = max_string_length(ref->properties);
                if (len > string_max) string_max = len;
            }

            Label** label_p = cell->label_array.items;
            for (uint64_t j = cell->label_array.count; j > 0; j--) {
                Label* label = *label_p++;
                len = strlen(label->text);
                if (len > string_max) string_max = len;
                len = max_string_length(label->properties);
                if (len > string_max) string_max = len;
            }
        }
        tmp_array.clear();

        remove_property(properties, s_max_int_size_property_name, true);
        set_property(properties, s_max_int_size_property_name, (uint64_t)sizeof(int64_t), true);

        remove_property(properties, s_max_uint_size_property_name, true);
        set_property(properties, s_max_uint_size_property_name, (uint64_t)sizeof(uint64_t), true);

        remove_property(properties, s_max_string_size_property_name, true);
        set_property(properties, s_max_string_size_property_name, string_max, true);

        remove_property(properties, s_max_polygon_property_name, true);
        set_property(properties, s_max_polygon_property_name, polygon_max, true);

        remove_property(properties, s_max_path_property_name, true);
        set_property(properties, s_max_path_property_name, path_max, true);
    }

    ErrorCode err = properties_to_oas(properties, out, state);
    if (err != ErrorCode::NoError) error_code = err;

    Map<uint64_t> cell_name_map = {};
    Map<uint64_t> cell_offset_map = {};
    Map<uint64_t> text_string_map = {};
    bool write_cell_offsets = state.config_flags & OASIS_CONFIG_PROPERTY_CELL_OFFSET;

    // Build cell name map. Other maps are built as the file is written.
    cell_name_map.resize((uint64_t)(2.0 + 10.0 / GDSTK_MAP_CAPACITY_THRESHOLD * c_size));
    if (write_cell_offsets) {
        cell_offset_map.resize((uint64_t)(2.0 + 10.0 / GDSTK_MAP_CAPACITY_THRESHOLD * c_size));
    }
    Cell** cell_p = cell_array.items;
    for (uint64_t i = 0; i < c_size; i++) {
        Cell* cell = *cell_p++;
        cell_name_map.set(cell->name, i);
    }

    cell_p = cell_array.items;
    for (uint64_t i = 0; i < c_size; i++) {
        Cell* cell = *cell_p++;
        if (write_cell_offsets) {
            cell_offset_map.set(cell->name, ftell(out.file));
        }
        oasis_putc((int)OasisRecord::CELL_REF_NUM, out);
        oasis_write_unsigned_integer(out, cell_name_map.get(cell->name));

        assert(cell_name_map.get(cell->name) == i);

        if (compression_level > 0) {
            out.cursor = out.data;
        }

        // TODO: Use modal variables
        // Cell contents
        Polygon** poly_p = cell->polygon_array.items;
        for (uint64_t j = cell->polygon_array.count; j > 0; j--) {
            err = (*poly_p++)->to_oas(out, state);
            if (err != ErrorCode::NoError) error_code = err;
        }

        FlexPath** flexpath_p = cell->flexpath_array.items;
        for (uint64_t j = cell->flexpath_array.count; j > 0; j--) {
            FlexPath* path = *flexpath_p++;
            if (path->simple_path) {
                err = path->to_oas(out, state);
                if (err != ErrorCode::NoError) error_code = err;
            } else {
                Array<Polygon*> array = {};
                err = path->to_polygons(false, 0, array);
                if (err != ErrorCode::NoError) error_code = err;
                poly_p = array.items;
                for (uint64_t k = array.count; k > 0; k--) {
                    Polygon* poly = *poly_p++;
                    err = poly->to_oas(out, state);
                    if (err != ErrorCode::NoError) error_code = err;
                    poly->clear();
                    free_allocation(poly);
                }
                array.clear();
            }
        }

        RobustPath** robustpath_p = cell->robustpath_array.items;
        for (uint64_t j = cell->robustpath_array.count; j > 0; j--) {
            RobustPath* path = *robustpath_p++;
            if (path->simple_path) {
                err = path->to_oas(out, state);
                if (err != ErrorCode::NoError) error_code = err;
            } else {
                Array<Polygon*> array = {};
                err = path->to_polygons(false, 0, array);
                if (err != ErrorCode::NoError) error_code = err;
                poly_p = array.items;
                for (uint64_t k = array.count; k > 0; k--) {
                    Polygon* poly = *poly_p++;
                    err = poly->to_oas(out, state);
                    if (err != ErrorCode::NoError) error_code = err;
                    poly->clear();
                    free_allocation(poly);
                }
                array.clear();
            }
        }

        Reference** ref_p = cell->reference_array.items;
        for (uint64_t j = cell->reference_array.count; j > 0; j--) {
            Reference* ref = *ref_p++;
            if (ref->type == ReferenceType::RawCell) {
                if (error_logger)
                    fputs("[GDSTK] Reference to a RawCell cannot be used in an OASIS file.\n",
                          error_logger);
                error_code = ErrorCode::MissingReference;
                continue;
            }
            const char* name_ = (ref->type == ReferenceType::Cell) ? ref->cell->name : ref->name;
            bool reference_exists = cell_name_map.has_key(name_);
            uint8_t info = reference_exists ? 0xF0 : 0xB0;
            bool has_repetition = ref->repetition.get_count() > 1;
            if (has_repetition) info |= 0x08;
            if (ref->x_reflection) info |= 0x01;
            int64_t m;
            if (ref->magnification == 1.0 && is_multiple_of_pi_over_2(ref->rotation, m)) {
                if (m < 0) {
                    info |= ((uint8_t)(0x03 & ((m % 4) + 4))) << 1;
                } else {
                    info |= ((uint8_t)(0x03 & (m % 4))) << 1;
                }
                oasis_putc((int)OasisRecord::PLACEMENT, out);
                oasis_putc(info, out);
                if (reference_exists) {
                    uint64_t index = cell_name_map.get(name_);
                    oasis_write_unsigned_integer(out, index);
                } else {
                    uint64_t len = strlen(name_);
                    oasis_write_unsigned_integer(out, len);
                    oasis_write(ref->name, 1, len, out);
                }
            } else {
                if (ref->magnification != 1) info |= 0x04;
                if (ref->rotation != 0) info |= 0x02;
                oasis_putc((int)OasisRecord::PLACEMENT_TRANSFORM, out);
                oasis_putc(info, out);
                if (reference_exists) {
                    uint64_t index = cell_name_map.get(name_);
                    oasis_write_unsigned_integer(out, index);
                } else {
                    uint64_t len = strlen(name_);
                    oasis_write_unsigned_integer(out, len);
                    oasis_write(ref->name, 1, len, out);
                }
                if (ref->magnification != 1) {
                    oasis_write_real(out, ref->magnification);
                }
                if (ref->rotation != 0) {
                    oasis_write_real(out, ref->rotation * (180.0 / M_PI));
                }
            }
            oasis_write_integer(out, (int64_t)llround(ref->origin.x * state.scaling));
            oasis_write_integer(out, (int64_t)llround(ref->origin.y * state.scaling));
            if (has_repetition) oasis_write_repetition(out, ref->repetition, state.scaling);
            err = properties_to_oas(ref->properties, out, state);
            if (err != ErrorCode::NoError) error_code = err;
        }

        Label** label_p = cell->label_array.items;
        for (uint64_t j = cell->label_array.count; j > 0; j--) {
            Label* label = *label_p++;
            uint8_t info = 0x7B;
            bool has_repetition = label->repetition.get_count() > 1;
            if (has_repetition) info |= 0x04;
            oasis_putc((int)OasisRecord::TEXT, out);
            oasis_putc(info, out);
            uint64_t index;
            if (text_string_map.has_key(label->text)) {
                index = text_string_map.get(label->text);
            } else {
                index = text_string_map.count;
                text_string_map.set(label->text, index);
            }
            oasis_write_unsigned_integer(out, index);
            oasis_write_unsigned_integer(out, get_layer(label->tag));
            oasis_write_unsigned_integer(out, get_type(label->tag));
            oasis_write_integer(out, (int64_t)llround(label->origin.x * state.scaling));
            oasis_write_integer(out, (int64_t)llround(label->origin.y * state.scaling));
            if (has_repetition) oasis_write_repetition(out, label->repetition, state.scaling);
            err = properties_to_oas(label->properties, out, state);
            if (err != ErrorCode::NoError) error_code = err;
        }

        if (compression_level > 0) {
            uint64_t uncompressed_size = out.cursor - out.data;
            out.cursor = NULL;

            // Skip empty cells
            if (uncompressed_size > 0) {
                z_stream s = {};
                s.zalloc = zalloc;
                s.zfree = zfree;
                if (deflateInit2(&s, compression_level, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) !=
                    Z_OK) {
                    if (error_logger) fputs("[GDSTK] Unable to initialize zlib.\n", error_logger);
                    error_code = ErrorCode::ZlibError;
                }
                s.avail_out = deflateBound(&s, (uLong)uncompressed_size);
                uint8_t* buffer = (uint8_t*)allocate(s.avail_out);
                s.next_out = buffer;
                s.avail_in = (uInt)uncompressed_size;
                s.next_in = out.data;
                int ret = deflate(&s, Z_FINISH);
                if (ret != Z_STREAM_END) {
                    if (error_logger) fputs("[GDSTK] Unable to compress CBLOCK.\n", error_logger);
                    error_code = ErrorCode::ZlibError;
                }

                oasis_putc((int)OasisRecord::CBLOCK, out);
                oasis_putc(0, out);
                oasis_write_unsigned_integer(out, uncompressed_size);
                oasis_write_unsigned_integer(out, s.total_out);
                oasis_write(buffer, 1, s.total_out, out);
                free_allocation(buffer);
                deflateEnd(&s);
            }
        }
    }

    uint64_t cell_name_offset = c_size > 0 ? ftell(out.file) : 0;
    cell_p = cell_array.items;
    Map<GeometryInfo> cache = {};
    for (uint64_t i = 0; i < c_size; i++) {
        Cell* cell = *cell_p++;
        oasis_putc((int)OasisRecord::CELLNAME_IMPLICIT, out);
        char* name_ = cell->name;
        uint64_t len = strlen(name_);
        oasis_write_unsigned_integer(out, len);
        oasis_write(name_, 1, len, out);

        if (state.config_flags & OASIS_CONFIG_PROPERTY_BOUNDING_BOX) {
            Vec2 bbmin, bbmax;
            GeometryInfo info = cell->bounding_box(cache);
            if (info.bounding_box_min.x > info.bounding_box_max.x) {
                bbmin = Vec2{0, 0};
                bbmax = Vec2{0, 0};
            } else {
                bbmin = info.bounding_box_min;
                bbmax = info.bounding_box_max;
            }
            int64_t xmin = llround(bbmin.x * state.scaling);
            int64_t ymin = llround(bbmin.y * state.scaling);
            uint64_t width = llround(bbmax.x * state.scaling) - xmin;
            uint64_t height = llround(bbmax.y * state.scaling) - ymin;
            remove_property(cell->properties, s_bounding_box_property_name, true);
            set_property(cell->properties, s_bounding_box_property_name, height, true);
            set_property(cell->properties, s_bounding_box_property_name, width, false);
            set_property(cell->properties, s_bounding_box_property_name, ymin, false);
            set_property(cell->properties, s_bounding_box_property_name, xmin, false);
            set_property(cell->properties, s_bounding_box_property_name, (uint64_t)0, false);
        }
        if (write_cell_offsets) {
            remove_property(cell->properties, s_cell_offset_property_name, true);
            set_property(cell->properties, s_cell_offset_property_name,
                         cell_offset_map.get(cell->name), true);
        }
        err = properties_to_oas(cell->properties, out, state);
        if (err != ErrorCode::NoError) error_code = err;
    }
    for (MapItem<GeometryInfo>* item = cache.next(NULL); item; item = cache.next(item)) {
        item->value.clear();
    }
    cache.clear();

    uint64_t text_string_offset = text_string_map.count > 0 ? ftell(out.file) : 0;
    for (MapItem<uint64_t>* item = text_string_map.next(NULL); item;
         item = text_string_map.next(item)) {
        oasis_putc((int)OasisRecord::TEXTSTRING, out);
        uint64_t len = strlen(item->key);
        oasis_write_unsigned_integer(out, len);
        oasis_write(item->key, 1, len, out);
        oasis_write_unsigned_integer(out, item->value);
    }

    uint64_t prop_name_offset = state.property_name_map.count > 0 ? ftell(out.file) : 0;
    for (MapItem<uint64_t>* item = state.property_name_map.next(NULL); item;
         item = state.property_name_map.next(item)) {
        oasis_putc((int)OasisRecord::PROPNAME, out);
        uint64_t len = strlen(item->key);
        oasis_write_unsigned_integer(out, len);
        oasis_write(item->key, 1, len, out);
        oasis_write_unsigned_integer(out, item->value);
    }

    uint64_t prop_string_offset = state.property_value_array.count > 0 ? ftell(out.file) : 0;
    PropertyValue** value_p = state.property_value_array.items;
    for (uint64_t i = state.property_value_array.count; i > 0; i--) {
        PropertyValue* value = *value_p++;
        oasis_putc((int)OasisRecord::PROPSTRING_IMPLICIT, out);
        oasis_write_unsigned_integer(out, value->count);
        oasis_write(value->bytes, 1, value->count, out);
    }

    oasis_putc((int)OasisRecord::END, out);

    // END (1) + table-offsets (?) + b-string length (2) + padding + validation (1 or 5) = 256
    uint64_t pad_len = 256 - 1 - 2 - 1 + ftell(out.file);
    if (out.crc32 || out.checksum32) pad_len -= 4;

    // Table offsets
    oasis_putc(1, out);
    oasis_write_unsigned_integer(out, cell_name_offset);
    oasis_putc(1, out);
    oasis_write_unsigned_integer(out, text_string_offset);
    oasis_putc(1, out);
    oasis_write_unsigned_integer(out, prop_name_offset);
    oasis_putc(1, out);
    oasis_write_unsigned_integer(out, prop_string_offset);
    oasis_putc(1, out);
    oasis_putc(0, out);  // LAYERNAME table
    oasis_putc(1, out);
    oasis_putc(0, out);  // XNAME table

    pad_len -= ftell(out.file);
    oasis_write_unsigned_integer(out, pad_len);
    for (; pad_len > 0; pad_len--) oasis_putc(0, out);

    if (out.crc32) {
        oasis_putc(1, out);
        little_endian_swap32(&out.signature, 1);
        fwrite(&out.signature, 4, 1, out.file);
    } else if (out.checksum32) {
        oasis_putc(2, out);
        little_endian_swap32(&out.signature, 1);
        fwrite(&out.signature, 4, 1, out.file);
    } else {
        oasis_putc(0, out);
    }

    fclose(out.file);
    free_allocation(out.data);

    cell_name_map.clear();
    cell_offset_map.clear();
    text_string_map.clear();
    state.property_name_map.clear();
    state.property_value_array.clear();
    return error_code;
}

Library read_gds(const char* filename, double unit, double tolerance, const Set<Tag>* shape_tags,
                 ErrorCode* error_code) {
    const char* gdsii_record_names[] = {
        "HEADER",    "BGNLIB",   "LIBNAME",   "UNITS",      "ENDLIB",      "BGNSTR",
        "STRNAME",   "ENDSTR",   "BOUNDARY",  "PATH",       "SREF",        "AREF",
        "TEXT",      "LAYER",    "DATATYPE",  "WIDTH",      "XY",          "ENDEL",
        "SNAME",     "COLROW",   "TEXTNODE",  "NODE",       "TEXTTYPE",    "PRESENTATION",
        "SPACING",   "STRING",   "STRANS",    "MAG",        "ANGLE",       "UINTEGER",
        "USTRING",   "REFLIBS",  "FONTS",     "PATHTYPE",   "GENERATIONS", "ATTRTABLE",
        "STYPTABLE", "STRTYPE",  "ELFLAGS",   "ELKEY",      "LINKTYPE",    "LINKKEYS",
        "NODETYPE",  "PROPATTR", "PROPVALUE", "BOX",        "BOXTYPE",     "PLEX",
        "BGNEXTN",   "ENDEXTN",  "TAPENUM",   "TAPECODE",   "STRCLASS",    "RESERVED",
        "FORMAT",    "MASK",     "ENDMASKS",  "LIBDIRSIZE", "SRFNAME",     "LIBSECUR"};

    Library library = {};
    // One extra char in case we need a 0-terminated string with max count (should never happen, but
    // it doesn't hurt to be prepared).
    uint8_t buffer[65537];
    uint16_t* udata16 = (uint16_t*)(buffer + 4);
    int16_t* data16 = (int16_t*)(buffer + 4);
    int32_t* data32 = (int32_t*)(buffer + 4);
    uint32_t* udata32 = (uint32_t*)(buffer + 4);
    uint64_t* data64 = (uint64_t*)(buffer + 4);
    char* str = (char*)(buffer + 4);

    Cell* cell = NULL;
    Polygon* polygon = NULL;
    FlexPath* path = NULL;
    Reference* reference = NULL;
    Label* label = NULL;

    double factor = 1;
    double width = 0;
    int16_t key = 0;

    FILE* in = fopen(filename, "rb");
    if (in == NULL) {
        fputs("[GDSTK] Unable to open GDSII file for input.\n", stderr);
        if (error_code) *error_code = ErrorCode::InputFileOpenError;
        return library;
    }

    while (true) {
        uint64_t record_length = COUNT(buffer);
        ErrorCode err = gdsii_read_record(in, buffer, record_length);
        if (err != ErrorCode::NoError) {
            if (error_code) *error_code = err;
            break;
        }

        // printf("0x%02X %s (%" PRIu64 " bytes)", buffer[2],
        //        buffer[2] < COUNT(gdsii_record_names) ? gdsii_record_names[buffer[2]] : "",
        //        record_length);

        uint64_t data_length;
        GdsiiDataType data_type = (GdsiiDataType)buffer[3];
        switch (data_type) {
            case GdsiiDataType::BitArray:
            case GdsiiDataType::TwoByteSignedInteger:
                data_length = (record_length - 4) / 2;
                big_endian_swap16((uint16_t*)data16, data_length);
                // for (uint32_t i = 0; i < data_length; i++) printf(" %" PRId16, data16[i]);
                break;
            case GdsiiDataType::FourByteSignedInteger:
            case GdsiiDataType::FourByteReal:
                data_length = (record_length - 4) / 4;
                big_endian_swap32((uint32_t*)data32, data_length);
                // for (uint32_t i = 0; i < data_length; i++) printf(" %" PRId32, data32[i]);
                break;
            case GdsiiDataType::EightByteReal:
                data_length = (record_length - 4) / 8;
                big_endian_swap64(data64, data_length);
                // for (uint32_t i = 0; i < data_length; i++)
                // printf(" %" PRIu64 " (%g)", data64[i], gdsii_real_to_double(data64[i]));
                break;
            default:
                data_length = record_length - 4;
                // for (uint32_t i = 0; i < data_length; i++) printf(" %c", str[i]);
        }

        // putchar('\n');

        switch ((GdsiiRecord)(buffer[2])) {
            case GdsiiRecord::HEADER:
            case GdsiiRecord::BGNLIB:
            case GdsiiRecord::ENDSTR:
                break;
            case GdsiiRecord::LIBNAME:
                if (str[data_length - 1] == 0) data_length--;
                library.name = (char*)allocate(data_length + 1);
                memcpy(library.name, str, data_length);
                library.name[data_length] = 0;
                break;
            case GdsiiRecord::UNITS: {
                const double db_in_user = gdsii_real_to_double(data64[0]);
                const double db_in_meters = gdsii_real_to_double(data64[1]);
                if (unit > 0) {
                    factor = db_in_meters / unit;
                    library.unit = unit;
                } else {
                    factor = db_in_user;
                    library.unit = db_in_meters / db_in_user;
                }
                library.precision = db_in_meters;
                if (tolerance <= 0) {
                    tolerance = library.precision / library.unit;
                }
            } break;
            case GdsiiRecord::ENDLIB: {
                Map<Cell*> map = {};
                uint64_t c_size = library.cell_array.count;
                map.resize((uint64_t)(2.0 + 10.0 / GDSTK_MAP_CAPACITY_THRESHOLD * c_size));
                Cell** c_item = library.cell_array.items;
                for (uint64_t i = c_size; i > 0; i--, c_item++) map.set((*c_item)->name, *c_item);
                c_item = library.cell_array.items;
                for (uint64_t i = c_size; i > 0; i--) {
                    cell = *c_item++;
                    Reference** ref = cell->reference_array.items;
                    for (uint64_t j = cell->reference_array.count; j > 0; j--) {
                        reference = *ref++;
                        Cell* cp = map.get(reference->name);
                        if (cp) {
                            free_allocation(reference->name);
                            reference->type = ReferenceType::Cell;
                            reference->cell = cp;
                        } else {
                            if (error_code) *error_code = ErrorCode::MissingReference;
                            if (error_logger)
                                fprintf(error_logger, "[GDSTK] Missing referenced cell %s\n",
                                        reference->name);
                        }
                    }
                }
                map.clear();
                fclose(in);
                return library;
            } break;
            case GdsiiRecord::BGNSTR:
                cell = (Cell*)allocate_clear(sizeof(Cell));
                break;
            case GdsiiRecord::STRNAME:
                if (cell) {
                    if (str[data_length - 1] == 0) data_length--;
                    cell->name = (char*)allocate(data_length + 1);
                    memcpy(cell->name, str, data_length);
                    cell->name[data_length] = 0;
                    library.cell_array.append(cell);
                }
                break;
            case GdsiiRecord::BOUNDARY:
            case GdsiiRecord::BOX:
                polygon = (Polygon*)allocate_clear(sizeof(Polygon));
                if (cell) cell->polygon_array.append(polygon);
                break;
            case GdsiiRecord::PATH:
            case GdsiiRecord::RAITHMBMSPATH:
                path = (FlexPath*)allocate_clear(sizeof(FlexPath));
                path->num_elements = 1;
                path->elements = (FlexPathElement*)allocate_clear(sizeof(FlexPathElement));
                path->simple_path = true;
                if (cell) cell->flexpath_array.append(path);
                break;
            case GdsiiRecord::RAITHPXXDATA:
                if (path) {
                    PXXData pxxdata;
                    memcpy(&pxxdata, buffer + 4, record_length);
                    path->raith_data.from_pxxdata(pxxdata);
                }
                break;
            case GdsiiRecord::SREF:
            case GdsiiRecord::AREF:
                reference = (Reference*)allocate_clear(sizeof(Reference));
                reference->magnification = 1;
                if (cell) cell->reference_array.append(reference);
                break;
            case GdsiiRecord::TEXT:
                label = (Label*)allocate_clear(sizeof(Label));
                label->magnification = 1;
                if (cell) cell->label_array.append(label);
                break;
            case GdsiiRecord::LAYER:
                if (polygon) {
                    if (data_type == GdsiiDataType::FourByteSignedInteger) {
                        set_layer(polygon->tag, udata32[0]);
                    } else {
                        set_layer(polygon->tag, udata16[0]);
                    }
                } else if (path) {
                    if (data_type == GdsiiDataType::FourByteSignedInteger) {
                        set_layer(path->elements[0].tag, udata32[0]);
                    } else {
                        set_layer(path->elements[0].tag, udata16[0]);
                    }
                } else if (label) {
                    if (data_type == GdsiiDataType::FourByteSignedInteger) {
                        set_layer(label->tag, udata32[0]);
                    } else {
                        set_layer(label->tag, udata16[0]);
                    }
                }
                break;
            case GdsiiRecord::DATATYPE:
            case GdsiiRecord::BOXTYPE:
                if (polygon) {
                    if (data_type == GdsiiDataType::FourByteSignedInteger) {
                        set_type(polygon->tag, udata32[0]);
                    } else {
                        set_type(polygon->tag, udata16[0]);
                    }
                } else if (path) {
                    if (data_type == GdsiiDataType::FourByteSignedInteger) {
                        set_type(path->elements[0].tag, udata32[0]);
                    } else {
                        set_type(path->elements[0].tag, udata16[0]);
                    }
                }
                break;
            case GdsiiRecord::WIDTH:
                if (data32[0] < 0) {
                    width = factor * -data32[0];
                    if (path) path->scale_width = false;
                } else {
                    width = factor * data32[0];
                    if (path) path->scale_width = true;
                }
                break;
            case GdsiiRecord::XY:
                if (polygon) {
                    polygon->point_array.ensure_slots(data_length / 2);
                    double* d = (double*)(polygon->point_array.items + polygon->point_array.count);
                    int32_t* s = data32;
                    for (uint64_t i = data_length; i > 0; i--) *d++ = factor * (*s++);
                    polygon->point_array.count += data_length / 2;
                } else if (path) {
                    Array<Vec2> point_array = {};
                    if (path->spine.point_array.count == 0) {
                        path->spine.tolerance = tolerance;
                        path->spine.append(Vec2{factor * data32[0], factor * data32[1]});
                        path->elements[0].half_width_and_offset.append(Vec2{width / 2, 0});
                        point_array.ensure_slots(data_length / 2 - 1);
                        double* d = (double*)point_array.items;
                        int32_t* s = data32 + 2;
                        for (uint64_t i = data_length - 2; i > 0; i--) *d++ = factor * (*s++);
                        point_array.count += data_length / 2 - 1;
                    } else {
                        point_array.ensure_slots(data_length / 2);
                        double* d = (double*)point_array.items;
                        int32_t* s = data32;
                        for (uint64_t i = data_length; i > 0; i--) *d++ = factor * (*s++);
                        point_array.count += data_length / 2;
                    }
                    path->segment(point_array, NULL, NULL, false);
                    point_array.clear();
                } else if (reference) {
                    Vec2 origin = Vec2{factor * data32[0], factor * data32[1]};
                    reference->origin = origin;
                    if (reference->repetition.type != RepetitionType::None) {
                        Repetition* repetition = &reference->repetition;
                        if (reference->rotation == 0 && !reference->x_reflection &&
                                data32[3] == 0  && data32[4] == 0) {
                            repetition->spacing.x =
                                (factor * data32[2] - origin.x) / repetition->columns;
                            repetition->spacing.y =
                                (factor * data32[5] - origin.y) / repetition->rows;
                        } else {
                            repetition->type = RepetitionType::Regular;
                            repetition->v1.x =
                                (factor * data32[2] - origin.x) / repetition->columns;
                            repetition->v1.y =
                                (factor * data32[3] - origin.y) / repetition->columns;
                            repetition->v2.x = (factor * data32[4] - origin.x) / repetition->rows;
                            repetition->v2.y = (factor * data32[5] - origin.y) / repetition->rows;
                        }
                    }
                } else if (label) {
                    label->origin.x = factor * data32[0];
                    label->origin.y = factor * data32[1];
                }
                break;
            case GdsiiRecord::ENDEL:
                if (polygon) {
                    // Polygons are closed in GDSII (first and last points are the same)
                    Array<Vec2>& pa = polygon->point_array;
                    if (pa[0] == pa[pa.count - 1]) pa.count--;
                    if (shape_tags && !shape_tags->has_value(polygon->tag) && cell) {
                        Array<Polygon*>* array = &cell->polygon_array;
                        uint64_t index = array->count - 1;
                        if (array->items[index] == polygon) {
                            array->remove_unordered(index);
                        } else {
                            array->remove_item(polygon);
                        }
                        polygon->clear();
                        free_allocation(polygon);
                    }
                } else if (path) {
                    if (shape_tags && !shape_tags->has_value(path->elements[0].tag) && cell) {
                        Array<FlexPath*>* array = &cell->flexpath_array;
                        uint64_t index = array->count - 1;
                        if (array->items[index] == path) {
                            array->remove_unordered(index);
                        } else {
                            array->remove_item(path);
                        }
                        path->clear();
                        free_allocation(path);
                    }
                }
                polygon = NULL;
                path = NULL;
                reference = NULL;
                label = NULL;
                break;
            case GdsiiRecord::SNAME:
                if (reference) {
                    if (str[data_length - 1] == 0) data_length--;
                    reference->name = (char*)allocate(data_length + 1);
                    memcpy(reference->name, str, data_length);
                    reference->name[data_length] = 0;
                    reference->type = ReferenceType::Name;
                } else if (path) {
                    if (str[data_length - 1] == 0) data_length--;
                    path->raith_data.base_cell_name = (char*)allocate(data_length + 1);
                    memcpy(path->raith_data.base_cell_name, str, data_length);
                    path->raith_data.base_cell_name[data_length] = 0;
                }
                break;
            case GdsiiRecord::COLROW:
                if (reference) {
                    Repetition* repetition = &reference->repetition;
                    repetition->type = RepetitionType::Rectangular;
                    repetition->columns = data16[0];
                    repetition->rows = data16[1];
                }
                break;
            case GdsiiRecord::TEXTTYPE:
                if (label) {
                    if (data_type == GdsiiDataType::FourByteSignedInteger) {
                        set_type(label->tag, udata32[0]);
                    } else {
                        set_type(label->tag, udata16[0]);
                    }
                }
                break;
            case GdsiiRecord::PRESENTATION:
                if (label) label->anchor = (Anchor)(data16[0] & 0x000F);
                break;
            case GdsiiRecord::STRING:
                if (label) {
                    if (str[data_length - 1] == 0) data_length--;
                    label->text = (char*)allocate(data_length + 1);
                    memcpy(label->text, str, data_length);
                    label->text[data_length] = 0;
                }
                break;
            case GdsiiRecord::STRANS:
                if (reference)
                    reference->x_reflection = (data16[0] & 0x8000) != 0;
                else if (label)
                    label->x_reflection = (data16[0] & 0x8000) != 0;
                if (data16[0] & 0x0006) {
                    if (error_logger)
                        fputs(
                            "[GDSTK] Absolute magnification and rotation of references is not supported.\n",
                            error_logger);
                    if (error_code) *error_code = ErrorCode::UnsupportedRecord;
                }
                break;
            case GdsiiRecord::MAG:
                if (reference)
                    reference->magnification = gdsii_real_to_double(data64[0]);
                else if (label)
                    label->magnification = gdsii_real_to_double(data64[0]);
                break;
            case GdsiiRecord::ANGLE:
                if (reference)
                    reference->rotation = M_PI / 180.0 * gdsii_real_to_double(data64[0]);
                else if (label)
                    label->rotation = M_PI / 180.0 * gdsii_real_to_double(data64[0]);
                break;
            case GdsiiRecord::PATHTYPE:
                if (path) {
                    switch (data16[0]) {
                        case 0:
                            path->elements[0].end_type = EndType::Flush;
                            break;
                        case 1:
                            path->elements[0].end_type = EndType::Round;
                            break;
                        case 2:
                            path->elements[0].end_type = EndType::HalfWidth;
                            break;
                        default:
                            path->elements[0].end_type = EndType::Extended;
                    }
                }
                break;
            case GdsiiRecord::PROPATTR:
                key = data16[0];
                break;
            case GdsiiRecord::PROPVALUE:
                if (str[data_length - 1] != 0) str[data_length++] = 0;
                if (polygon) {
                    set_gds_property(polygon->properties, key, str, data_length);
                } else if (path) {
                    set_gds_property(path->properties, key, str, data_length);
                } else if (reference) {
                    set_gds_property(reference->properties, key, str, data_length);
                } else if (label) {
                    set_gds_property(label->properties, key, str, data_length);
                }
                break;
            case GdsiiRecord::BGNEXTN:
                if (path) path->elements[0].end_extensions.u = factor * data32[0];
                break;
            case GdsiiRecord::ENDEXTN:
                if (path) path->elements[0].end_extensions.v = factor * data32[0];
                break;
            // TODO: Consider NODE support (even though it is not available for OASIS)
            // case GdsiiRecord::TEXTNODE:
            // case GdsiiRecord::NODE:
            // case GdsiiRecord::SPACING:
            // case GdsiiRecord::UINTEGER:
            // case GdsiiRecord::USTRING:
            // case GdsiiRecord::REFLIBS:
            // case GdsiiRecord::FONTS:
            // case GdsiiRecord::GENERATIONS:
            // case GdsiiRecord::ATTRTABLE:
            // case GdsiiRecord::STYPTABLE:
            // case GdsiiRecord::STRTYPE:
            // case GdsiiRecord::ELFLAGS:
            // case GdsiiRecord::ELKEY:
            // case GdsiiRecord::LINKTYPE:
            // case GdsiiRecord::LINKKEYS:
            // case GdsiiRecord::NODETYPE:
            // case GdsiiRecord::PLEX:
            // case GdsiiRecord::TAPENUM:
            // case GdsiiRecord::TAPECODE:
            // case GdsiiRecord::STRCLASS:
            // case GdsiiRecord::RESERVED:
            // case GdsiiRecord::FORMAT:
            // case GdsiiRecord::MASK:
            // case GdsiiRecord::ENDMASKS:
            // case GdsiiRecord::LIBDIRSIZE:
            // case GdsiiRecord::SRFNAME:
            // case GdsiiRecord::LIBSECUR:
            default:
                if (buffer[2] < COUNT(gdsii_record_names)) {
                    if (error_logger)
                        fprintf(error_logger, "[GDSTK] Record type %s (0x%02X) is not supported.\n",
                                gdsii_record_names[buffer[2]], buffer[2]);
                } else {
                    if (error_logger)
                        fprintf(error_logger, "[GDSTK] Unknown record type 0x%02X.\n", buffer[2]);
                }
                if (error_code) *error_code = ErrorCode::UnsupportedRecord;
        }
    }

    library.free_all();
    fclose(in);
    return Library{};
}

// TODO: verify modal variables are correctly updated
Library read_oas(const char* filename, double unit, double tolerance, ErrorCode* error_code) {
    Library library = {};

    OasisStream in = {};
    in.file = fopen(filename, "rb");
    if (in.file == NULL) {
        if (error_logger) fputs("[GDSTK] Unable to open OASIS file for input.\n", error_logger);
        if (error_code) *error_code = ErrorCode::InputFileOpenError;
        return library;
    }

    // Check header bytes and START record
    char header[14];
    if (fread(header, 1, 14, in.file) < 14 || memcmp(header, "%SEMI-OASIS\r\n\x01", 14) != 0) {
        if (error_logger) fputs("[GDSTK] Invalid OASIS header found.\n", error_logger);
        if (error_code) *error_code = ErrorCode::InvalidFile;
        fclose(in.file);
        return library;
    }

    // Process START record
    uint64_t len;
    uint8_t* version = oasis_read_string(in, false, len);
    if (in.error_code != ErrorCode::NoError) {
        if (error_code) *error_code = in.error_code;
        fclose(in.file);
        return library;
    }
    if (len != 3 || memcmp(version, "1.0", 3) != 0) {
        if (error_logger) fputs("[GDSTK] Unsupported OASIS file version.\n", error_logger);
        if (error_code) *error_code = ErrorCode::InvalidFile;
    }
    free_allocation(version);

    double factor = 1 / oasis_read_real(in);
    library.precision = 1e-6 * factor;
    if (unit > 0) {
        library.unit = unit;
        factor *= 1e-6 / unit;
    } else {
        library.unit = 1e-6;
    }
    if (tolerance <= 0) {
        tolerance = library.precision / library.unit;
    }

    uint64_t offset_table_flag = oasis_read_unsigned_integer(in);
    if (offset_table_flag == 0) {
        // Skip offset table
        for (uint8_t i = 12; i > 0; i--) oasis_read_unsigned_integer(in);
    }

    // State variables
    bool modal_absolute_pos = true;
    uint32_t modal_layer = 0;
    uint32_t modal_datatype = 0;
    uint32_t modal_textlayer = 0;
    uint32_t modal_texttype = 0;
    Vec2 modal_placement_pos = {0, 0};
    Vec2 modal_text_pos = {0, 0};
    Vec2 modal_geom_pos = {0, 0};
    Vec2 modal_geom_dim = {0, 0};
    Repetition modal_repetition = {RepetitionType::None};
    Label* modal_text_string = NULL;
    Reference* modal_placement_cell = NULL;
    Array<Vec2> modal_polygon_points = {};
    modal_polygon_points.append(Vec2{0, 0});
    Array<Vec2> modal_path_points = {};
    modal_path_points.append(Vec2{0, 0});
    double modal_path_halfwidth = 0;
    Vec2 modal_path_extensions = {0, 0};
    uint8_t modal_ctrapezoid_type = 0;
    double modal_circle_radius = 0;
    Property* modal_property = NULL;
    PropertyValue* modal_property_value_list = NULL;

    Property** next_property = &library.properties;

    Array<Property*> unfinished_property_name = {};
    Array<PropertyValue*> unfinished_property_value = {};
    bool modal_property_unfinished = false;

    // Name tables
    Array<ByteArray> cell_name_table = {};
    Array<ByteArray> label_text_table = {};
    Array<ByteArray> property_name_table = {};
    Array<ByteArray> property_value_table = {};

    // Elements
    Cell* cell = NULL;

    // const char* oasis_record_names[] = {"PAD",
    //                                     "START",
    //                                     "END",
    //                                     "CELLNAME_IMPLICIT",
    //                                     "CELLNAME",
    //                                     "TEXTSTRING_IMPLICIT",
    //                                     "TEXTSTRING",
    //                                     "PROPNAME_IMPLICIT",
    //                                     "PROPNAME",
    //                                     "PROPSTRING_IMPLICIT",
    //                                     "PROPSTRING",
    //                                     "LAYERNAME_DATA",
    //                                     "LAYERNAME_TEXT",
    //                                     "CELL_REF_NUM",
    //                                     "CELL",
    //                                     "XYABSOLUTE",
    //                                     "XYRELATIVE",
    //                                     "PLACEMENT",
    //                                     "PLACEMENT_TRANSFORM",
    //                                     "TEXT",
    //                                     "RECTANGLE",
    //                                     "POLYGON",
    //                                     "PATH",
    //                                     "TRAPEZOID_AB",
    //                                     "TRAPEZOID_A",
    //                                     "TRAPEZOID_B",
    //                                     "CTRAPEZOID",
    //                                     "CIRCLE",
    //                                     "PROPERTY",
    //                                     "LAST_PROPERTY",
    //                                     "XNAME_IMPLICIT",
    //                                     "XNAME",
    //                                     "XELEMENT",
    //                                     "XGEOMETRY",
    //                                     "CBLOCK"};

    OasisRecord record;
    while ((error_code == NULL || *error_code == ErrorCode::NoError) &&
           oasis_read(&record, 1, 1, in) == ErrorCode::NoError) {
        // DEBUG_PRINT("Record [%02u] %s\n", (uint8_t)record,
        //             (uint8_t)record < COUNT(oasis_record_names)
        //                 ? oasis_record_names[(uint8_t)record]
        //                 : "---");
        switch (record) {
            case OasisRecord::PAD:
                break;
            case OasisRecord::START:
                // START is parsed before this loop
                if (error_logger)
                    fputs("[GDSTK] Unexpected START record out of position in file.\n",
                          error_logger);
                if (error_code) *error_code = ErrorCode::InvalidFile;
                break;
            case OasisRecord::END: {
                FSEEK64(in.file, 0, SEEK_END);
                library.name = (char*)allocate(4);
                library.name[0] = 'L';
                library.name[1] = 'I';
                library.name[2] = 'B';
                library.name[3] = 0;

                uint64_t c_size = library.cell_array.count;
                Map<Cell*> map = {};
                map.resize((uint64_t)(2.0 + 10.0 / GDSTK_MAP_CAPACITY_THRESHOLD * c_size));

                Cell** cell_p = library.cell_array.items;
                for (uint64_t i = c_size; i > 0; i--) {
                    cell = *cell_p++;
                    if (cell->name == NULL) {
                        ByteArray* cell_name = cell_name_table.items + (uint64_t)cell->owner;
                        cell->owner = NULL;
                        cell->name = copy_string((char*)cell_name->bytes, NULL);
                        if (cell_name->properties) {
                            Property* last = cell_name->properties;
                            while (last->next) last = last->next;
                            last->next = cell->properties;
                            cell->properties = cell_name->properties;
                            cell_name->properties = NULL;
                        }
                    }
                    map.set(cell->name, cell);

                    Label** label_p = cell->label_array.items;
                    for (uint64_t j = cell->label_array.count; j > 0; j--) {
                        Label* label = *label_p++;
                        if (label->text == NULL) {
                            ByteArray* label_text = label_text_table.items + (uint64_t)label->owner;
                            label->owner = NULL;
                            label->text = copy_string((char*)label_text->bytes, NULL);
                            if (label_text->properties) {
                                Property* copy = properties_copy(label_text->properties);
                                Property* last = copy;
                                while (last->next) last = last->next;
                                last->next = label->properties;
                                label->properties = copy;
                            }
                        }
                    }
                }

                cell_p = library.cell_array.items;
                for (uint64_t i = c_size; i > 0; i--, cell_p++) {
                    Reference** ref_p = (*cell_p)->reference_array.items;
                    for (uint64_t j = (*cell_p)->reference_array.count; j > 0; j--, ref_p++) {
                        Reference* ref = *ref_p;
                        if (ref->type == ReferenceType::Cell) {
                            // Using reference number
                            ByteArray* cell_name = cell_name_table.items + (uint64_t)ref->cell;
                            ref->cell = map.get((char*)cell_name->bytes);
                            if (!ref->cell) {
                                ref->type = ReferenceType::Name;
                                ref->name = (char*)allocate(cell_name->count);
                                memcpy(ref->name, cell_name->bytes, cell_name->count);
                                if (error_code) *error_code = ErrorCode::MissingReference;
                                if (error_logger)
                                    fprintf(error_logger, "[GDSTK] Missing referenced cell %s\n",
                                            ref->name);
                            }
                        } else {
                            // Using name
                            cell = map.get(ref->name);
                            if (cell) {
                                free_allocation(ref->name);
                                ref->cell = cell;
                                ref->type = ReferenceType::Cell;
                            } else {
                                if (error_code) *error_code = ErrorCode::MissingReference;
                                if (error_logger)
                                    fprintf(error_logger, "[GDSTK] Missing referenced cell %s\n",
                                            ref->name);
                            }
                        }
                    }
                }
                map.clear();

                Property** prop_p = unfinished_property_name.items;
                for (uint64_t i = unfinished_property_name.count; i > 0; i--) {
                    Property* property = *prop_p++;
                    ByteArray* prop_name = property_name_table.items + (uint64_t)property->name;
                    property->name = copy_string((char*)prop_name->bytes, NULL);
                }
                PropertyValue** prop_value_p = unfinished_property_value.items;
                for (uint64_t i = unfinished_property_value.count; i > 0; i--) {
                    PropertyValue* property_value = *prop_value_p++;
                    ByteArray* prop_string =
                        property_value_table.items + (uint64_t)property_value->unsigned_integer;
                    property_value->type = PropertyType::String;
                    property_value->count = prop_string->count;
                    property_value->bytes = (uint8_t*)allocate(prop_string->count);
                    memcpy(property_value->bytes, prop_string->bytes, prop_string->count);
                }
                goto CLEANUP;
            } break;
            case OasisRecord::CELLNAME_IMPLICIT: {
                uint8_t* bytes = oasis_read_string(in, true, len);
                cell_name_table.append(ByteArray{len, bytes, NULL});
                next_property = &cell_name_table[cell_name_table.count - 1].properties;
            } break;
            case OasisRecord::CELLNAME: {
                uint8_t* bytes = oasis_read_string(in, true, len);
                uint64_t ref_number = oasis_read_unsigned_integer(in);
                if (ref_number >= cell_name_table.count) {
                    cell_name_table.ensure_slots(ref_number + 1 - cell_name_table.count);
                    for (uint64_t i = cell_name_table.count; i < ref_number; i++) {
                        cell_name_table[i] = ByteArray{0, NULL, NULL};
                    }
                    cell_name_table.count = ref_number + 1;
                }
                cell_name_table[ref_number] = ByteArray{len, bytes, NULL};
                next_property = &cell_name_table[ref_number].properties;
            } break;
            case OasisRecord::TEXTSTRING_IMPLICIT: {
                uint8_t* bytes = oasis_read_string(in, true, len);
                label_text_table.append(ByteArray{len, bytes, NULL});
                next_property = &label_text_table[label_text_table.count - 1].properties;
            } break;
            case OasisRecord::TEXTSTRING: {
                uint8_t* bytes = oasis_read_string(in, true, len);
                uint64_t ref_number = oasis_read_unsigned_integer(in);
                if (ref_number >= label_text_table.count) {
                    label_text_table.ensure_slots(ref_number + 1 - label_text_table.count);
                    for (uint64_t i = label_text_table.count; i < ref_number; i++) {
                        label_text_table[i] = ByteArray{0, NULL, NULL};
                    }
                    label_text_table.count = ref_number + 1;
                }
                label_text_table[ref_number] = ByteArray{len, bytes, NULL};
                next_property = &label_text_table[ref_number].properties;
            } break;
            case OasisRecord::PROPNAME_IMPLICIT: {
                uint8_t* bytes = oasis_read_string(in, true, len);
                property_name_table.append(ByteArray{len, bytes, NULL});
                next_property = &property_name_table[property_name_table.count - 1].properties;
            } break;
            case OasisRecord::PROPNAME: {
                uint8_t* bytes = oasis_read_string(in, true, len);
                uint64_t ref_number = oasis_read_unsigned_integer(in);
                if (ref_number >= property_name_table.count) {
                    property_name_table.ensure_slots(ref_number + 1 - property_name_table.count);
                    for (uint64_t i = property_name_table.count; i < ref_number; i++) {
                        property_name_table[i] = ByteArray{0, NULL, NULL};
                    }
                    property_name_table.count = ref_number + 1;
                }
                property_name_table[ref_number] = ByteArray{len, bytes, NULL};
                next_property = &property_name_table[ref_number].properties;
            } break;
            case OasisRecord::PROPSTRING_IMPLICIT: {
                uint8_t* bytes = oasis_read_string(in, false, len);
                property_value_table.append(ByteArray{len, bytes, NULL});
                next_property = &property_value_table[property_value_table.count - 1].properties;
            } break;
            case OasisRecord::PROPSTRING: {
                uint8_t* bytes = oasis_read_string(in, false, len);
                uint64_t ref_number = oasis_read_unsigned_integer(in);
                if (ref_number >= property_value_table.count) {
                    property_value_table.ensure_slots(ref_number + 1 - property_value_table.count);
                    for (uint64_t i = property_value_table.count; i < ref_number; i++) {
                        property_value_table[i] = ByteArray{0, NULL, NULL};
                    }
                    property_value_table.count = ref_number + 1;
                }
                property_value_table[ref_number] = ByteArray{len, bytes, NULL};
                next_property = &property_value_table[ref_number].properties;
            } break;
            case OasisRecord::LAYERNAME_DATA:
            case OasisRecord::LAYERNAME_TEXT:
                // Unused record
                free_allocation(oasis_read_string(in, false, len));
                for (uint32_t i = 2; i > 0; i--) {
                    uint64_t type = oasis_read_unsigned_integer(in);
                    if (type > 0) {
                        if (type == 4) oasis_read_unsigned_integer(in);
                        oasis_read_unsigned_integer(in);
                    }
                }
                break;
            case OasisRecord::CELL_REF_NUM:
            case OasisRecord::CELL: {
                cell = (Cell*)allocate_clear(sizeof(Cell));
                library.cell_array.append(cell);
                next_property = &cell->properties;
                if (record == OasisRecord::CELL_REF_NUM) {
                    // Use owner as temporary storage for the reference number
                    cell->owner = (void*)oasis_read_unsigned_integer(in);
                } else {
                    cell->name = (char*)oasis_read_string(in, true, len);
                }
                modal_absolute_pos = true;
                modal_placement_pos = Vec2{0, 0};
                modal_geom_pos = Vec2{0, 0};
                modal_text_pos = Vec2{0, 0};
            } break;
            case OasisRecord::XYABSOLUTE:
                modal_absolute_pos = true;
                break;
            case OasisRecord::XYRELATIVE:
                modal_absolute_pos = false;
                break;
            case OasisRecord::PLACEMENT:
            case OasisRecord::PLACEMENT_TRANSFORM: {
                Reference* reference = (Reference*)allocate_clear(sizeof(Reference));
                cell->reference_array.append(reference);
                next_property = &reference->properties;
                uint8_t info;
                oasis_read(&info, 1, 1, in);
                if (info & 0x80) {
                    // Explicit reference
                    if (info & 0x40) {
                        // Reference number
                        reference->type = ReferenceType::Cell;
                        reference->cell = (Cell*)oasis_read_unsigned_integer(in);
                    } else {
                        // Cell name
                        reference->type = ReferenceType::Name;
                        reference->name = (char*)oasis_read_string(in, true, len);
                    }
                    modal_placement_cell = reference;
                } else {
                    // Use modal_placement_cell
                    if (modal_placement_cell->type == ReferenceType::Cell) {
                        reference->type = ReferenceType::Cell;
                        reference->cell = modal_placement_cell->cell;
                    } else {
                        reference->type = ReferenceType::Name;
                        reference->name = copy_string(modal_placement_cell->name, NULL);
                    }
                }
                if (record == OasisRecord::PLACEMENT) {
                    reference->magnification = 1;
                    switch (info & 0x06) {
                        case 0x02:
                            reference->rotation = M_PI * 0.5;
                            break;
                        case 0x04:
                            reference->rotation = M_PI;
                            break;
                        case 0x06:
                            reference->rotation = M_PI * 1.5;
                    }
                } else {
                    if (info & 0x04) {
                        reference->magnification = oasis_read_real(in);
                    } else {
                        reference->magnification = 1;
                    }
                    if (info & 0x02) {
                        reference->rotation = oasis_read_real(in) * (M_PI / 180.0);
                    }
                }
                reference->x_reflection = info & 0x01;
                if (info & 0x20) {
                    double x = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_placement_pos.x = x;
                    } else {
                        modal_placement_pos.x += x;
                    }
                }
                if (info & 0x10) {
                    double y = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_placement_pos.y = y;
                    } else {
                        modal_placement_pos.y += y;
                    }
                }
                reference->origin = modal_placement_pos;
                if (info & 0x08) {
                    oasis_read_repetition(in, factor, modal_repetition);
                    reference->repetition.copy_from(modal_repetition);
                }
            } break;
            case OasisRecord::TEXT: {
                Label* label = (Label*)allocate_clear(sizeof(Label));
                label->magnification = 1;
                label->anchor = Anchor::SW;
                cell->label_array.append(label);
                next_property = &label->properties;
                uint8_t info;
                oasis_read(&info, 1, 1, in);
                if (info & 0x40) {
                    // Explicit text
                    if (info & 0x20) {
                        // Reference number: use owner to temporarily store it
                        label->owner = (void*)oasis_read_unsigned_integer(in);
                    } else {
                        label->text = (char*)oasis_read_string(in, true, len);
                    }
                    modal_text_string = label;
                } else {
                    // Use modal_text_string
                    if (modal_text_string->text == NULL) {
                        label->owner = modal_text_string->owner;
                    } else {
                        label->text = copy_string(modal_text_string->text, NULL);
                    }
                }
                if (info & 0x01) {
                    modal_textlayer = (uint32_t)oasis_read_unsigned_integer(in);
                }
                set_layer(label->tag, modal_textlayer);
                if (info & 0x02) {
                    modal_texttype = (uint32_t)oasis_read_unsigned_integer(in);
                }
                set_type(label->tag, modal_texttype);
                if (info & 0x10) {
                    double x = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_text_pos.x = x;
                    } else {
                        modal_text_pos.x += x;
                    }
                }
                if (info & 0x08) {
                    double y = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_text_pos.y = y;
                    } else {
                        modal_text_pos.y += y;
                    }
                }
                label->origin = modal_text_pos;
                if (info & 0x04) {
                    oasis_read_repetition(in, factor, modal_repetition);
                    label->repetition.copy_from(modal_repetition);
                }
            } break;
            case OasisRecord::RECTANGLE: {
                Polygon* polygon = (Polygon*)allocate_clear(sizeof(Polygon));
                cell->polygon_array.append(polygon);
                next_property = &polygon->properties;
                uint8_t info;
                oasis_read(&info, 1, 1, in);
                if (info & 0x01) {
                    modal_layer = (uint32_t)oasis_read_unsigned_integer(in);
                }
                if (info & 0x02) {
                    modal_datatype = (uint32_t)oasis_read_unsigned_integer(in);
                }
                if (info & 0x40) {
                    modal_geom_dim.x = factor * oasis_read_unsigned_integer(in);
                }
                if (info & 0x20) {
                    modal_geom_dim.y = factor * oasis_read_unsigned_integer(in);
                } else if (info & 0x80) {
                    modal_geom_dim.y = modal_geom_dim.x;
                }
                if (info & 0x10) {
                    double x = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_geom_pos.x = x;
                    } else {
                        modal_geom_pos.x += x;
                    }
                }
                if (info & 0x08) {
                    double y = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_geom_pos.y = y;
                    } else {
                        modal_geom_pos.y += y;
                    }
                }
                *polygon = rectangle(modal_geom_pos, modal_geom_pos + modal_geom_dim,
                                     make_tag(modal_layer, modal_datatype));
                if (info & 0x04) {
                    oasis_read_repetition(in, factor, modal_repetition);
                    polygon->repetition.copy_from(modal_repetition);
                }
            } break;
            case OasisRecord::POLYGON: {
                Polygon* polygon = (Polygon*)allocate_clear(sizeof(Polygon));
                cell->polygon_array.append(polygon);
                next_property = &polygon->properties;
                uint8_t info;
                oasis_read(&info, 1, 1, in);
                if (info & 0x01) {
                    modal_layer = (uint32_t)oasis_read_unsigned_integer(in);
                }
                set_layer(polygon->tag, modal_layer);
                if (info & 0x02) {
                    modal_datatype = (uint32_t)oasis_read_unsigned_integer(in);
                }
                set_type(polygon->tag, modal_datatype);
                if (info & 0x20) {
                    modal_polygon_points.count = 1;
                    oasis_read_point_list(in, factor, true, modal_polygon_points);
                }
                polygon->point_array.copy_from(modal_polygon_points);
                if (info & 0x10) {
                    double x = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_geom_pos.x = x;
                    } else {
                        modal_geom_pos.x += x;
                    }
                }
                if (info & 0x08) {
                    double y = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_geom_pos.y = y;
                    } else {
                        modal_geom_pos.y += y;
                    }
                }
                Vec2* v = polygon->point_array.items;
                for (uint64_t i = polygon->point_array.count; i > 0; i--) {
                    *v++ += modal_geom_pos;
                }
                if (info & 0x04) {
                    oasis_read_repetition(in, factor, modal_repetition);
                    polygon->repetition.copy_from(modal_repetition);
                }
            } break;
            case OasisRecord::PATH: {
                FlexPath* path = (FlexPath*)allocate_clear(sizeof(FlexPath));
                FlexPathElement* element =
                    (FlexPathElement*)allocate_clear(sizeof(FlexPathElement));
                cell->flexpath_array.append(path);
                next_property = &path->properties;
                path->spine.tolerance = tolerance;
                path->elements = element;
                path->num_elements = 1;
                path->simple_path = true;
                path->scale_width = true;
                uint8_t info;
                oasis_read(&info, 1, 1, in);
                if (info & 0x01) {
                    modal_layer = (uint32_t)oasis_read_unsigned_integer(in);
                }
                set_layer(element->tag, modal_layer);
                if (info & 0x02) {
                    modal_datatype = (uint32_t)oasis_read_unsigned_integer(in);
                }
                set_type(element->tag, modal_datatype);
                if (info & 0x40) {
                    modal_path_halfwidth = factor * oasis_read_unsigned_integer(in);
                }
                element->half_width_and_offset.append(Vec2{modal_path_halfwidth, 0});
                if (info & 0x80) {
                    uint8_t extension_scheme;
                    oasis_read(&extension_scheme, 1, 1, in);
                    switch (extension_scheme & 0x0c) {
                        case 0x04:
                            modal_path_extensions.u = 0;
                            break;
                        case 0x08:
                            modal_path_extensions.u = modal_path_halfwidth;
                            break;
                        case 0x0c:
                            modal_path_extensions.u = factor * oasis_read_integer(in);
                    }
                    switch (extension_scheme & 0x03) {
                        case 0x01:
                            modal_path_extensions.v = 0;
                            break;
                        case 0x02:
                            modal_path_extensions.v = modal_path_halfwidth;
                            break;
                        case 0x03:
                            modal_path_extensions.v = factor * oasis_read_integer(in);
                    }
                }
                if (modal_path_extensions.u == 0 && modal_path_extensions.v == 0) {
                    element->end_type = EndType::Flush;
                } else if (modal_path_extensions.u == modal_path_halfwidth &&
                           modal_path_extensions.v == modal_path_halfwidth) {
                    element->end_type = EndType::HalfWidth;
                } else {
                    element->end_type = EndType::Extended;
                    element->end_extensions = modal_path_extensions;
                }
                if (info & 0x20) {
                    modal_path_points.count = 1;
                    oasis_read_point_list(in, factor, false, modal_path_points);
                }
                if (info & 0x10) {
                    double x = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_geom_pos.x = x;
                    } else {
                        modal_geom_pos.x += x;
                    }
                }
                if (info & 0x08) {
                    double y = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_geom_pos.y = y;
                    } else {
                        modal_geom_pos.y += y;
                    }
                }
                path->spine.append(modal_geom_pos);
                const Array<Vec2> skip_first = {0, modal_path_points.count - 1,
                                                modal_path_points.items + 1};
                path->segment(skip_first, NULL, NULL, true);
                if (info & 0x04) {
                    oasis_read_repetition(in, factor, modal_repetition);
                    path->repetition.copy_from(modal_repetition);
                }
            } break;
            case OasisRecord::TRAPEZOID_AB:
            case OasisRecord::TRAPEZOID_A:
            case OasisRecord::TRAPEZOID_B: {
                Polygon* polygon = (Polygon*)allocate_clear(sizeof(Polygon));
                cell->polygon_array.append(polygon);
                next_property = &polygon->properties;
                uint8_t info;
                oasis_read(&info, 1, 1, in);
                if (info & 0x01) {
                    modal_layer = (uint32_t)oasis_read_unsigned_integer(in);
                }
                set_layer(polygon->tag, modal_layer);
                if (info & 0x02) {
                    modal_datatype = (uint32_t)oasis_read_unsigned_integer(in);
                }
                set_type(polygon->tag, modal_datatype);
                if (info & 0x40) {
                    modal_geom_dim.x = factor * oasis_read_unsigned_integer(in);
                }
                if (info & 0x20) {
                    modal_geom_dim.y = factor * oasis_read_unsigned_integer(in);
                }
                double delta_a, delta_b;
                if (record == OasisRecord::TRAPEZOID_AB) {
                    delta_a = factor * oasis_read_1delta(in);
                    delta_b = factor * oasis_read_1delta(in);
                } else if (record == OasisRecord::TRAPEZOID_A) {
                    delta_a = factor * oasis_read_1delta(in);
                    delta_b = 0;
                } else {
                    delta_a = 0;
                    delta_b = factor * oasis_read_1delta(in);
                }
                if (info & 0x10) {
                    double x = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_geom_pos.x = x;
                    } else {
                        modal_geom_pos.x += x;
                    }
                }
                if (info & 0x08) {
                    double y = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_geom_pos.y = y;
                    } else {
                        modal_geom_pos.y += y;
                    }
                }
                Array<Vec2>* point_array = &polygon->point_array;
                point_array->ensure_slots(4);
                point_array->count = 4;
                Vec2* r = point_array->items;
                Vec2* s = r + 1;
                Vec2* q = s + 1;
                Vec2* p = q + 1;
                if (info & 0x80) {
                    p->x = q->x = modal_geom_pos.x;
                    r->x = s->x = modal_geom_pos.x + modal_geom_dim.x;
                    if (delta_a < 0) {
                        p->y = modal_geom_pos.y;
                        r->y = p->y - delta_a;
                    } else {
                        r->y = modal_geom_pos.y;
                        p->y = r->y + delta_a;
                    }
                    if (delta_b < 0) {
                        s->y = modal_geom_pos.y + modal_geom_dim.y;
                        q->y = s->y + delta_b;
                    } else {
                        q->y = modal_geom_pos.y + modal_geom_dim.y;
                        s->y = q->y - delta_b;
                    }
                } else {
                    p->y = q->y = modal_geom_pos.y + modal_geom_dim.y;
                    r->y = s->y = modal_geom_pos.y;
                    if (delta_a < 0) {
                        p->x = modal_geom_pos.x;
                        r->x = p->x - delta_a;
                    } else {
                        r->x = modal_geom_pos.x;
                        p->x = r->x + delta_a;
                    }
                    if (delta_b < 0) {
                        s->x = modal_geom_pos.x + modal_geom_dim.x;
                        q->x = s->x + delta_b;
                    } else {
                        q->x = modal_geom_pos.x + modal_geom_dim.x;
                        s->x = q->x - delta_b;
                    }
                }
                if (info & 0x04) {
                    oasis_read_repetition(in, factor, modal_repetition);
                    polygon->repetition.copy_from(modal_repetition);
                }
            } break;
            case OasisRecord::CTRAPEZOID: {
                Polygon* polygon = (Polygon*)allocate_clear(sizeof(Polygon));
                cell->polygon_array.append(polygon);
                next_property = &polygon->properties;
                uint8_t info;
                oasis_read(&info, 1, 1, in);
                if (info & 0x01) {
                    modal_layer = (uint32_t)oasis_read_unsigned_integer(in);
                }
                set_layer(polygon->tag, modal_layer);
                if (info & 0x02) {
                    modal_datatype = (uint32_t)oasis_read_unsigned_integer(in);
                }
                set_type(polygon->tag, modal_datatype);
                if (info & 0x80) {
                    oasis_read(&modal_ctrapezoid_type, 1, 1, in);
                }
                if (info & 0x40) {
                    modal_geom_dim.x = factor * oasis_read_unsigned_integer(in);
                }
                if (info & 0x20) {
                    modal_geom_dim.y = factor * oasis_read_unsigned_integer(in);
                }
                if (info & 0x10) {
                    double x = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_geom_pos.x = x;
                    } else {
                        modal_geom_pos.x += x;
                    }
                }
                if (info & 0x08) {
                    double y = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_geom_pos.y = y;
                    } else {
                        modal_geom_pos.y += y;
                    }
                }
                Array<Vec2>* point_array = &polygon->point_array;
                Vec2* v;
                if (modal_ctrapezoid_type > 15 && modal_ctrapezoid_type < 24) {
                    point_array->ensure_slots(3);
                    point_array->count = 3;
                    v = point_array->items;
                    v[0] = modal_geom_pos;
                    v[1] = modal_geom_pos;
                    v[2] = modal_geom_pos;
                } else {
                    point_array->ensure_slots(4);
                    point_array->count = 4;
                    v = point_array->items;
                    v[0] = modal_geom_pos;
                    v[1] = modal_geom_pos + Vec2{modal_geom_dim.x, 0};
                    v[2] = modal_geom_pos + modal_geom_dim;
                    v[3] = modal_geom_pos + Vec2{0, modal_geom_dim.y};
                }
                switch (modal_ctrapezoid_type) {
                    case 0:
                        v[2].x -= modal_geom_dim.y;
                        break;
                    case 1:
                        v[1].x -= modal_geom_dim.y;
                        break;
                    case 2:
                        v[3].x += modal_geom_dim.y;
                        break;
                    case 3:
                        v[0].x += modal_geom_dim.y;
                        break;
                    case 4:
                        v[2].x -= modal_geom_dim.y;
                        v[3].x += modal_geom_dim.y;
                        break;
                    case 5:
                        v[0].x += modal_geom_dim.y;
                        v[1].x -= modal_geom_dim.y;
                        break;
                    case 6:
                        v[1].x -= modal_geom_dim.y;
                        v[3].x += modal_geom_dim.y;
                        break;
                    case 7:
                        v[0].x += modal_geom_dim.y;
                        v[2].x -= modal_geom_dim.y;
                        break;
                    case 8:
                        v[2].y -= modal_geom_dim.x;
                        break;
                    case 9:
                        v[3].y -= modal_geom_dim.x;
                        break;
                    case 10:
                        v[1].y += modal_geom_dim.x;
                        break;
                    case 11:
                        v[0].y += modal_geom_dim.x;
                        break;
                    case 12:
                        v[1].y += modal_geom_dim.x;
                        v[2].y -= modal_geom_dim.x;
                        break;
                    case 13:
                        v[0].y += modal_geom_dim.x;
                        v[3].y -= modal_geom_dim.x;
                        break;
                    case 14:
                        v[1].y += modal_geom_dim.x;
                        v[3].y -= modal_geom_dim.x;
                        break;
                    case 15:
                        v[0].y += modal_geom_dim.x;
                        v[2].y -= modal_geom_dim.x;
                        break;
                    case 16:
                        v[1].x += modal_geom_dim.x;
                        v[2].y += modal_geom_dim.x;
                        modal_geom_dim.y = modal_geom_dim.x;
                        break;
                    case 17:
                        v[1] += modal_geom_dim.x;
                        v[2].y += modal_geom_dim.x;
                        modal_geom_dim.y = modal_geom_dim.x;
                        break;
                    case 18:
                        v[1].x += modal_geom_dim.x;
                        v[2] += modal_geom_dim.x;
                        modal_geom_dim.y = modal_geom_dim.x;
                        break;
                    case 19:
                        v[0].x += modal_geom_dim.x;
                        v[1] += modal_geom_dim.x;
                        v[2].y += modal_geom_dim.x;
                        modal_geom_dim.y = modal_geom_dim.x;
                        break;
                    case 20:
                        v[1].x += 2 * modal_geom_dim.y;
                        v[2] += modal_geom_dim.y;
                        modal_geom_dim.x = 2 * modal_geom_dim.y;
                        break;
                    case 21:
                        v[0].x += modal_geom_dim.y;
                        v[1].x += 2 * modal_geom_dim.y;
                        v[1].y += modal_geom_dim.y;
                        v[2].y += modal_geom_dim.y;
                        modal_geom_dim.x = 2 * modal_geom_dim.y;
                        break;
                    case 22:
                        v[1] += modal_geom_dim.x;
                        v[2].y += 2 * modal_geom_dim.x;
                        modal_geom_dim.y = 2 * modal_geom_dim.x;
                        break;
                    case 23:
                        v[0].x += modal_geom_dim.x;
                        v[1].x += modal_geom_dim.x;
                        v[1].y += 2 * modal_geom_dim.x;
                        v[2].y += modal_geom_dim.x;
                        modal_geom_dim.y = 2 * modal_geom_dim.x;
                        break;
                    case 25:
                        v[2].y = v[3].y = modal_geom_pos.y + modal_geom_dim.x;
                        break;
                }
                if (info & 0x04) {
                    oasis_read_repetition(in, factor, modal_repetition);
                    polygon->repetition.copy_from(modal_repetition);
                }
            } break;
            case OasisRecord::CIRCLE: {
                Polygon* polygon = (Polygon*)allocate_clear(sizeof(Polygon));
                cell->polygon_array.append(polygon);
                next_property = &polygon->properties;
                uint8_t info;
                oasis_read(&info, 1, 1, in);
                if (info & 0x01) {
                    modal_layer = (uint32_t)oasis_read_unsigned_integer(in);
                }
                if (info & 0x02) {
                    modal_datatype = (uint32_t)oasis_read_unsigned_integer(in);
                }
                if (info & 0x20) {
                    modal_circle_radius = factor * oasis_read_unsigned_integer(in);
                }
                if (info & 0x10) {
                    double x = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_geom_pos.x = x;
                    } else {
                        modal_geom_pos.x += x;
                    }
                }
                if (info & 0x08) {
                    double y = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_geom_pos.y = y;
                    } else {
                        modal_geom_pos.y += y;
                    }
                }
                *polygon = ellipse(modal_geom_pos, modal_circle_radius, modal_circle_radius, 0, 0,
                                   0, 0, tolerance, make_tag(modal_layer, modal_datatype));
                if (info & 0x04) {
                    oasis_read_repetition(in, factor, modal_repetition);
                    polygon->repetition.copy_from(modal_repetition);
                }
            } break;
            case OasisRecord::PROPERTY:
            case OasisRecord::LAST_PROPERTY: {
                Property* property = (Property*)allocate_clear(sizeof(Property));
                *next_property = property;
                next_property = &property->next;
                uint8_t info;
                if (record == OasisRecord::LAST_PROPERTY) {
                    info = 0x08;
                } else {
                    oasis_read(&info, 1, 1, in);
                }
                if (info & 0x04) {
                    // Explicit name
                    if (info & 0x02) {
                        // Reference number
                        property->name = (char*)oasis_read_unsigned_integer(in);
                        unfinished_property_name.append(property);
                        modal_property_unfinished = true;
                    } else {
                        property->name = (char*)oasis_read_string(in, true, len);
                        modal_property_unfinished = false;
                    }
                    modal_property = property;
                } else {
                    // Use modal variable
                    if (modal_property_unfinished) {
                        property->name = modal_property->name;
                        unfinished_property_name.append(property);
                    } else {
                        property->name = copy_string(modal_property->name, NULL);
                    }
                }
                if (info & 0x08) {
                    // Use modal value list
                    property->value = property_values_copy(modal_property_value_list);
                    PropertyValue* src = modal_property_value_list;
                    PropertyValue* dst = property->value;
                    while (src) {
                        if (src->type == PropertyType::UnsignedInteger &&
                            unfinished_property_value.contains(src)) {
                            unfinished_property_value.append(dst);
                        }
                        src = src->next;
                        dst = dst->next;
                    }
                } else {
                    // Explicit value list
                    uint64_t num_values = info >> 4;
                    if (num_values == 15) {
                        num_values = oasis_read_unsigned_integer(in);
                    }
                    PropertyValue** next = &property->value;
                    for (; num_values > 0; num_values--) {
                        PropertyValue* property_value =
                            (PropertyValue*)allocate_clear(sizeof(PropertyValue));
                        *next = property_value;
                        next = &property_value->next;
                        OasisDataType data_type;
                        oasis_read(&data_type, 1, 1, in);
                        switch (data_type) {
                            case OasisDataType::RealPositiveInteger:
                            case OasisDataType::RealNegativeInteger:
                            case OasisDataType::RealPositiveReciprocal:
                            case OasisDataType::RealNegativeReciprocal:
                            case OasisDataType::RealPositiveRatio:
                            case OasisDataType::RealNegativeRatio:
                            case OasisDataType::RealFloat:
                            case OasisDataType::RealDouble: {
                                property_value->type = PropertyType::Real;
                                property_value->real = oasis_read_real_by_type(in, data_type);
                            } break;
                            case OasisDataType::UnsignedInteger: {
                                property_value->type = PropertyType::UnsignedInteger;
                                property_value->unsigned_integer = oasis_read_unsigned_integer(in);
                            } break;
                            case OasisDataType::SignedInteger: {
                                property_value->type = PropertyType::Integer;
                                property_value->integer = oasis_read_integer(in);
                            } break;
                            case OasisDataType::AString:
                            case OasisDataType::BString:
                            case OasisDataType::NString: {
                                property_value->type = PropertyType::String;
                                property_value->bytes =
                                    oasis_read_string(in, false, property_value->count);
                            } break;
                            case OasisDataType::ReferenceA:
                            case OasisDataType::ReferenceB:
                            case OasisDataType::ReferenceN: {
                                property_value->type = PropertyType::UnsignedInteger;
                                property_value->unsigned_integer = oasis_read_unsigned_integer(in);
                                unfinished_property_value.append(property_value);
                            } break;
                        }
                    }
                    modal_property_value_list = property->value;
                }
            } break;
            case OasisRecord::XNAME_IMPLICIT: {
                oasis_read_unsigned_integer(in);
                free_allocation(oasis_read_string(in, false, len));
                if (error_logger) fputs("[GDSTK] Record type XNAME ignored.\n", error_logger);
                if (error_code) *error_code = ErrorCode::UnsupportedRecord;
            } break;
            case OasisRecord::XNAME: {
                oasis_read_unsigned_integer(in);
                free_allocation(oasis_read_string(in, false, len));
                oasis_read_unsigned_integer(in);
                if (error_logger) fputs("[GDSTK] Record type XNAME ignored.\n", error_logger);
                if (error_code) *error_code = ErrorCode::UnsupportedRecord;
            } break;
            case OasisRecord::XELEMENT: {
                oasis_read_unsigned_integer(in);
                free_allocation(oasis_read_string(in, false, len));
                if (error_logger) fputs("[GDSTK] Record type XELEMENT ignored.\n", error_logger);
                if (error_code) *error_code = ErrorCode::UnsupportedRecord;
            } break;
            case OasisRecord::XGEOMETRY: {
                uint8_t info;
                oasis_read(&info, 1, 1, in);
                oasis_read_unsigned_integer(in);
                if (info & 0x01) {
                    modal_layer = (uint32_t)oasis_read_unsigned_integer(in);
                }
                if (info & 0x02) {
                    modal_datatype = (uint32_t)oasis_read_unsigned_integer(in);
                }
                free_allocation(oasis_read_string(in, false, len));
                if (info & 0x10) {
                    double x = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_geom_pos.x = x;
                    } else {
                        modal_geom_pos.x += x;
                    }
                }
                if (info & 0x08) {
                    double y = factor * oasis_read_integer(in);
                    if (modal_absolute_pos) {
                        modal_geom_pos.y = y;
                    } else {
                        modal_geom_pos.y += y;
                    }
                }
                if (info & 0x04) {
                    oasis_read_repetition(in, factor, modal_repetition);
                }
                if (error_logger) fputs("[GDSTK] Record type XGEOMETRY ignored.\n", error_logger);
                if (error_code) *error_code = ErrorCode::UnsupportedRecord;
            } break;
            case OasisRecord::CBLOCK: {
                if (oasis_read_unsigned_integer(in) != 0) {
                    if (error_logger)
                        fputs("[GDSTK] CBLOCK compression method not supported.\n", error_logger);
                    if (error_code) *error_code = ErrorCode::InvalidFile;
                    oasis_read_unsigned_integer(in);
                    len = oasis_read_unsigned_integer(in);
                    assert(len <= INT64_MAX);
                    FSEEK64(in.file, (int64_t)len, SEEK_SET);
                } else {
                    z_stream s = {};
                    s.zalloc = zalloc;
                    s.zfree = zfree;
                    in.data_size = oasis_read_unsigned_integer(in);
                    s.avail_out = (uInt)in.data_size;
                    s.avail_in = (uInt)oasis_read_unsigned_integer(in);
                    in.data = (uint8_t*)allocate(in.data_size);
                    in.cursor = in.data;
                    s.next_out = in.data;
                    uint8_t* data = (uint8_t*)allocate(s.avail_in);
                    s.next_in = (Bytef*)data;
                    if (fread(s.next_in, 1, s.avail_in, in.file) != s.avail_in) {
                        if (error_logger)
                            fputs("[GDSTK] Unable to read full CBLOCK.\n", error_logger);
                        if (error_code) *error_code = ErrorCode::InvalidFile;
                    }
                    if (inflateInit2(&s, -15) != Z_OK) {
                        if (error_logger)
                            fputs("[GDSTK] Unable to initialize zlib.\n", error_logger);
                        if (error_code) *error_code = ErrorCode::ZlibError;
                    }
                    int ret = inflate(&s, Z_FINISH);
                    if (ret != Z_STREAM_END) {
                        if (error_logger)
                            fputs("[GDSTK] Unable to decompress CBLOCK.\n", error_logger);
                        if (error_code) *error_code = ErrorCode::ZlibError;
                    }
                    free_allocation(data);
                    inflateEnd(&s);
                    // Empty CBLOCK
                    if (in.data_size == 0) {
                        free_allocation(in.data);
                        in.data = NULL;
                    }
                }
            } break;
            default:
                if (error_logger)
                    fprintf(error_logger, "[GDSTK] Unknown record type <0x%02X>.\n",
                            (uint8_t)record);
                if (error_code) *error_code = ErrorCode::UnsupportedRecord;
        }
    }
    if (in.error_code != ErrorCode::NoError && error_code) *error_code = in.error_code;

CLEANUP:
    fclose(in.file);

    ByteArray* ba = cell_name_table.items;
    for (uint64_t i = cell_name_table.count; i > 0; i--, ba++) {
        if (ba->bytes) free_allocation(ba->bytes);
        properties_clear(ba->properties);
    }
    cell_name_table.clear();

    ba = label_text_table.items;
    for (uint64_t i = label_text_table.count; i > 0; i--, ba++) {
        if (ba->bytes) free_allocation(ba->bytes);
        properties_clear(ba->properties);
    }
    label_text_table.clear();

    ba = property_name_table.items;
    for (uint64_t i = property_name_table.count; i > 0; i--, ba++) {
        if (ba->bytes) free_allocation(ba->bytes);
        properties_clear(ba->properties);
    }
    property_name_table.clear();

    ba = property_value_table.items;
    for (uint64_t i = property_value_table.count; i > 0; i--, ba++) {
        if (ba->bytes) free_allocation(ba->bytes);
        properties_clear(ba->properties);
    }
    property_value_table.clear();

    modal_repetition.clear();
    modal_polygon_points.clear();
    modal_path_points.clear();

    unfinished_property_name.clear();
    unfinished_property_value.clear();

    return library;
}

ErrorCode gds_units(const char* filename, double& unit, double& precision) {
    uint8_t buffer[65537];
    uint64_t* data64 = (uint64_t*)(buffer + 4);
    FILE* in = fopen(filename, "rb");
    if (in == NULL) {
        fputs("[GDSTK] Unable to open GDSII file for input.\n", stderr);
        return ErrorCode::InputFileOpenError;
    }

    while (true) {
        uint64_t record_length = COUNT(buffer);
        ErrorCode error_code = gdsii_read_record(in, buffer, record_length);
        if (error_code != ErrorCode::NoError) {
            fclose(in);
            return error_code;
        }
        if ((GdsiiRecord)buffer[2] == GdsiiRecord::UNITS) {
            big_endian_swap64(data64, 2);
            precision = gdsii_real_to_double(data64[1]);
            unit = precision / gdsii_real_to_double(data64[0]);
            fclose(in);
            return ErrorCode::NoError;
        }
    }
    fclose(in);
    fputs("[GDSTK] GDSII file missing units definition.\n", stderr);
    return ErrorCode::InvalidFile;
}

tm gds_timestamp(const char* filename, const tm* new_timestamp, ErrorCode* error_code) {
    tm result = {};
    uint8_t buffer[65537];
    uint16_t* data16 = (uint16_t*)(buffer + 4);
    uint16_t new_tm_buffer[12];
    FILE* inout = NULL;

    if (new_timestamp) {
        new_tm_buffer[0] = new_timestamp->tm_year;
        new_tm_buffer[1] = new_timestamp->tm_mon + 1;
        new_tm_buffer[2] = new_timestamp->tm_mday;
        new_tm_buffer[3] = new_timestamp->tm_hour;
        new_tm_buffer[4] = new_timestamp->tm_min;
        new_tm_buffer[5] = new_timestamp->tm_sec;
        big_endian_swap16(new_tm_buffer, 6);
        memcpy(new_tm_buffer + 6, new_tm_buffer, 6 * sizeof(uint16_t));
        inout = fopen(filename, "r+b");
    } else {
        inout = fopen(filename, "rb");
    }
    if (inout == NULL) {
        if (error_logger) fputs("[GDSTK] Unable to open GDSII file.\n", error_logger);
        if (error_code) *error_code = ErrorCode::InputFileOpenError;
        return result;
    }

    while (true) {
        uint64_t record_length = COUNT(buffer);
        ErrorCode err = gdsii_read_record(inout, buffer, record_length);
        if (err != ErrorCode::NoError) {
            if (error_code) *error_code = err;
            fclose(inout);
            return result;
        }

        GdsiiRecord record = (GdsiiRecord)buffer[2];
        if (record == GdsiiRecord::BGNLIB) {
            if (record_length != 28) {
                fclose(inout);
                if (error_logger) fputs("[GDSTK] Invalid or corrupted GDSII file.\n", error_logger);
                if (error_code) *error_code = ErrorCode::InvalidFile;
                return result;
            }
            big_endian_swap16(data16, 6);
            result.tm_year = data16[0];
            result.tm_mon = data16[1] - 1;
            result.tm_mday = data16[2];
            result.tm_hour = data16[3];
            result.tm_min = data16[4];
            result.tm_sec = data16[5];
            if (!new_timestamp) {
                fclose(inout);
                return result;
            }
            if (FSEEK64(inout, -24, SEEK_CUR) != 0) {
                fclose(inout);
                if (error_logger)
                    fputs("[GDSTK] Unable to rewrite library timestamp.\n", error_logger);
                if (error_code) *error_code = ErrorCode::FileError;
                return result;
            }
            fwrite(new_tm_buffer, sizeof(uint16_t), 12, inout);
        } else if (record == GdsiiRecord::BGNSTR && new_timestamp) {
            if (record_length != 28) {
                fclose(inout);
                if (error_logger) fputs("[GDSTK] Invalid or corrupted GDSII file.\n", error_logger);
                if (error_code) *error_code = ErrorCode::InvalidFile;
                return result;
            }
            if (FSEEK64(inout, -24, SEEK_CUR) != 0) {
                fclose(inout);
                if (error_logger)
                    fputs("[GDSTK] Unable to rewrite cell timestamp.\n", error_logger);
                if (error_code) *error_code = ErrorCode::FileError;
                return result;
            }
            fwrite(new_tm_buffer, sizeof(uint16_t), 12, inout);
        } else if (record == GdsiiRecord::ENDLIB) {
            break;
        }
    }
    fclose(inout);
    return result;
}

ErrorCode gds_info(const char* filename, LibraryInfo& info) {
    // One extra char in case we need a 0-terminated string with max count (should never happen, but
    // it doesn't hurt to be prepared).
    uint8_t buffer[65537];
    uint16_t* data16 = (uint16_t*)(buffer + 4);
    uint32_t* data32 = (uint32_t*)(buffer + 4);
    uint64_t* data64 = (uint64_t*)(buffer + 4);
    char* str = (char*)(buffer + 4);

    FILE* in = fopen(filename, "rb");
    if (in == NULL) {
        if (error_logger) fputs("[GDSTK] Unable to open GDSII file for input.\n", error_logger);
        return ErrorCode::InputFileOpenError;
    }

    ErrorCode error = ErrorCode::NoError;
    uint32_t layer = 0;
    Set<Tag>* next_set = NULL;
    while (true) {
        uint64_t record_length = COUNT(buffer);
        ErrorCode err = gdsii_read_record(in, buffer, record_length);
        if (err != ErrorCode::NoError) {
            fclose(in);
            return err;
        }

        uint64_t data_length;
        switch ((GdsiiRecord)(buffer[2])) {
            case GdsiiRecord::ENDLIB:
                fclose(in);
                return error;
                break;
            case GdsiiRecord::STRNAME: {
                data_length = record_length - 4;
                if (str[data_length - 1] == 0) data_length--;
                char* name = (char*)allocate(data_length + 1);
                memcpy(name, str, data_length);
                name[data_length] = 0;
                info.cell_names.append(name);
            } break;
            case GdsiiRecord::UNITS:
                data_length = (record_length - 4) / 8;
                big_endian_swap64(data64, data_length);
                info.precision = gdsii_real_to_double(data64[1]);
                info.unit = info.precision / gdsii_real_to_double(data64[0]);
                break;
            case GdsiiRecord::BOUNDARY:
            case GdsiiRecord::BOX:
                info.num_polygons++;
                next_set = &info.shape_tags;
                break;
            case GdsiiRecord::PATH:
                info.num_paths++;
                next_set = &info.shape_tags;
                break;
            case GdsiiRecord::SREF:
            case GdsiiRecord::AREF:
                info.num_references++;
                next_set = NULL;
                break;
            case GdsiiRecord::TEXT:
                info.num_labels++;
                next_set = &info.label_tags;
                break;
            case GdsiiRecord::LAYER:
                if ((GdsiiDataType)buffer[3] == GdsiiDataType::FourByteSignedInteger) {
                    big_endian_swap32((uint32_t*)data32, 1);
                    layer = data32[0];
                } else {
                    big_endian_swap16((uint16_t*)data16, 1);
                    layer = data16[0];
                }
                break;
            case GdsiiRecord::DATATYPE:
            case GdsiiRecord::BOXTYPE:
            case GdsiiRecord::TEXTTYPE:
                if (!next_set) {
                    if (error_logger)
                        fputs("[GDSTK] Inconsistency detected in GDSII file.\n", error_logger);
                    error = ErrorCode::InvalidFile;
                } else if ((GdsiiDataType)buffer[3] == GdsiiDataType::FourByteSignedInteger) {
                    big_endian_swap32((uint32_t*)data32, 1);
                    next_set->add(make_tag(layer, data32[0]));
                    next_set = NULL;
                } else {
                    big_endian_swap16((uint16_t*)data16, 1);
                    next_set->add(make_tag(layer, data16[0]));
                    next_set = NULL;
                }
                break;
            // case GdsiiRecord::HEADER:
            // case GdsiiRecord::BGNLIB:
            // case GdsiiRecord::ENDSTR:
            // case GdsiiRecord::LIBNAME:
            // case GdsiiRecord::BGNSTR:
            // case GdsiiRecord::WIDTH:
            // case GdsiiRecord::XY:
            // case GdsiiRecord::ENDEL:
            // case GdsiiRecord::SNAME:
            // case GdsiiRecord::COLROW:
            // case GdsiiRecord::PRESENTATION:
            // case GdsiiRecord::STRING:
            // case GdsiiRecord::STRANS:
            // case GdsiiRecord::MAG:
            // case GdsiiRecord::ANGLE:
            // case GdsiiRecord::PATHTYPE:
            // case GdsiiRecord::PROPATTR:
            // case GdsiiRecord::PROPVALUE:
            // case GdsiiRecord::BGNEXTN:
            // case GdsiiRecord::ENDEXTN:
            // case GdsiiRecord::TEXTNODE:
            // case GdsiiRecord::NODE:
            // case GdsiiRecord::SPACING:
            // case GdsiiRecord::UINTEGER:
            // case GdsiiRecord::USTRING:
            // case GdsiiRecord::REFLIBS:
            // case GdsiiRecord::FONTS:
            // case GdsiiRecord::GENERATIONS:
            // case GdsiiRecord::ATTRTABLE:
            // case GdsiiRecord::STYPTABLE:
            // case GdsiiRecord::STRTYPE:
            // case GdsiiRecord::ELFLAGS:
            // case GdsiiRecord::ELKEY:
            // case GdsiiRecord::LINKTYPE:
            // case GdsiiRecord::LINKKEYS:
            // case GdsiiRecord::NODETYPE:
            // case GdsiiRecord::PLEX:
            // case GdsiiRecord::TAPENUM:
            // case GdsiiRecord::TAPECODE:
            // case GdsiiRecord::STRCLASS:
            // case GdsiiRecord::RESERVED:
            // case GdsiiRecord::FORMAT:
            // case GdsiiRecord::MASK:
            // case GdsiiRecord::ENDMASKS:
            // case GdsiiRecord::LIBDIRSIZE:
            // case GdsiiRecord::SRFNAME:
            // case GdsiiRecord::LIBSECUR:
            default:
                break;
        }
    }
    return ErrorCode::InvalidFile;
}

ErrorCode oas_precision(const char* filename, double& precision) {
    FILE* in = fopen(filename, "rb");
    if (in == NULL) {
        if (error_logger) fputs("[GDSTK] Unable to open OASIS file for input.\n", error_logger);
        return ErrorCode::InputFileOpenError;
    }

    // Check header bytes and START record
    char header[14];
    if (fread(header, 1, 14, in) < 14 || memcmp(header, "%SEMI-OASIS\r\n\x01", 14) != 0) {
        if (error_logger) fputs("[GDSTK] Invalid OASIS header found.\n", error_logger);
        fclose(in);
        return ErrorCode::InvalidFile;
    }

    // Process START record
    OasisStream s = {in};
    uint64_t len;
    uint8_t* version = oasis_read_string(s, false, len);
    if (memcmp(version, "1.0", 3) != 0) {
        if (error_logger) fputs("[GDSTK] Unsupported OASIS file version.\n", error_logger);
        free_allocation(version);
        return ErrorCode::InvalidFile;
    }
    free_allocation(version);

    precision = 1e-6 / oasis_read_real(s);
    fclose(in);
    return ErrorCode::NoError;
}

bool oas_validate(const char* filename, uint32_t* signature, ErrorCode* error_code) {
    uint8_t buffer[32 * 1024];
    FILE* in = fopen(filename, "rb");
    if (in == NULL) {
        if (error_logger) fputs("[GDSTK] Unable to open OASIS file for input.\n", error_logger);
        if (error_code) *error_code = ErrorCode::InputFileOpenError;
        return false;
    }

    // Check header bytes and START record
    char header[14];
    if (fread(header, 1, 14, in) < 14 || memcmp(header, "%SEMI-OASIS\r\n\x01", 14) != 0) {
        if (error_logger) fputs("[GDSTK] Invalid OASIS header found.\n", error_logger);
        if (error_code) *error_code = ErrorCode::InvalidFile;
        fclose(in);
        return false;
    }

    if (FSEEK64(in, -5, SEEK_END) != 0) {
        if (error_logger)
            fputs("[GDSTK] Unable to find the END record of the file.\n", error_logger);
        if (error_code) *error_code = ErrorCode::InvalidFile;
        fclose(in);
        return false;
    }

    uint64_t size = ftell(in) + 1;
    uint8_t file_sum[5];
    if (fread(file_sum, 1, COUNT(file_sum), in) < 5) {
        if (error_logger)
            fputs("[GDSTK] Unable to read the END record of the file.\n", error_logger);
        if (error_code) *error_code = ErrorCode::InvalidFile;
        fclose(in);
        return false;
    }

    if (file_sum[0] == 1) {
        // CRC32
        uint32_t sig = crc32(0, NULL, 0);
        FSEEK64(in, 0, SEEK_SET);
        while (size >= COUNT(buffer)) {
            if (fread(buffer, 1, COUNT(buffer), in) < COUNT(buffer)) {
                if (error_logger) fprintf(error_logger, "[GDSTK] Error reading file %s", filename);
                if (error_code) *error_code = ErrorCode::InvalidFile;
            }
            sig = crc32(sig, buffer, COUNT(buffer));
            size -= COUNT(buffer);
        }
        if (fread(buffer, 1, size, in) < size) {
            if (error_logger) fprintf(error_logger, "[GDSTK] Error reading file %s", filename);
            if (error_code) *error_code = ErrorCode::InvalidFile;
        }
        sig = crc32(sig, buffer, (unsigned int)size);
        little_endian_swap32(&sig, 1);
        if (signature) *signature = sig;
        // printf("CRC32: 0x%08X == 0x%08X\n", sig, *(uint32_t*)(file_sum + 1));
        if (sig != *(uint32_t*)(file_sum + 1)) return false;
    } else if (file_sum[0] == 2) {
        // Checksum32
        uint32_t sig = 0;
        FSEEK64(in, 0, SEEK_SET);
        while (size >= COUNT(buffer)) {
            if (fread(buffer, 1, COUNT(buffer), in) < COUNT(buffer)) {
                if (error_logger) fprintf(error_logger, "[GDSTK] Error reading file %s", filename);
                if (error_code) *error_code = ErrorCode::InvalidFile;
            }
            sig = checksum32(sig, buffer, COUNT(buffer));
            size -= COUNT(buffer);
        }
        if (fread(buffer, 1, size, in) < size) {
            if (error_logger) fprintf(error_logger, "[GDSTK] Error reading file %s", filename);
            if (error_code) *error_code = ErrorCode::InvalidFile;
        }
        sig = checksum32(sig, buffer, size);
        little_endian_swap32(&sig, 1);
        if (signature) *signature = sig;
        // printf("Checksum32: 0x%08X == 0x%08X\n", sig, *(uint32_t*)(file_sum + 1));
        if (sig != *(uint32_t*)(file_sum + 1)) return false;
    } else {
        // No checksum
        if (error_code) *error_code = ErrorCode::ChecksumError;
        if (signature) *signature = 0;
    }

    return true;
}

}  // namespace gdstk
