/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "library.h"

#include <cfloat>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#include "cell.h"
#include "flexpath.h"
#include "label.h"
#include "map.h"
#include "polygon.h"
#include "rawcell.h"
#include "reference.h"
#include "utils.h"
#include "vec.h"

namespace gdstk {

void Library::print(bool all) const {
    printf("Library <%p> %s, unit %lg, precision %lg, %" PRId64 " cells, %" PRId64
           " raw cells, owner <%p>\n",
           this, name, unit, precision, cell_array.size, rawcell_array.size, owner);
    if (all) {
        for (int64_t i = 0; i < cell_array.size; i++) {
            printf("{%" PRId64 "} ", i);
            cell_array[i]->print(true);
        }
        for (int64_t i = 0; i < rawcell_array.size; i++) {
            printf("{%" PRId64 "} ", i);
            rawcell_array[i]->print(true);
        }
    }
}

void Library::copy_from(const Library& library, bool deep_copy) {
    name = (char*)malloc(sizeof(char) * (strlen(library.name) + 1));
    strcpy(name, library.name);
    unit = library.unit;
    precision = library.precision;
    if (deep_copy) {
        cell_array.capacity = library.cell_array.capacity;
        cell_array.size = library.cell_array.size;
        cell_array.items = (Cell**)malloc(sizeof(Cell*) * cell_array.capacity);
        Cell** src = library.cell_array.items;
        Cell** dst = cell_array.items;
        for (int64_t i = 0; i < library.cell_array.size; i++, src++, dst++) {
            *dst = (Cell*)malloc(sizeof(Cell));
            (*dst)->copy_from(**src, NULL, true);
        }
    } else {
        cell_array.copy_from(library.cell_array);
    }
    // raw cells should be immutable, so there's no need to perform a deep copy
    rawcell_array.copy_from(library.rawcell_array);
}

void Library::top_level(Array<Cell*>& top_cells, Array<RawCell*>& top_rawcells) const {
    Array<Cell*> cell_deps = {0};
    Array<RawCell*> rawcell_deps = {0};
    cell_deps.ensure_slots(cell_array.size * 2);
    cell_deps.ensure_slots(rawcell_array.size * 2);

    Cell** cell = cell_array.items;
    for (int64_t i = 0; i < cell_array.size; i++, cell++) {
        (*cell)->get_dependencies(false, cell_deps);
        (*cell)->get_raw_dependencies(false, rawcell_deps);
    }

    RawCell** rawcell = rawcell_array.items;
    for (int64_t i = 0; i < rawcell_array.size; i++, rawcell++)
        (*rawcell)->get_dependencies(false, rawcell_deps);

    cell = cell_array.items;
    for (int64_t i = 0; i < cell_array.size; i++, cell++)
        if (cell_deps.index(*cell) < 0) top_cells.append(*cell);

    rawcell = rawcell_array.items;
    for (int64_t i = 0; i < rawcell_array.size; i++, rawcell++)
        if (rawcell_deps.index(*rawcell) < 0) top_rawcells.append(*rawcell);
}

void Library::write_gds(FILE* out, int64_t max_points, std::tm* timestamp) const {
    int64_t len = strlen(name);
    if (len % 2) len++;
    if (!timestamp) {
        time_t now = time(NULL);
        timestamp = localtime(&now);
    }
    uint16_t buffer_start[] = {6,
                               0x0002,
                               0x0258,
                               28,
                               0x0102,
                               (uint16_t)(timestamp->tm_year + 1900),
                               (uint16_t)(timestamp->tm_mon + 1),
                               (uint16_t)timestamp->tm_mday,
                               (uint16_t)timestamp->tm_hour,
                               (uint16_t)timestamp->tm_min,
                               (uint16_t)timestamp->tm_sec,
                               (uint16_t)(timestamp->tm_year + 1900),
                               (uint16_t)(timestamp->tm_mon + 1),
                               (uint16_t)timestamp->tm_mday,
                               (uint16_t)timestamp->tm_hour,
                               (uint16_t)timestamp->tm_min,
                               (uint16_t)timestamp->tm_sec,
                               (uint16_t)(4 + len),
                               0x0206};
    swap16(buffer_start, COUNT(buffer_start));
    fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);
    fwrite(name, sizeof(char), len, out);

    uint16_t buffer_units[] = {20, 0x0305};
    swap16(buffer_units, COUNT(buffer_units));
    fwrite(buffer_units, sizeof(uint16_t), COUNT(buffer_units), out);
    uint64_t units[] = {gdsii_real_from_double(precision / unit),
                        gdsii_real_from_double(precision)};
    swap64(units, COUNT(units));
    fwrite(units, sizeof(uint64_t), COUNT(units), out);

    double scaling = unit / precision;
    Cell** cell = cell_array.items;
    for (int64_t i = 0; i < cell_array.size; i++, cell++)
        (*cell)->to_gds(out, scaling, max_points, precision, timestamp);

    RawCell** rawcell = rawcell_array.items;
    for (int64_t i = 0; i < rawcell_array.size; i++, rawcell++) (*rawcell)->to_gds(out);

    uint16_t buffer_end[] = {4, 0x0400};
    swap16(buffer_end, COUNT(buffer_end));
    fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
}

Library read_gds(FILE* in, double unit) {
    const char record_names[][13] = {
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

    Library library = {0};
    // One extra char in case we need a 0-terminated string with max size (should never happen, but
    // it doesn't hurt to be prepared).
    uint8_t buffer[65537];
    int16_t* data16 = (int16_t*)(buffer + 4);
    int32_t* data32 = (int32_t*)(buffer + 4);
    uint64_t* data64 = (uint64_t*)(buffer + 4);
    char* str = (char*)(buffer + 4);
    int32_t record_length;

    Cell* cell = NULL;
    Polygon* polygon = NULL;
    FlexPath* path = NULL;
    Reference* reference = NULL;
    Label* label = NULL;

    double factor = 1;
    double width = 0;
    int16_t key = 0;

    while ((record_length = read_record(in, buffer)) > 0) {
        int32_t data_length;

        // printf("%02X %s (%d bytes)", buffer[2], record_names[buffer[2]], record_length);

        switch (buffer[3]) {
            case 1:
            case 2:
                data_length = (record_length - 4) / 2;
                swap16((uint16_t*)data16, data_length);
                // for (int64_t i = 0; i < data_length; i++) printf(" %hd", data16[i]);
                break;
            case 3:
            case 4:
                data_length = (record_length - 4) / 4;
                swap32((uint32_t*)data32, data_length);
                // for (int64_t i = 0; i < data_length; i++) printf(" %d", data32[i]);
                break;
            case 5:
                data_length = (record_length - 4) / 8;
                swap64(data64, data_length);
                // for (int64_t i = 0; i < data_length; i++) printf(" %lx", data64[i]);
                break;
            default:
                data_length = record_length - 4;
                // for (int64_t i = 0; i < data_length; i++) printf(" %c", str[i]);
        }

        // printf("\n");

        switch (buffer[2]) {
            case 0x00:  // HEADER
            case 0x01:  // BGNLIB
            case 0x07:  // ENDSTR
                break;
            case 0x02:  // LIBNAME
                if (str[data_length - 1] == 0) data_length--;
                library.name = (char*)malloc(sizeof(char) * (data_length + 1));
                memcpy(library.name, str, data_length);
                library.name[data_length] = 0;
                break;
            case 0x03: {  // UNITS
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
            } break;
            case 0x04: {  // ENDLIB
                Map<Cell*> map = {0};
                map.resize((int64_t)(1.0 + 10.0 / MAP_CAP * library.cell_array.size));
                for (int64_t i = library.cell_array.size - 1; i >= 0; i--)
                    map.set(library.cell_array[i]->name, library.cell_array[i]);
                for (int64_t i = library.cell_array.size - 1; i >= 0; i--) {
                    Reference** ref = library.cell_array[i]->reference_array.items;
                    for (int64_t j = library.cell_array[i]->reference_array.size; j > 0;
                         j--, ref++) {
                        reference = *ref;
                        Cell* cp = map.get(reference->name);
                        if (cp) {
                            free(reference->name);
                            reference->type = ReferenceType::Cell;
                            reference->cell = cp;
                        }
                    }
                }
                map.clear();
                return library;
            } break;
            case 0x05:  // BGNSTR
                cell = (Cell*)calloc(1, sizeof(Cell));
                break;
            case 0x06:  // STRNAME
                if (cell) {
                    if (str[data_length - 1] == 0) data_length--;
                    cell->name = (char*)malloc(sizeof(char) * (data_length + 1));
                    memcpy(cell->name, str, data_length);
                    cell->name[data_length] = 0;
                    library.cell_array.append(cell);
                }
                break;
            case 0x08:  // BOUNDARY
            case 0x2D:  // BOX
                polygon = (Polygon*)calloc(1, sizeof(Polygon));
                if (cell) cell->polygon_array.append(polygon);
                break;
            case 0x09:  // PATH
                path = (FlexPath*)calloc(1, sizeof(FlexPath));
                path->num_elements = 1;
                path->elements = (FlexPathElement*)calloc(1, sizeof(FlexPathElement));
                path->gdsii_path = true;
                if (cell) cell->flexpath_array.append(path);
                break;
            case 0x0A:  // SREF
            case 0x0B:  // AREF
                reference = (Reference*)calloc(1, sizeof(Reference));
                reference->magnification = 1;
                reference->columns = 1;
                reference->rows = 1;
                if (cell) cell->reference_array.append(reference);
                break;
            case 0x0C:  // TEXT
                label = (Label*)calloc(1, sizeof(Label));
                if (cell) cell->label_array.append(label);
                break;
            case 0x0D:  // LAYER
                if (polygon)
                    polygon->layer = data16[0];
                else if (path)
                    path->elements[0].layer = data16[0];
                else if (label)
                    label->layer = data16[0];
                break;
            case 0x0E:  // DATATYPE
            case 0x2E:  // BOXTYPE
                if (polygon)
                    polygon->datatype = data16[0];
                else if (path)
                    path->elements[0].datatype = data16[0];
                break;
            case 0x0F:  // WIDTH
                if (data32[0] < 0) {
                    width = factor * -data32[0];
                    if (path) path->scale_width = false;
                } else {
                    width = factor * data32[0];
                    if (path) path->scale_width = true;
                }
                break;
            case 0x10:  // XY
                if (polygon) {
                    polygon->point_array.ensure_slots(data_length / 2);
                    double* d = (double*)polygon->point_array.items + polygon->point_array.size;
                    int32_t* s = data32;
                    for (int64_t i = data_length; i > 0; i--) *d++ = factor * (*s++);
                    polygon->point_array.size += data_length / 2;
                } else if (path) {
                    Array<Vec2> point_array = {0};
                    if (path->spine.point_array.size == 0) {
                        double offset = 0;
                        Vec2 initial_position = Vec2{factor * data32[0], factor * data32[1]};
                        path->init(initial_position, &width, &offset);
                        point_array.ensure_slots(data_length / 2 - 1);
                        double* d = (double*)point_array.items;
                        int32_t* s = data32 + 2;
                        for (int64_t i = data_length - 2; i > 0; i--) *d++ = factor * (*s++);
                        point_array.size += data_length / 2 - 1;
                    } else {
                        point_array.ensure_slots(data_length / 2);
                        double* d = (double*)point_array.items;
                        int32_t* s = data32;
                        for (int64_t i = data_length; i > 0; i--) *d++ = factor * (*s++);
                        point_array.size += data_length / 2;
                    }
                    path->segment(point_array, NULL, NULL, false);
                    point_array.clear();
                } else if (reference) {
                    Vec2 origin = Vec2{factor * data32[0], factor * data32[1]};
                    reference->origin = origin;
                    if (reference->columns > 1 || reference->rows > 1) {
                        double sa = -sin(reference->rotation);
                        double ca = cos(reference->rotation);
                        double x2 = ((factor * data32[2] - origin.x) * ca -
                                     (factor * data32[3] - origin.y) * sa + origin.x);
                        double y3 = ((factor * data32[4] - origin.x) * sa +
                                     (factor * data32[5] - origin.y) * ca + origin.y);
                        if (reference->x_reflection) y3 = 2 * factor * data32[1] - y3;
                        reference->spacing.x = (x2 - origin.x) / reference->columns;
                        reference->spacing.y = (y3 - origin.y) / reference->rows;
                    }
                } else if (label) {
                    label->origin.x = factor * data32[0];
                    label->origin.y = factor * data32[1];
                }
                break;
            case 0x11:  // ENDEL
                if (polygon) {
                    // Polygons are closed in GDSII (first and last points are the same)
                    polygon->point_array.size--;
                    polygon = NULL;
                }
                path = NULL;
                reference = NULL;
                label = NULL;
                break;
            case 0x12: {  // SNAME
                if (reference) {
                    if (str[data_length - 1] == 0) data_length--;
                    reference->name = (char*)malloc(sizeof(char) * (data_length + 1));
                    memcpy(reference->name, str, data_length);
                    reference->name[data_length] = 0;
                    reference->type = ReferenceType::Name;
                }
            } break;
            case 0x13:  // COLROW
                if (reference) {
                    reference->columns = data16[0];
                    reference->rows = data16[1];
                }
                break;
            case 0x16:  // TEXTTYPE
                if (label) label->texttype = data16[0];
                break;
            case 0x17:  // PRESENTATION
                if (label) label->anchor = (Anchor)(data16[0] & 0x000F);
                break;
            case 0x19:  // STRING
                if (label) {
                    if (str[data_length - 1] == 0) data_length--;
                    label->text = (char*)malloc(sizeof(char) * (data_length + 1));
                    memcpy(label->text, str, data_length);
                    label->text[data_length] = 0;
                }
                break;
            case 0x1A:  // STRANS
                if (reference)
                    reference->x_reflection = (data16[0] & 0x8000) != 0;
                else if (label)
                    label->x_reflection = (data16[0] & 0x8000) != 0;
                if (data16[0] & 0x0006)
                    fputs("[GDSTK] Absolute magnification and rotation of references is not supported.\n", stderr);
                break;
            case 0x1B:  // MAG
                if (reference)
                    reference->magnification = gdsii_real_to_double(data64[0]);
                else if (label)
                    label->magnification = gdsii_real_to_double(data64[0]);
                break;
            case 0x1C:  // ANGLE
                if (reference)
                    reference->rotation = M_PI / 180.0 * gdsii_real_to_double(data64[0]);
                else if (label)
                    label->rotation = M_PI / 180.0 * gdsii_real_to_double(data64[0]);
                break;
            case 0x21:  // PATHTYPE
                if (path) {
                    if (data16[0] == 0)
                        path->elements[0].end_type = EndType::Flush;
                    else if (data16[0] == 1)
                        path->elements[0].end_type = EndType::Round;
                    else
                        path->elements[0].end_type = EndType::Extended;
                }
                break;
            case 0x2B:  // PROPATTR
                key = data16[0];
                break;
            case 0x2C:  // PROPVALUE
                if (str[data_length - 1] != 0) str[data_length++] = 0;
                if (polygon) {
                    set_property(polygon->properties, key, str);
                } else if (path) {
                    set_property(path->properties, key, str);
                } else if (reference) {
                    set_property(reference->properties, key, str);
                } else if (label) {
                    set_property(label->properties, key, str);
                }
                break;
            case 0x30:  // BGNEXTN
                if (path) path->elements[0].end_extensions.u = factor * data32[0];
                break;
            case 0x31:  // ENDEXTN
                if (path) path->elements[0].end_extensions.v = factor * data32[0];
                break;
            // case 0x14:  // TEXTNODE
            // case 0x15:  // NODE
            // case 0x18:  // SPACING
            // case 0x1D:  // UINTEGER
            // case 0x1E:  // USTRING
            // case 0x1F:  // REFLIBS
            // case 0x20:  // FONTS
            // case 0x22:  // GENERATIONS
            // case 0x23:  // ATTRTABLE
            // case 0x24:  // STYPTABLE
            // case 0x25:  // STRTYPE
            // case 0x26:  // ELFLAGS
            // case 0x27:  // ELKEY
            // case 0x28:  // LINKTYPE
            // case 0x29:  // LINKKEYS
            // case 0x2A:  // NODETYPE
            // case 0x2F:  // PLEX
            // case 0x32:  // TAPENUM
            // case 0x33:  // TAPECODE
            // case 0x34:  // STRCLASS
            // case 0x35:  // RESERVED
            // case 0x36:  // FORMAT
            // case 0x37:  // MASK
            // case 0x38:  // ENDMASKS
            // case 0x39:  // LIBDIRSIZE
            // case 0x3A:  // SRFNAME
            // case 0x3B:  // LIBSECUR
            default:
                if (buffer[2] < COUNT(record_names))
                    fprintf(stderr, "[GDSTK] Record type %s (0x%02X) is not supported.\n",
                            record_names[buffer[2]], buffer[2]);
                else
                    fprintf(stderr, "[GDSTK] Unknown record type 0x%02X.\n", buffer[2]);
        }
    }
    return Library{0};
}

int gds_units(FILE* in, double& unit, double& precision) {
    uint8_t buffer[65537];
    uint64_t* data64 = (uint64_t*)(buffer + 4);
    while (read_record(in, buffer) > 0) {
        if (buffer[2] == 0x03) {  // UNITS
            swap64(data64, 2);
            precision = gdsii_real_to_double(data64[1]);
            unit = precision / gdsii_real_to_double(data64[0]);
            return 0;
        }
    }
    fputs("[GDSTK] GDSII file missing units definition.\n", stderr);
    return -1;
}

}  // namespace gdstk
