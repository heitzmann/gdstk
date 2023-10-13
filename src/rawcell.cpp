/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include <gdstk/allocator.hpp>
#include <gdstk/gdsii.hpp>
#include <gdstk/rawcell.hpp>

namespace gdstk {

void RawCell::print(bool all) const {
    if (source) {
        printf("RawCell <%p>, %s, size %" PRIu64 ", source offset %" PRIu64 ", owner <%p>\n", this,
               name, size, offset, owner);
    } else {
        printf("RawCell <%p>, %s, size %" PRIu64 ", data <%p>, owner <%p>\n", this, name, size,
               data, owner);
    }
    if (all) {
        printf("Dependencies (%" PRIu64 "/%" PRIu64 "):\n", dependencies.count,
               dependencies.capacity);
        for (uint64_t i = 0; i < dependencies.count; i++) {
            printf("Dependency %" PRIu64 "", i);
            dependencies[i]->print(false);
        }
    }
}

void RawCell::clear() {
    if (name) {
        free_allocation(name);
        name = NULL;
    }
    if (source) {
        source->uses--;
        if (source->uses == 0) {
            fclose(source->file);
            free_allocation(source);
        }
        source = NULL;
        offset = 0;
    } else if (data) {
        free_allocation(data);
        data = NULL;
    }
    size = 0;
    dependencies.clear();
}

void RawCell::get_dependencies(bool recursive, Map<RawCell*>& result) const {
    RawCell** r_item = dependencies.items;
    for (uint64_t i = 0; i < dependencies.count; i++) {
        RawCell* rawcell = *r_item++;
        if (recursive && result.get(rawcell->name) != rawcell) {
            rawcell->get_dependencies(true, result);
        }
        result.set(rawcell->name, rawcell);
    }
}

ErrorCode RawCell::to_gds(FILE* out) {
    ErrorCode error_code = ErrorCode::NoError;
    if (source) {
        uint64_t off = offset;
        data = (uint8_t*)allocate(size);
        int64_t result = source->offset_read(data, size, off);
        if (result < 0 || (uint64_t)result != size) {
            if (error_logger)
                fputs("[GDSTK] Unable to read RawCell data form input file.\n", error_logger);
            error_code = ErrorCode::InputFileError;
            size = 0;
        }
        source->uses--;
        if (source->uses == 0) {
            fclose(source->file);
            free_allocation(source);
        }
        source = NULL;
    }
    fwrite(data, 1, size, out);
    return error_code;
}

Map<RawCell*> read_rawcells(const char* filename, ErrorCode* error_code) {
    Map<RawCell*> result = {};
    uint8_t buffer[65537];
    char* str = (char*)(buffer + 4);

    RawSource* source = (RawSource*)allocate(sizeof(RawSource));
    source->uses = 0;
    source->file = fopen(filename, "rb");
    if (source->file == NULL) {
        if (error_logger) fputs("[GDSTK] Unable to open input GDSII file.\n", error_logger);
        if (error_code) *error_code = ErrorCode::InputFileOpenError;
        return result;
    }

    RawCell* rawcell = NULL;

    while (true) {
        uint64_t record_length = COUNT(buffer);
        ErrorCode err = gdsii_read_record(source->file, buffer, record_length);
        if (err != ErrorCode::NoError) {
            if (error_code) *error_code = err;
            break;
        }

        switch (buffer[2]) {
            case 0x04: {  // ENDLIB
                for (MapItem<RawCell*>* item = result.next(NULL); item; item = result.next(item)) {
                    Array<RawCell*>* dependencies = &item->value->dependencies;
                    for (uint64_t i = 0; i < dependencies->count;) {
                        char* name = (char*)((*dependencies)[i]);
                        rawcell = result.get(name);
                        if (rawcell) {
                            if (dependencies->contains(rawcell)) {
                                dependencies->remove_unordered(i);
                            } else {
                                (*dependencies)[i] = rawcell;
                                i++;
                            }
                        } else {
                            dependencies->remove_unordered(i);
                            if (error_logger)
                                fprintf(error_logger, "[GDSTK] Referenced cell %s not found.\n",
                                        name);
                            if (error_code) *error_code = ErrorCode::MissingReference;
                        }
                        free_allocation(name);
                    }
                }
                if (source->uses == 0) {
                    fclose(source->file);
                    free_allocation(source);
                }
                return result;
            } break;
            case 0x05:  // BGNSTR
                rawcell = (RawCell*)allocate_clear(sizeof(RawCell));
                rawcell->source = source;
                source->uses++;
                rawcell->offset = ftell(source->file) - record_length;
                rawcell->size = record_length;
                break;
            case 0x06:  // STRNAME
                if (rawcell) {
                    uint32_t data_length = (uint32_t)(record_length - 4);
                    if (str[data_length - 1] == 0) data_length--;
                    rawcell->name = (char*)allocate(data_length + 1);
                    memcpy(rawcell->name, str, data_length);
                    rawcell->name[data_length] = 0;
                    result.set(rawcell->name, rawcell);
                    rawcell->size += record_length;
                }
                break;
            case 0x07:  // ENDSTR
                if (rawcell) {
                    rawcell->size += record_length;
                    rawcell = NULL;
                }
                break;
            case 0x12:  // SNAME
                if (rawcell) {
                    uint32_t data_length = (uint32_t)(record_length - 4);
                    if (str[data_length - 1] == 0) data_length--;
                    char* name = (char*)allocate(data_length + 1);
                    memcpy(name, str, data_length);
                    name[data_length] = 0;
                    rawcell->dependencies.append((RawCell*)name);
                    rawcell->size += record_length;
                }
                break;
            default:
                if (rawcell) rawcell->size += record_length;
        }
    }

    source->uses++;  // ensure rawcell->clear() won't close and free source
    for (MapItem<RawCell*>* item = result.next(NULL); item; item = result.next(item)) {
        rawcell = item->value;
        Array<RawCell*>* dependencies = &rawcell->dependencies;
        for (uint64_t i = 0; i < dependencies->count;) {
            char* name = (char*)((*dependencies)[i]);
            free_allocation(name);
        }
        rawcell->clear();
    }
    fclose(source->file);
    free_allocation(source);
    result.clear();
    if (error_logger) fprintf(error_logger, "[GDSTK] Invalid GDSII file %s.\n", filename);
    if (error_code) *error_code = ErrorCode::InvalidFile;
    return result;
}

}  // namespace gdstk
