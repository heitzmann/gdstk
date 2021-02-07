/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "rawcell.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "allocator.h"
#include "gdsii.h"

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
            printf("(%" PRIu64 ")", i);
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

void RawCell::to_gds(FILE* out) {
    if (source) {
        uint64_t off = offset;
        data = (uint8_t*)allocate(size);
        int64_t result = source->offset_read(data, size, off);
        if (result < 0 || (uint64_t)result != size) {
            fputs("[GDSTK] Unable to read RawCell data form input file.\n", stderr);
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
}

Map<RawCell*> read_rawcells(const char* filename) {
    uint32_t record_length;
    uint8_t buffer[65537];
    char* str = (char*)(buffer + 4);

    Map<RawCell*> result = {0};
    RawCell* rawcell = NULL;

    RawSource* source = (RawSource*)allocate(sizeof(RawSource));
    source->file = fopen(filename, "rb");
    source->uses = 0;
    if (source->file == NULL) {
        fputs("[GDSTK] Unable to open input GDSII file.\n", stderr);
        return result;
    }

    while ((record_length = gdsii_read_record(source->file, buffer)) > 0) {
        switch (buffer[2]) {
            case 0x04:  // ENDLIB
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
                            fprintf(stderr, "[GDSTK] Referenced cell %s not found.", name);
                        }
                        free_allocation(name);
                    }
                }
                if (source->uses == 0) {
                    fclose(source->file);
                    free_allocation(source);
                }
                return result;
                break;
            case 0x05:  // BGNSTR
                rawcell = (RawCell*)allocate_clear(sizeof(RawCell));
                rawcell->source = source;
                source->uses++;
                rawcell->offset = ftell(source->file) - record_length;
                rawcell->size = record_length;
                break;
            case 0x06:  // STRNAME
                if (rawcell) {
                    uint32_t data_length = record_length - 4;
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
                    uint32_t data_length = record_length - 4;
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

    if (source->uses == 0) {
        fclose(source->file);
        free_allocation(source);
    }
    return Map<RawCell*>{0};
}

}  // namespace gdstk
