/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "rawcell.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>

namespace gdstk {

void RawCell::print(bool all) const {
    printf("RawCell <%p>, %s, size %" PRId64 ", data <%p>, owner <%p>\n", this, name, size, data,
           owner);
    if (all) {
        printf("Dependencies (%" PRId64 "/%" PRId64 "):\n", dependencies.size,
               dependencies.capacity);
        for (int64_t i = 0; i < dependencies.size; i++) {
            printf("(%" PRId64 ")", i);
            dependencies[i]->print(false);
        }
    }
}

void RawCell::clear() {
    if (name) {
        free(name);
        name = NULL;
    }
    if (data) {
        free(data);
        data = NULL;
    }
    size = 0;
    dependencies.clear();
}

void RawCell::get_dependencies(bool recursive, Array<RawCell*>& result) const {
    RawCell** item = dependencies.items;
    for (int64_t i = 0; i < dependencies.size; i++, item++) {
        if (result.index(*item) < 0) {
            result.append(*item);
            if (recursive) (*item)->get_dependencies(true, result);
        }
    }
}

void RawCell::to_gds(FILE* out) const { fwrite(data, sizeof(uint8_t), size, out); }

Map<RawCell*> read_rawcells(FILE* in) {
    int32_t record_length;
    uint8_t buffer[65537];
    char* str = (char*)(buffer + 4);

    Map<RawCell*> result = {0};
    RawCell* rawcell = NULL;

    while ((record_length = read_record(in, buffer)) > 0) {
        switch (buffer[2]) {
            case 0x04:  // ENDLIB
                for (MapItem<RawCell*>* item = result.next(NULL); item; item = result.next(item)) {
                    Array<RawCell*>* dependencies = &item->value->dependencies;
                    for (int64_t i = 0; i < dependencies->size;) {
                        char* name = (char*)((*dependencies)[i]);
                        rawcell = result.get(name);
                        if (rawcell) {
                            if (dependencies->index(rawcell) >= 0) {
                                dependencies->remove_unordered(i);
                            } else {
                                (*dependencies)[i] = rawcell;
                                i++;
                            }
                        } else {
                            dependencies->remove_unordered(i);
                            fprintf(stderr, "[GDSTK] Referenced cell %s not found.", name);
                        }
                        free(name);
                    }
                }
                return result;
                break;
            case 0x05:  // BGNSTR
                rawcell = (RawCell*)calloc(1, sizeof(RawCell));
                rawcell->size = record_length;
                break;
            case 0x06:  // STRNAME
                if (rawcell) {
                    int32_t data_length = record_length - 4;
                    if (str[data_length - 1] == 0) data_length--;
                    rawcell->name = (char*)malloc(sizeof(char) * (data_length + 1));
                    memcpy(rawcell->name, str, data_length);
                    rawcell->name[data_length] = 0;
                    result.set(rawcell->name, rawcell);
                    rawcell->size += record_length;
                }
                break;
            case 0x07:  // ENDSTR
                if (rawcell) {
                    rawcell->size += record_length;
                    if (fseek(in, -rawcell->size, SEEK_CUR) != 0) {
                        fputs(
                            "[GDSTK] Unable to rewind the position on the input file. No data will be imported into the RawCell.\n", stderr);
                        rawcell->size = 0;
                    }
                    rawcell->data = (uint8_t*)malloc(sizeof(uint8_t) * rawcell->size);
                    fread(rawcell->data, sizeof(uint8_t), rawcell->size, in);
                    rawcell = NULL;
                }
                break;
            case 0x12:  // SNAME
                if (rawcell) {
                    int32_t data_length = record_length - 4;
                    if (str[data_length - 1] == 0) data_length--;
                    char* name = (char*)malloc(sizeof(char) * (data_length + 1));
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

    return Map<RawCell*>{0};
}

}  // namespace gdstk
