/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "property.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "utils.h"

namespace gdstk {

void properties_clear(Property* properties) {
    while (properties) {
        Property* next = properties->next;
        free(properties->value);
        free(properties);
        properties = next;
    }
}

static void value_copy(const char* value, Property* dst) {
    int64_t len = strlen(value) + 1;
    dst->value = (char*)malloc(sizeof(char) * len);
    memcpy(dst->value, value, len);
}

Property* properties_copy(const Property* properties) {
    Property* result = NULL;
    Property* dst;
    for (; properties; properties = properties->next) {
        if (result == NULL) {
            result = (Property*)malloc(sizeof(Property));
            dst = result;
        } else {
            dst->next = (Property*)malloc(sizeof(Property));
            dst = dst->next;
        }
        dst->key = properties->key;
        value_copy(properties->value, dst);
        dst->next = NULL;
    }
    return result;
}

void set_property(Property*& properties, int16_t key, const char* value) {
    Property root = {0, NULL, properties};
    properties = &root;
    while (properties->next && properties->next->key < key) properties = properties->next;
    if (properties->next == NULL || properties->next->key > key) {
        Property* p = (Property*)malloc(sizeof(Property));
        p->key = key;
        value_copy(value, p);
        p->next = properties->next;
        properties->next = p;
    } else {
        // Same key: replace value
        properties = properties->next;
        free(properties->value);
        value_copy(value, properties);
    }
    properties = root.next;
}

const char* get_property(const Property* properties, int16_t key) {
    Property root = {0, NULL, (Property*)properties};
    properties = &root;
    while (properties->next && properties->next->key < key) properties = properties->next;
    if (properties->next == NULL || properties->next->key > key) return NULL;
    return properties->next->value;
}

void delete_property(Property*& properties, int16_t key) {
    Property root = {0, NULL, properties};
    properties = &root;
    while (properties->next && properties->next->key < key) properties = properties->next;
    if (properties->next && properties->next->key == key) {
        Property* p = properties->next;
        properties->next = p->next;
        free(p->value);
        free(p);
    }
    properties = root.next;
}

void properties_to_gds(const Property* properties, FILE* out) {
    uint64_t size = 0;
    for (; properties; properties = properties->next) {
        uint16_t len = strlen(properties->value);
        if (len % 2) len++;
        uint16_t buffer_prop[] = {6, 0x2B02, (uint16_t)properties->key, (uint16_t)(4 + len),
                                  0x2C06};
        size += len;
        swap16(buffer_prop, COUNT(buffer_prop));
        fwrite(buffer_prop, sizeof(uint16_t), COUNT(buffer_prop), out);
        fwrite(properties->value, sizeof(char), len, out);
    }
    if (size > 128)
        fputs(
            "[GDSTK] Properties with size larger than 128 bytes are not officially supported by the GDSII specification.  This file might not be compatible with all readers.\n", stderr);
}

}  // namespace gdstk
