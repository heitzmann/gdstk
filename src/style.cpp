/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "style.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "utils.h"

namespace gdstk {

// Kenneth Kelly's 22 colors of maximum contrast (minus B/W: "F2F3F4", "222222")
const char* colors[] = {"F3C300", "875692", "F38400", "A1CAF1", "BE0032", "C2B280", "848482",
                        "008856", "E68FAC", "0067A5", "F99379", "604E97", "F6A600", "B3446C",
                        "DCD300", "882D17", "8DB600", "654522", "E25822", "2B3D26"};

static const char* default_style(int16_t layer, int16_t type) {
    static char buffer[] = "stroke: #XXXXXX; fill: #XXXXXX; fill-opacity: 0.5;";
    const char* c = colors[HASH2(layer, type) % COUNT(colors)];
    memcpy(buffer + 9, c, sizeof(char) * 6);
    memcpy(buffer + 24, c, sizeof(char) * 6);
    return buffer;
}

void StyleMap::clear() {
    if (style) {
        for (int64_t i = 0; i < capacity; i++) {
            Style* s = style[i].next;
            while (s) {
                Style* tmp = s->next;
                free(s->value);
                free(s);
                s = tmp;
            }
        }
        free(style);
        style = NULL;
    }
    capacity = 0;
    size = 0;
}

void StyleMap::copy_from(const StyleMap& map) {
    size = 0;
    capacity = map.capacity;
    style = (Style*)calloc(capacity, sizeof(Style));
    for (Style* s = map.next(NULL); s; s = map.next(s)) set(s->layer, s->type, s->value);
}

void StyleMap::resize(int64_t new_capacity) {
    StyleMap new_map;
    new_map.capacity = new_capacity;
    new_map.style = (Style*)calloc(new_capacity, sizeof(Style));
    for (int64_t i = 0; i < capacity; i++) {
        Style* prop = style[i].next;
        while (prop != NULL) {
            int64_t j = HASH2(prop->layer, prop->type) % new_map.capacity;
            Style* slot = new_map.style + j;
            while (slot->next != NULL) slot = slot->next;
            slot->next = prop;
            slot = prop->next;
            prop->next = NULL;
            prop = slot;
        }
    }
    free(style);
    style = new_map.style;
    capacity = new_map.capacity;
}

void StyleMap::set(int16_t layer, int16_t type, const char* value) {
    // Equallity is important for capacity == 0
    if (size * 10 >= capacity * MAP_CAP) resize(capacity > 0 ? 2 * capacity : 4);

    int64_t idx = HASH2(layer, type) % capacity;
    Style* s = style + idx;
    while (!(s->next == NULL || (s->next->layer == layer && s->next->type == type))) s = s->next;
    if (!s->next) {
        s->next = (Style*)malloc(sizeof(Style));
        s->next->layer = layer;
        s->next->type = type;
        s->next->value = NULL;
        s->next->next = NULL;
        size++;
    }
    s = s->next;

    if (s->value) {
        if (!value) return;
        free(s->value);
    }
    if (value) {
        int64_t len = strlen(value) + 1;
        s->value = (char*)malloc(sizeof(char) * len);
        memcpy(s->value, value, len);
    } else {
        const char* default_value = default_style(layer, type);
        int64_t len = strlen(default_value) + 1;
        s->value = (char*)malloc(sizeof(char) * len);
        memcpy(s->value, default_value, len);
    }
}

const char* StyleMap::get(int16_t layer, int16_t type) const {
    if (size == 0) return default_style(layer, type);
    int64_t i = HASH2(layer, type) % capacity;
    Style* s = style[i].next;
    while (!(s == NULL || (s->layer == layer && s->type == type))) s = s->next;

    if (s) return s->value;
    return default_style(layer, type);
}

void StyleMap::del(int16_t layer, int16_t type) {
    if (size == 0) return;

    int64_t i = HASH2(layer, type) % capacity;
    Style* s = style + i;
    while (!(s->next == NULL || (s->next->layer == layer && s->next->type == type))) s = s->next;
    if (!s->next) return;

    Style* rem = s->next;
    s->next = rem->next;
    if (rem->value) free(rem->value);
    free(rem);
    size--;
}

Style* StyleMap::next(const Style* current) const {
    if (!current) {
        for (int64_t i = 0; i < capacity; i++)
            if (style[i].next) return style[i].next;
        return NULL;
    }
    if (current->next) return current->next;
    for (int64_t i = HASH2(current->layer, current->type) % capacity + 1; i < capacity; i++)
        if (style[i].next) return style[i].next;
    return NULL;
}

}  // namespace gdstk
