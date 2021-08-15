/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "style.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "allocator.h"
#include "map.h"

namespace gdstk {

// FNV-1a hash function (64 bits)
#define STYLE_FNV_PRIME 0x00000100000001b3
#define STYLE_FNV_OFFSET 0xcbf29ce484222325
inline static uint64_t hash(Tag tag) {
    uint64_t result = STYLE_FNV_OFFSET;
    uint8_t* byte = (uint8_t*)(&tag);
    for (unsigned i = sizeof(Tag); i > 0; i--) {
        result ^= *byte++;
        result *= STYLE_FNV_PRIME;
    }
    return result;
}

// Kenneth Kelly's 22 colors of maximum contrast (minus B/W: "F2F3F4", "222222")
const char* colors[] = {"F3C300", "875692", "F38400", "A1CAF1", "BE0032", "C2B280", "848482",
                        "008856", "E68FAC", "0067A5", "F99379", "604E97", "F6A600", "B3446C",
                        "DCD300", "882D17", "8DB600", "654522", "E25822", "2B3D26"};

static const char* default_style(Tag tag) {
    static char buffer[] = "stroke: #XXXXXX; fill: #XXXXXX; fill-opacity: 0.5;";
    const char* c = colors[(2 + get_layer(tag) + get_type(tag) * 13) % COUNT(colors)];
    memcpy(buffer + 9, c, 6);
    memcpy(buffer + 24, c, 6);
    return buffer;
}

void StyleMap::clear() {
    if (style) {
        for (uint64_t i = 0; i < capacity; i++) {
            Style* s = style[i].next;
            while (s) {
                Style* tmp = s->next;
                free_allocation(s->value);
                free_allocation(s);
                s = tmp;
            }
        }
        free_allocation(style);
        style = NULL;
    }
    capacity = 0;
    count = 0;
}

void StyleMap::copy_from(const StyleMap& map) {
    count = 0;
    capacity = map.capacity;
    style = (Style*)allocate_clear(capacity * sizeof(Style));
    for (Style* s = map.next(NULL); s; s = map.next(s)) set(s->tag, s->value);
}

void StyleMap::resize(uint64_t new_capacity) {
    StyleMap new_map;
    new_map.capacity = new_capacity;
    new_map.style = (Style*)allocate_clear(new_capacity * sizeof(Style));
    for (uint64_t i = 0; i < capacity; i++) {
        Style* prop = style[i].next;
        while (prop != NULL) {
            uint64_t j = hash(prop->tag) % new_map.capacity;
            Style* slot = new_map.style + j;
            while (slot->next != NULL) slot = slot->next;
            slot->next = prop;
            slot = prop->next;
            prop->next = NULL;
            prop = slot;
        }
    }
    free_allocation(style);
    style = new_map.style;
    capacity = new_map.capacity;
}

void StyleMap::set(Tag tag, const char* value) {
    // Equallity is important for capacity == 0
    if (count * 10 >= capacity * MAP_CAPACITY_THRESHOLD)
        resize(capacity >= INITIAL_MAP_CAPACITY ? capacity * MAP_GROWTH_FACTOR
                                                : INITIAL_MAP_CAPACITY);

    uint64_t idx = hash(tag) % capacity;
    Style* s = style + idx;
    while (!(s->next == NULL || s->next->tag == tag)) s = s->next;
    if (!s->next) {
        s->next = (Style*)allocate(sizeof(Style));
        s->next->tag = tag;
        s->next->value = NULL;
        s->next->next = NULL;
        count++;
    }
    s = s->next;

    if (s->value) {
        if (!value) return;
        free_allocation(s->value);
    }

    uint64_t len;
    if (value) {
        s->value = copy_string(value, len);
    } else {
        s->value = copy_string(default_style(tag), len);
    }
}

const char* StyleMap::get(Tag tag) const {
    if (count == 0) return default_style(tag);
    uint64_t i = hash(tag) % capacity;
    Style* s = style[i].next;
    while (!(s == NULL || s->tag == tag)) s = s->next;

    if (s) return s->value;
    return default_style(tag);
}

void StyleMap::del(Tag tag) {
    if (count == 0) return;

    uint64_t i = hash(tag) % capacity;
    Style* s = style + i;
    while (!(s->next == NULL || s->next->tag == tag)) s = s->next;
    if (!s->next) return;

    Style* rem = s->next;
    s->next = rem->next;
    if (rem->value) free_allocation(rem->value);
    free_allocation(rem);
    count--;
}

Style* StyleMap::next(const Style* current) const {
    if (!current) {
        for (uint64_t i = 0; i < capacity; i++)
            if (style[i].next) return style[i].next;
        return NULL;
    }
    if (current->next) return current->next;
    for (uint64_t i = hash(current->tag) % capacity + 1; i < capacity; i++)
        if (style[i].next) return style[i].next;
    return NULL;
}

}  // namespace gdstk
