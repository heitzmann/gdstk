/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "style.h"

#include <assert.h>
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

void StyleMap::print(bool all) const {
    printf("StyleMap <%p>, count %" PRIu64 "/%" PRIu64 ", items <%p>\n", this, count, capacity,
           items);
    if (all) {
        Style* item = items;
        for (uint64_t i = 0; i < capacity; i++, item++) {
            printf("(%" PRIu64 ") Item <%p>, layer %" PRIu32 "/type %" PRIu32
                   ", value <%p> \"%s\"\n",
                   i, item, get_layer(item->tag), get_type(item->tag), item->value,
                   item->value ? item->value : "");
        }
    }
}

void StyleMap::clear() {
    if (items) {
        Style* item = items;
        for (uint64_t i = 0; i < capacity; i++, item++) {
            if (item->value) free_allocation(item->value);
            item->value = NULL;
        }
    }
    free_allocation(items);
    items = NULL;
    capacity = 0;
    count = 0;
}

void StyleMap::copy_from(const StyleMap& map) {
    count = 0;
    capacity = map.capacity;
    items = (Style*)allocate_clear(capacity * sizeof(Style));
    for (Style* s = map.next(NULL); s; s = map.next(s)) set(s->tag, s->value);
}

void StyleMap::resize(uint64_t new_capacity) {
    StyleMap new_map;
    new_map.count = 0;
    new_map.capacity = new_capacity;
    new_map.items = (Style*)allocate_clear(new_capacity * sizeof(Style));
    const Style* limit = items + capacity;
    for (Style* it = items; it != limit; it++) {
        if (it->value) new_map.set(it->tag, it->value);
    }
    clear();
    capacity = new_map.capacity;
    count = new_map.count;
    items = new_map.items;
}

Style* StyleMap::get_slot(Tag tag) const {
    assert(capacity > 0);
    uint64_t h = hash(tag) % capacity;
    Style* item = items + h;
    while (item->value != NULL && item->tag != tag) {
        item++;
        if (item == items + capacity) item = items;
    }
    return item;
}

void StyleMap::set(Tag tag, const char* value) {
    // Equallity is important for capacity == 0
    if (count * 10 >= capacity * MAP_CAPACITY_THRESHOLD)
        resize(capacity >= INITIAL_MAP_CAPACITY ? capacity * MAP_GROWTH_FACTOR
                                                : INITIAL_MAP_CAPACITY);

    uint64_t len;
    Style* s = get_slot(tag);
    s->tag = tag;
    if (value) {
        if (s->value != NULL) {
            free_allocation(s->value);
        }
        s->value = copy_string(value, len);
    } else if (s->value == NULL) {
        s->value = copy_string(default_style(tag), len);
    }
}

const char* StyleMap::get(Tag tag) const {
    if (count == 0) return default_style(tag);
    const Style* s = get_slot(tag);
    return s->value ? s->value : default_style(tag);
}

bool StyleMap::del(Tag tag) {
    if (count == 0) return false;

    Style* item = get_slot(tag);
    if (item->value == NULL) return false;

    free_allocation(item->value);
    item->value = NULL;
    count--;

    // Re-insert this block to fill any undesired gaps
    while (true) {
        item++;
        if (item == items + capacity) item = items;
        if (item->value == NULL) return true;
        char* temp_value = item->value;
        item->value = NULL;
        Style* new_item = get_slot(item->tag);
        new_item->tag = item->tag;
        new_item->value = temp_value;
    }
    assert(false);
    return true;
}

Style* StyleMap::next(const Style* current) const {
    Style* next = current ? (Style*)(current + 1) : items;
    const Style* limit = items + capacity;
    while (next < limit) {
        if (next->value) return next;
        next++;
    }
    return NULL;
}

}  // namespace gdstk
