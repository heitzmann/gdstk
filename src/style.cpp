/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <gdstk/allocator.hpp>
#include <gdstk/map.hpp>
#include <gdstk/style.hpp>

namespace gdstk {

void StyleMap::print(bool all) const {
    printf("StyleMap <%p>, count %" PRIu64 "/%" PRIu64 ", items <%p>\n", this, count, capacity,
           items);
    if (all) {
        Style* item = items;
        for (uint64_t i = 0; i < capacity; i++, item++) {
            printf("Item[%" PRIu64 "]: tag %" PRIu32 "/%" PRIu32 ", value <%p> \"%s\"\n", i,
                   get_layer(item->tag), get_type(item->tag), item->value,
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
    assert(value);
    // Equality is important for capacity == 0
    if (count * 10 >= capacity * GDSTK_MAP_CAPACITY_THRESHOLD)
        resize(capacity >= GDSTK_INITIAL_MAP_CAPACITY ? capacity * GDSTK_MAP_GROWTH_FACTOR
                                                      : GDSTK_INITIAL_MAP_CAPACITY);

    Style* s = get_slot(tag);
    s->tag = tag;
    if (s->value == NULL) {
        count++;
    } else {
        free_allocation(s->value);
    }
    s->value = copy_string(value, NULL);
}

const char* StyleMap::get(Tag tag) const {
    if (count == 0) return NULL;
    const Style* s = get_slot(tag);
    return s->value;
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
    Style* next_ = current ? (Style*)(current + 1) : items;
    const Style* limit = items + capacity;
    while (next_ < limit) {
        if (next_->value) return next_;
        next_++;
    }
    return NULL;
}

}  // namespace gdstk
