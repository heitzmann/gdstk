/*
Copyright 2023 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_TAGMAP
#define GDSTK_HEADER_TAGMAP

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <assert.h>

#include "allocator.hpp"
#include "utils.hpp"

namespace gdstk {

struct TagMapItem {
    Tag key;
    Tag value;
};

// Hash map between tags.  Items with the value equal to the key are ignored.
struct TagMap {
    uint64_t capacity;  // allocated capacity
    uint64_t count;     // number of items in the map
    TagMapItem* items;  // array with length capacity

    void print(bool all) const {
        printf("TagMap <%p>, count %" PRIu64 "/%" PRIu64 ", items <%p>\n", this, count, capacity,
               items);
        if (all) {
            TagMapItem* item = items;
            for (uint64_t i = 0; i < capacity; i++, item++) {
                printf("Item %" PRIu64 " <%p>, key (%" PRIu32 ", %" PRIu32 "): value (%" PRIu32
                       ", %" PRIu32 ")\n",
                       i, item, get_layer(item->key), get_type(item->key), get_layer(item->value),
                       get_type(item->value));
            }
        }
    }

    // The instance should be zeroed before using copy_from
    void copy_from(const TagMap& map) {
        count = 0;
        capacity = map.capacity;
        items = (TagMapItem*)allocate_clear(capacity * sizeof(TagMapItem));
        for (TagMapItem* item = map.next(NULL); item; item = map.next(item)) {
            set(item->key, item->value);
        }
    }

    void resize(uint64_t new_capacity) {
        TagMap new_map;
        new_map.count = 0;
        new_map.capacity = new_capacity;
        new_map.items = (TagMapItem*)allocate_clear(new_capacity * sizeof(TagMapItem));
        const TagMapItem* limit = items + capacity;
        for (TagMapItem* it = items; it != limit; it++) {
            if (it->key != it->value) new_map.set(it->key, it->value);
        }
        clear();
        capacity = new_map.capacity;
        count = new_map.count;
        items = new_map.items;
    }

    // Function to iterate over all values in the map:
    // for (TagMapItem* item = map.next(NULL); item; item = map.next(item)) {…}
    TagMapItem* next(const TagMapItem* current) const {
        TagMapItem* next_ = current ? (TagMapItem*)(current + 1) : items;
        const TagMapItem* limit = items + capacity;
        while (next_ < limit) {
            if (next_->key != next_->value) return next_;
            next_++;
        }
        return NULL;
    }

    void clear() {
        if (items) {
            free_allocation(items);
            items = NULL;
        }
        capacity = 0;
        count = 0;
    }

    TagMapItem* get_slot(Tag key) const {
        assert(capacity > 0);
        assert(count < capacity);
        uint64_t h = hash(key) % capacity;
        TagMapItem* item = items + h;
        while (item->key != item->value && item->key != key) {
            item++;
            if (item == items + capacity) item = items;
        }
        // DEBUG_PRINT("get_slot %s [%" PRIu64 "] -> [%" PRIu64 "]\n", key, h, item - items);
        return item;
    }

    // Key is internally allocated and copied; value is simply assigned
    void set(Tag key, Tag value) {
        if (key == value) {
            del(key);
            return;
        }

        // Equality is important for capacity == 0
        if (count * 10 >= capacity * GDSTK_MAP_CAPACITY_THRESHOLD)
            resize(capacity >= GDSTK_INITIAL_MAP_CAPACITY ? capacity * GDSTK_MAP_GROWTH_FACTOR
                                                          : GDSTK_INITIAL_MAP_CAPACITY);
        TagMapItem* item = get_slot(key);
        if (item->key == item->value) {
            item->key = key;
            count++;
        }
        item->value = value;
    }

    bool has_key(Tag key) const {
        if (count == 0) return false;
        const TagMapItem* item = get_slot(key);
        return item->key != item->value;
    }

    // If the desired key is not found, returns the key itself
    Tag get(Tag key) const {
        if (count == 0) return key;
        const TagMapItem* item = get_slot(key);
        return item->key == item->value ? key : item->value;
    }

    // Return true if the key existed, false otherwise
    bool del(Tag key) {
        if (count == 0) return false;
        TagMapItem* item = get_slot(key);
        if (item->key == item->value) return false;

        // DEBUG_PRINT("DEL [%" PRIu64 "] %s\n", item - items, item->key);
        item->key = 0;
        item->value = 0;
        count--;

        // Re-insert this block to fill any undesired gaps
        while (true) {
            item++;
            if (item == items + capacity) item = items;
            if (item->key == item->value) return true;
            Tag temp_key = item->key;
            item->key = item->value;
            TagMapItem* new_item = get_slot(temp_key);
            new_item->key = temp_key;
            new_item->value = item->value;
            // if (new_item != item) {
            //     DEBUG_PRINT("MOVE %s [%" PRIu64 "] -> [%" PRIu64 "]\n", new_item->key,
            //                 item - items, new_item - items);
            // }
        }
        assert(false);
        return true;
    }
};

}  // namespace gdstk

#endif
