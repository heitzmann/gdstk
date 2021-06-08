/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_MAP
#define GDSTK_HEADER_MAP

#define __STDC_FORMAT_MACROS
#define _USE_MATH_DEFINES

#define MAP_GROWTH_FACTOR 2
#define INITIAL_MAP_CAPACITY 8
#define MAP_CAPACITY_THRESHOLD 5  // in tenths

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allocator.h"
#include "utils.h"

namespace gdstk {

// FNV-1a hash function (64 bits)
#define MAP_FNV_PRIME 0x00000100000001b3
#define MAP_FNV_OFFSET 0xcbf29ce484222325
static inline uint64_t hash(const char* key) {
    uint64_t result = MAP_FNV_OFFSET;
    for (const char* c = key; *c; c++) {
        result ^= (uint64_t)(*c);
        result *= MAP_FNV_PRIME;
    }
    return result;
}

template <class T>
struct MapItem {
    char* key;
    T value;
};

// Hash map indexed by NULL-terminated strings
template <class T>
struct Map {
    uint64_t capacity;  // allocated capacity
    uint64_t count;     // number of items in the map
    MapItem<T>* items;  // array with length capacity

    void print(bool all) const {
        printf("Map <%p>, count %" PRIu64 "/%" PRIu64 ", items <%p>\n", this, count, capacity,
               items);
        if (all) {
            MapItem<T>* item = items;
            for (uint64_t i = 0; i < capacity; i++, item++) {
                printf("(%" PRIu64 ") Item <%p>, key %p (%s), value <%p>\n", i, item, item->key,
                       item->key ? item->key : "", item->value);
            }
        }
    }

    // The instance should be zeroed before using copy_from
    void copy_from(const Map<T>& map) {
        count = 0;
        capacity = map.capacity;
        items = (MapItem<T>*)allocate_clear(capacity * sizeof(MapItem<T>));
        for (MapItem<T>* item = map.next(NULL); item; item = map.next(item))
            set(item->key, item->value);
    }

    void resize(uint64_t new_capacity) {
        Map<T> new_map;
        new_map.count = 0;
        new_map.capacity = new_capacity;
        new_map.items = (MapItem<T>*)allocate_clear(new_capacity * sizeof(MapItem<T>));
        const MapItem<T>* limit = items + capacity;
        for (MapItem<T>* it = items; it != limit; it++) {
            if (it->key) new_map.set(it->key, it->value);
        }
        clear();
        capacity = new_map.capacity;
        count = new_map.count;
        items = new_map.items;
    }

    // Function to iterate over all values in the map:
    // for (MapItem<T>* item = map.next(NULL); item; item = map.next(item)) {…}
    MapItem<T>* next(const MapItem<T>* current) const {
        MapItem<T>* next = current ? (MapItem<T>*)(current + 1) : items;
        const MapItem<T>* limit = items + capacity;
        while (next < limit) {
            if (next->key) return next;
            next++;
        }
        return NULL;
    }

    void to_array(Array<T>& result) const {
        result.ensure_slots(count);
        const MapItem<T>* limit = items + capacity;
        for (MapItem<T>* it = items; it != limit; it++) {
            if (it->key) result.append_unsafe(it->value);
        }
    }

    void clear() {
        if (items) {
            MapItem<T>* item = items;
            for (uint64_t i = 0; i < capacity; i++, item++) {
                if (item->key) {
                    free_allocation(item->key);
                    item->key = NULL;
                }
            }
            free_allocation(items);
            items = NULL;
        }
        capacity = 0;
        count = 0;
    }

    MapItem<T>* get_slot(const char* key) const {
        assert(capacity > 0);
        uint64_t h = hash(key) % capacity;
        MapItem<T>* item = items + h;
        while (item->key != NULL && strcmp(item->key, key) != 0) {
            item++;
            if (item == items + capacity) item = items;
        }
        // DEBUG_PRINT("get_slot %s [%" PRIu64 "] -> [%" PRIu64 "]\n", key, h, item - items);
        return item;
    }

    // Key is internally allocated and copied; value is simply assigned
    void set(const char* key, T value) {
        // Equallity is important for capacity == 0
        if (count * 10 >= capacity * MAP_CAPACITY_THRESHOLD)
            resize(capacity >= INITIAL_MAP_CAPACITY ? capacity * MAP_GROWTH_FACTOR
                                                    : INITIAL_MAP_CAPACITY);
        MapItem<T>* item = get_slot(key);
        if (item->key == NULL) {
            uint64_t len;
            item->key = copy_string(key, len);
            count++;
        }
        item->value = value;
    }

    bool has_key(const char* key) const {
        if (count == 0) return false;
        const MapItem<T>* item = get_slot(key);
        return item->key != NULL;
    }

    // If the desired key is not found, returns T{0}
    T get(const char* key) const {
        if (count == 0) return T{0};
        const MapItem<T>* item = get_slot(key);
        return item->key == NULL ? T{0} : item->value;
    }

    // Return true if the key existed, false otherwise
    bool del(const char* key) {
        if (count == 0) return false;
        MapItem<T>* item = get_slot(key);
        if (item->key == NULL) return false;

        // DEBUG_PRINT("DEL [%" PRIu64 "] %s\n", item - items, item->key);
        free_allocation(item->key);
        item->key = NULL;
        count--;

        // Re-insert this block to fill any undesired gaps
        while (true) {
            item++;
            if (item == items + capacity) item = items;
            if (item->key == NULL) return true;
            char* temp_key = item->key;
            item->key = NULL;
            MapItem<T>* new_item = get_slot(temp_key);
            new_item->key = temp_key;
            if (new_item != item) {
                // DEBUG_PRINT("MOVE %s [%" PRIu64 "] -> [%" PRIu64 "]\n", new_item->key,
                //             item - items, new_item - items);
                new_item->value == item->value;
            }
        }
        assert(false);
        return true;
    }
};

}  // namespace gdstk

#endif
