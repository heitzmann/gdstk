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
#define INITIAL_MAP_CAPACITY 4
#define MAP_CAPACITY_THRESHOLD 7  // in tenths

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
    MapItem<T>* next;
};

template <class T>
struct Map {
    uint64_t capacity;  // allocated capacity
    uint64_t count;     // number of slots used
    MapItem<T>* items;  // slots

    void print(bool all) const {
        printf("Map <%p>, count %" PRIu64 "/%" PRIu64 ", items <%p>\n", this, count, capacity,
               items);
        if (all) {
            MapItem<T>* item = items;
            for (uint64_t i = 0; i < capacity; i++, item++) {
                printf("(%" PRIu64 ") Item <%p>, key <%p> %s, value <%p>, next <%p>\n", i, item,
                       item->key, item->key ? item->key : "", item->value, item->next);
                for (MapItem<T>* it = item->next; it; it = it->next) {
                    printf("(%" PRIu64 ") Item <%p>, key <%p> %s, value <%p>, next <%p>\n", i, it,
                           it->key, it->key ? it->key : "", it->value, it->next);
                }
            }
        }
    }

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
        for (MapItem<T>* it = next(NULL); it; it = next(it)) new_map.set(it->key, it->value);
        clear();
        capacity = new_map.capacity;
        count = new_map.count;
        items = new_map.items;
    }

    MapItem<T>* next(const MapItem<T>* current) const {
        if (!current) {
            for (uint64_t i = 0; i < capacity; i++)
                if (items[i].key) return items + i;
            return NULL;
        }
        if (current->next) return current->next;
        for (uint64_t i = hash(current->key) % capacity + 1; i < capacity; i++)
            if (items[i].key) return items + i;
        return NULL;
    }

    void to_array(Array<T>& result) const {
        result.ensure_slots(count);
        for (MapItem<T>* it = next(NULL); it; it = next(it)) result.append_unsafe(it->value);
    }

    void clear() {
        if (items) {
            MapItem<T>* item = items;
            for (uint64_t i = 0; i < capacity; i++, item++) {
                if (item->key) {
                    free_allocation(item->key);
                    item->key = NULL;
                }
                MapItem<T>* it = item->next;
                while (it) {
                    MapItem<T>* tmp = it->next;
                    free_allocation(it->key);
                    free_allocation(it);
                    it = tmp;
                }
            }
            free_allocation(items);
            items = NULL;
        }
        capacity = 0;
        count = 0;
    }

    void set(const char* key, T value) {
        // Equallity is important for capacity == 0
        if (count * 10 >= capacity * MAP_CAPACITY_THRESHOLD)
            resize(capacity >= INITIAL_MAP_CAPACITY ? capacity * MAP_GROWTH_FACTOR
                                                    : INITIAL_MAP_CAPACITY);

        uint64_t h = hash(key) % capacity;
        MapItem<T>* item = items + h;
        if (item->key == NULL) {
            uint64_t len;
            item->key = copy_string(key, len);
            item->value = value;
            count++;
        } else if (strcmp(item->key, key) == 0) {
            item->value = value;
        } else {
            while (item->next && strcmp(item->next->key, key) != 0) item = item->next;
            if (item->next) {
                item->next->value = value;
            } else {
                item->next = (MapItem<T>*)allocate(sizeof(MapItem<T>));
                item = item->next;
                uint64_t len;
                item->key = copy_string(key, len);
                item->value = value;
                item->next = NULL;
                count++;
            }
        }
    }

    bool has_key(const char* key) const {
        if (count == 0) return false;
        uint64_t h = hash(key) % capacity;
        MapItem<T>* item = items + h;
        if (item->key == NULL) return false;
        for (; item != NULL; item = item->next) {
            if (strcmp(item->key, key) == 0) return true;
        }
        return false;
    }

    T get(const char* key) const {
        if (count == 0) return T{0};
        uint64_t h = hash(key) % capacity;
        MapItem<T>* item = items + h;
        if (item->key == NULL) return T{0};
        if (strcmp(item->key, key) == 0) return item->value;
        while (item->next && strcmp(item->next->key, key) != 0) item = item->next;
        if (item->next) return item->next->value;
        return T{0};
    }

    void del(const char* key) {
        if (count == 0) return;
        uint64_t h = hash(key) % capacity;
        MapItem<T>* item = items + h;
        if (item->key == NULL) return;
        if (strcmp(item->key, key) == 0) {
            free_allocation(item->key);
            if (item->next) {
                MapItem<T>* it = item;
                while (it->next->next) it = it->next;
                item->key = it->next->key;
                item->value = it->next->value;
                it->next = NULL;
            } else {
                item->key = 0;
            }
            return;
        }
        while (item->next && strcmp(item->next->key, key) != 0) item = item->next;
        if (item->next) {
            MapItem<T>* it = item->next->next;
            free_allocation(item->next->key);
            free_allocation(item->next);
            item->next = it;
        }
    }
};

}  // namespace gdstk

#endif
