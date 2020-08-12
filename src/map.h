/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef __MAP_H__
#define __MAP_H__

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "utils.h"

namespace gdstk {

// FNV-1a hash function (32 bits)
#define MAP_FNV_PRIME 0x01000193
#define MAP_FNV_OFFSET 0x811c9dc5
static int64_t hash(const char* key) {
    uint32_t h = MAP_FNV_OFFSET;
    for (const char* c = key; *c; c++) {
        h ^= (uint32_t)(*c);
        h *= MAP_FNV_PRIME;
    }
    return (int64_t)h;
}

template <class T>
struct MapItem {
    char* key;
    T value;
    MapItem<T>* next;
};

template <class T>
struct Map {
    int64_t capacity;   // allocated capacity
    int64_t size;       // number of slots used
    MapItem<T>* items;  // slots

    Map<T>(int64_t initial_capacity) : capacity{initial_capacity}, size{0} {
        items = (MapItem<T>*)calloc(initial_capacity, sizeof(MapItem<T>));
    }

    Map<T>(const Map<T>& map) : size{0} {
        capacity = map.capacity;
        items = (MapItem<T>*)calloc(capacity, sizeof(MapItem<T>));
        for (MapItem<T>* item = next_item(NULL); item; item = next_item(item))
            set(item->key, item->value);
    }

    void print(bool all) const {
        printf("Map <%p>, size %ld/%ld, items <%p>\n", this, size, capacity, items);
        if (all) {
            MapItem<T>* item = items;
            for (int64_t i = 0; i < capacity; i++, item++) {
                printf("(%ld) Item <%p>, key <%p:%s>, value <%p>, next <%p>\n", i, item, item->key,
                       item->key ? item->key : "", item->value, item->next);
                for (MapItem<T>* it = item->next; it; it = it->next) {
                    printf("(%ld) Item <%p>, key <%p:%s>, value <%p>, next <%p>\n", i, it, it->key,
                           it->key ? it->key : "", it->value, it->next);
                }
            }
        }
    }

    MapItem<T>* next_item(const MapItem<T>* current) const {
        if (!current) {
            for (int64_t i = 0; i < capacity; i++)
                if (items[i].key) return items + i;
            return NULL;
        }
        if (current->next) return current->next;
        for (int64_t i = hash(current->key) % capacity + 1; i < capacity; i++)
            if (items[i].key) return items + i;
        return NULL;
    }

    void clear() {
        if (items) {
            MapItem<T>* item = items;
            for (int64_t i = 0; i < capacity; i++, item++) {
                if (item->key) {
                    free(item->key);
                    item->key = NULL;
                }
                MapItem<T>* it = item->next;
                while (it) {
                    MapItem<T>* tmp = it->next;
                    free(it->key);
                    free(it);
                    it = tmp;
                }
            }
            free(items);
            items = NULL;
        }
        capacity = 0;
        size = 0;
    }

    void set(const char* key, T value) {
        int64_t h = hash(key) % capacity;
        MapItem<T>* item = items + h;
        if (item->key == NULL) {
            int64_t len = 1 + strlen(key);
            item->key = (char*)malloc(sizeof(char) * len);
            memcpy(item->key, key, len);
            item->value = value;
            size++;
        } else if (strcmp(item->key, key) == 0) {
            item->value = value;
        } else {
            while (item->next && strcmp(item->next->key, key) != 0) item = item->next;
            if (item->next) {
                item->next->value = value;
            } else {
                item->next = (MapItem<T>*)malloc(sizeof(MapItem<T>));
                item = item->next;
                int64_t len = 1 + strlen(key);
                item->key = (char*)malloc(sizeof(char) * len);
                memcpy(item->key, key, len);
                item->value = value;
                item->next = NULL;
                size++;
            }
        }
        if (size * 10 > capacity * MAP_CAP) {
            Map<T> new_map(2 * capacity);
            for (MapItem<T>* it = next_item(NULL); it; it = next_item(it))
                new_map.set(it->key, it->value);
            clear();
            capacity = new_map.capacity;
            size = new_map.size;
            items = new_map.items;
        }
    }

    T get(const char* key) const {
        int64_t h = hash(key) % capacity;
        MapItem<T>* item = items + h;
        if (item->key == NULL) return T{0};
        if (strcmp(item->key, key) == 0) return item->value;
        while (item->next && strcmp(item->next->key, key) != 0) item = item->next;
        if (item->next) return item->next->value;
        return T{0};
    }

    void del(const char* key) {
        int64_t h = hash(key) % capacity;
        MapItem<T>* item = items + h;
        if (item->key == NULL) return;
        if (strcmp(item->key, key) == 0) {
            free(item->key);
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
            free(item->next->key);
            free(item->next);
            item->next = it;
        }
    }
};

}  // namespace gdstk

#endif
