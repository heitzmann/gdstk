/*
Copyright 2021 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_SET
#define GDSTK_HEADER_SET

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <assert.h>

#include "allocator.hpp"
#include "utils.hpp"

namespace gdstk {

template <class T>
struct SetItem {
    T value;
    bool valid;
};

template <class T>
struct Set {
    uint64_t capacity;  // allocated capacity
    uint64_t count;     // number of items in the set
    SetItem<T>* items;  // array with length capacity

    void print(bool all, void (*value_print)(const T&)) const {
        printf("Set <%p>, count %" PRIu64 "/%" PRIu64 ", items <%p>\n", this, count, capacity,
               items);
        if (all) {
            SetItem<T>* item = items;
            if (value_print) {
                for (uint64_t i = 0; i < capacity; i++, item++) {
                    if (item->valid) {
                        printf("Item[%" PRIu64 "] ", i);
                        value_print(item->value);
                    } else {
                        printf("Item[%" PRIu64 "]\n", i);
                    }
                }
            } else {
                for (uint64_t i = 0; i < capacity; i++, item++) {
                    printf("Item[%" PRIu64 "] %s\n", i, item->valid ? "valid" : "invalid");
                }
            }
        }
    }

    void clear() {
        if (items) {
            free_allocation(items);
            items = NULL;
        }
        capacity = 0;
        count = 0;
    }

    // The instance should be zeroed before using copy_from
    void copy_from(const Set<T>& set) {
        count = 0;
        capacity = set.capacity;
        items = (SetItem<T>*)allocate_clear(capacity * sizeof(SetItem<T>));
        for (SetItem<T>* item = set.next(NULL); item; item = set.next(item)) add(item->value);
    }

    void resize(uint64_t new_capacity) {
        Set<T> new_set;
        new_set.count = 0;
        new_set.capacity = new_capacity;
        new_set.items = (SetItem<T>*)allocate_clear(new_capacity * sizeof(SetItem<T>));
        const SetItem<T>* limit = items + capacity;
        for (SetItem<T>* it = items; it != limit; it++) {
            if (it->valid) new_set.add(it->value);
        }
        clear();
        capacity = new_set.capacity;
        count = new_set.count;
        items = new_set.items;
    }

    // Function to iterate over all values in the set:
    // for (SetItem<T>* item = set.next(NULL); item; item = set.next(item)) {…}
    SetItem<T>* next(const SetItem<T>* current) const {
        SetItem<T>* next_ = current ? (SetItem<T>*)(current + 1) : items;
        const SetItem<T>* limit = items + capacity;
        while (next_ < limit) {
            if (next_->valid) return next_;
            next_++;
        }
        return NULL;
    }

    void to_array(Array<T>& result) const {
        result.ensure_slots(count);
        const SetItem<T>* limit = items + capacity;
        for (SetItem<T>* it = items; it != limit; it++) {
            if (it->valid) result.append_unsafe(it->value);
        }
    }

    SetItem<T>* get_slot(T value) const {
        assert(capacity > 0);
        uint64_t h = hash(value) % capacity;
        SetItem<T>* item = items + h;
        while (item->valid && item->value != value) {
            item++;
            if (item == items + capacity) item = items;
        }
        return item;
    }

    void add(T value) {
        // Equality is important for capacity == 0
        if (count * 10 >= capacity * GDSTK_MAP_CAPACITY_THRESHOLD)
            resize(capacity >= GDSTK_INITIAL_MAP_CAPACITY ? capacity * GDSTK_MAP_GROWTH_FACTOR
                                                          : GDSTK_INITIAL_MAP_CAPACITY);
        SetItem<T>* item = get_slot(value);
        if (!item->valid) {
            count++;
            item->value = value;
            item->valid = true;
        }
    }

    bool has_value(T value) const {
        if (count == 0) return false;
        const SetItem<T>* item = get_slot(value);
        return item->valid;
    }

    // Return true if the value was in the set, false otherwise
    bool del(T value) {
        if (count == 0) return false;
        SetItem<T>* item = get_slot(value);
        if (!item->valid) return false;

        item->valid = false;
        count--;

        // Re-insert this block to fill any undesired gaps
        while (true) {
            item++;
            if (item == items + capacity) item = items;
            if (!item->valid) return true;
            item->valid = false;
            SetItem<T>* new_item = get_slot(item->value);
            new_item->valid = true;
            new_item->value = item->value;
        }
        assert(false);
        return true;
    }
};

}  // namespace gdstk

#endif
