/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "allocator.h"

#include <sys/mman.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "utils.h"

#define PAGE_SIZE 0x1000  // This should be default for all targeted systems.

namespace gdstk {

static uint8_t* system_allocate(uint64_t size) {
    void* result = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (result == MAP_FAILED) {
        int err = errno;
        fprintf(stderr, "Error in mmap (errno %d).", err);
        return NULL;
    }
    // printf("Allocating %lu bytes at %p.\n", size, result); fflush(stdout);
    return (uint8_t*)result;
}

static void system_deallocate(void* ptr, uint64_t size) {
    // printf("Deallocating %lu bytes at %p.\n", size, ptr); fflush(stdout);
    munmap(ptr, size);
}

static uint64_t pow2_ceil(uint64_t n) {
    if (n == 0) return 1;
    if (n & (n - 1)) {
        n = n | (n >> 1);
        n = n | (n >> 2);
        n = n | (n >> 4);
        n = n | (n >> 8);
        n = n | (n >> 16);
        n = n | (n >> 32);
        n++;
    }
    return n;
}

// I'm not proud of this.
static uint8_t log2_lookup(uint64_t pow2) {
    switch (pow2) {
        case 0x0000000000000001:
            return 0;
        case 0x0000000000000002:
            return 1;
        case 0x0000000000000004:
            return 2;
        case 0x0000000000000008:
            return 3;
        case 0x0000000000000010:
            return 4;
        case 0x0000000000000020:
            return 5;
        case 0x0000000000000040:
            return 6;
        case 0x0000000000000080:
            return 7;
        case 0x0000000000000100:
            return 8;
        case 0x0000000000000200:
            return 9;
        case 0x0000000000000400:
            return 10;
        case 0x0000000000000800:
            return 11;
        case 0x0000000000001000:
            return 12;
        case 0x0000000000002000:
            return 13;
        case 0x0000000000004000:
            return 14;
        case 0x0000000000008000:
            return 15;
        case 0x0000000000010000:
            return 16;
        case 0x0000000000020000:
            return 17;
        case 0x0000000000040000:
            return 18;
        case 0x0000000000080000:
            return 19;
        case 0x0000000000100000:
            return 20;
        case 0x0000000000200000:
            return 21;
        case 0x0000000000400000:
            return 22;
        case 0x0000000000800000:
            return 23;
        case 0x0000000001000000:
            return 24;
        case 0x0000000002000000:
            return 25;
        case 0x0000000004000000:
            return 26;
        case 0x0000000008000000:
            return 27;
        case 0x0000000010000000:
            return 28;
        case 0x0000000020000000:
            return 29;
        case 0x0000000040000000:
            return 30;
        case 0x0000000080000000:
            return 31;
        case 0x0000000100000000:
            return 32;
        case 0x0000000200000000:
            return 33;
        case 0x0000000400000000:
            return 34;
        case 0x0000000800000000:
            return 35;
        case 0x0000001000000000:
            return 36;
        case 0x0000002000000000:
            return 37;
        case 0x0000004000000000:
            return 38;
        case 0x0000008000000000:
            return 39;
        case 0x0000010000000000:
            return 40;
        case 0x0000020000000000:
            return 41;
        case 0x0000040000000000:
            return 42;
        case 0x0000080000000000:
            return 43;
        case 0x0000100000000000:
            return 44;
        case 0x0000200000000000:
            return 45;
        case 0x0000400000000000:
            return 46;
        case 0x0000800000000000:
            return 47;
        case 0x0001000000000000:
            return 48;
        case 0x0002000000000000:
            return 49;
        case 0x0004000000000000:
            return 50;
        case 0x0008000000000000:
            return 51;
        case 0x0010000000000000:
            return 52;
        case 0x0020000000000000:
            return 53;
        case 0x0040000000000000:
            return 54;
        case 0x0080000000000000:
            return 55;
        case 0x0100000000000000:
            return 56;
        case 0x0200000000000000:
            return 57;
        case 0x0400000000000000:
            return 58;
        case 0x0800000000000000:
            return 59;
        case 0x1000000000000000:
            return 60;
        case 0x2000000000000000:
            return 61;
        case 0x4000000000000000:
            return 62;
        case 0x8000000000000000:
            return 63;
    }
    return 0;
}

// Arena allocator.  Small allocations are incresed to powers of 2 and have a SmallAllocationHeader
// with a pointer to a free list where they should be inserted when not in use.  This list is for a
// specific allocation size, so no size information is required in the header.  Large allocations
// include a LargeAllocationHeader with size information and a pointer, which is NULL while in use
// (thus differentiatiog them from small allocations) and used to keep a free list otherwise.
struct Arena {
    uint64_t available_size;
    uint8_t* cursor;
    Arena* next;
};

static Arena arena = {0, NULL, NULL};

static void* arena_allocate(uint64_t size) {
    static uint64_t next_arena_size = 1024 * 1024;
    Arena* a = &arena;
    while (a->available_size < size && a->next) {
        a = a->next;
    }
    if (a->available_size < size) {
        if (next_arena_size < 8 * size) {
            next_arena_size = 8 * size;
        }
        uint64_t full_size = sizeof(Arena) + next_arena_size;
        full_size = (full_size / PAGE_SIZE + 1) * PAGE_SIZE;
        next_arena_size = full_size - sizeof(Arena);
        a->next = (Arena*)system_allocate(full_size);
        a = a->next;
        a->available_size = next_arena_size;
        a->cursor = (uint8_t*)(a + 1);
        a->next = NULL;
        next_arena_size *= 2;
        // printf("New arena: %p with %lu bytes.\n", a, a->available_size); fflush(stdout);
    }
    void* result = (void*)a->cursor;
    a->cursor += size;
    a->available_size -= size;
    // printf("Arena allocation: %lu bytes at %p.\n", size, result); fflush(stdout);
    return result;
}

struct SmallAllocationHeader {
    SmallAllocationHeader* next;
};

struct LargeAllocationHeader {
    uint64_t size;
    LargeAllocationHeader* next;
};

// Sorted linked list
static LargeAllocationHeader free_large = {0, NULL};
static SmallAllocationHeader free_small[] = {{NULL},   // 8-byte allocations
                                             {NULL},   // 16
                                             {NULL},   // 32
                                             {NULL},   // 64
                                             {NULL},   // 128
                                             {NULL}};  // 256

void* allocate(uint64_t size) {
    const uint64_t largest_small_size = 1 << (2 + COUNT(free_small));
    if (size > largest_small_size) {
        LargeAllocationHeader* result = NULL;
        LargeAllocationHeader* free = &free_large;
        while (free->next && free->next->size < size) {
            free = free->next;
        }
        if (free->next && free->next->size <= 2 * size) {
            result = free->next;
            free->next = result->next;
        } else {
            result = (LargeAllocationHeader*)arena_allocate(size + sizeof(LargeAllocationHeader));
            result->size = size;
        }
        result->next = NULL;
        // printf("Large allocation: %lu bytes at %p (header: %p).\n", size, result + 1, result); fflush(stdout);
        return (void*)(result + 1);
    } else {
        SmallAllocationHeader* result = NULL;
        uint64_t pow2_size = pow2_ceil(size);
        if (pow2_size < 8) {
            pow2_size = 8;
        }
        uint8_t small_index = log2_lookup(pow2_size) - 3;
        SmallAllocationHeader* free = free_small + small_index;
        if (free->next) {
            result = free->next;
            free->next = result->next;
        } else {
            result =
                (SmallAllocationHeader*)arena_allocate(pow2_size + sizeof(SmallAllocationHeader));
        }
        result->next = free;
        // printf("Small allocation: %lu/%lu bytes at %p (header: %p).\n", size, pow2_size, result + 1, result); fflush(stdout);
        return (void*)(result + 1);
    }
    return NULL;
}

void* reallocate(void* ptr, uint64_t size) {
    if (ptr == NULL) return allocate(size);

    SmallAllocationHeader* sh =
        (SmallAllocationHeader*)((uint8_t*)ptr - sizeof(SmallAllocationHeader));
    if (sh->next) {
        // Small allocation
        uint8_t small_index = sh->next - free_small;
        uint64_t ptr_size = 1 << (3 + small_index);
        if (size <= ptr_size) return ptr;

        void* new_ptr = allocate(size);
        memcpy(new_ptr, ptr, ptr_size);
        free_mem(ptr);
        // printf("Small reallocation: %p (%lu bytes) -> %p (%lu bytes).\n", ptr, ptr_size, new_ptr, size); fflush(stdout);
        return new_ptr;
    } else {
        // Large allocation
        LargeAllocationHeader* lh =
            (LargeAllocationHeader*)((uint8_t*)ptr - sizeof(LargeAllocationHeader));
        if (size <= lh->size) return ptr;

        Arena* a = arena.next;
        // Find the arena to which this ptr belongs.
        while (a->cursor < (uint8_t*)lh || (uint8_t*)a > (uint8_t*)lh) a = a->next;

        // Check to see if this is the last allocation and there is enough space afterwards.
        uint64_t size_increase = size - lh->size;
        if (a->cursor == (uint8_t*)ptr + lh->size && a->available_size >= size_increase) {
            // printf("Reallocation of %p: %lu -> %lu\n", ptr, lh->size, size); fflush(stdout);
            a->available_size -= size_increase;
            a->cursor += size_increase;
            lh->size += size_increase;
            return ptr;
        }

        void* new_ptr = allocate(size);
        memcpy(new_ptr, ptr, lh->size);
        free_mem(ptr);
        // printf("Large reallocation: %p (%lu bytes) -> %p (%lu bytes).\n", ptr, lh->size, new_ptr, size); fflush(stdout);
        return new_ptr;
    }
    return NULL;
}

void* allocate_clear(uint64_t size) {
    void* result = allocate(size);
    memset(result, 0, size);
    return result;
}

void free_mem(void* ptr) {
    if (ptr == NULL) return;
    SmallAllocationHeader* sh =
        (SmallAllocationHeader*)((uint8_t*)ptr - sizeof(SmallAllocationHeader));
    if (sh->next) {
        // Small allocation
        SmallAllocationHeader* free = sh->next;
        sh->next = free->next;
        free->next = sh;
        // printf("Free small allocation %p (header: %p).\n", ptr, sh); fflush(stdout);
    } else {
        LargeAllocationHeader* lh =
            (LargeAllocationHeader*)((uint8_t*)ptr - sizeof(LargeAllocationHeader));
        LargeAllocationHeader* free = &free_large;
        while (free->next && free->next->size < lh->size) {
            free = free->next;
        }
        lh->next = free->next;
        free->next = lh;
        // printf("Free large allocation %p (header: %p).\n", ptr, lh); fflush(stdout);
    }
}

void gdstk_finalize() {
    Arena* a = arena.next;
    while (a) {
        Arena* b = a->next;
        system_deallocate(a, a->cursor + a->available_size - (uint8_t*)a);
        a = b;
    }
}

}  // namespace gdstk
