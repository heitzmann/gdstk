/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "allocator.h"

#ifdef GDSTK_CUSTOM_ALLOCATOR

#ifdef _WIN32
// clang-format off
#include <Windows.h>
#include <Memoryapi.h>
// clang-format on
#else
#include <sys/mman.h>
#endif

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

// TODO: Thread safety

#define PAGE_SIZE 0x1000  // This should be default for all targeted systems.

namespace gdstk {

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

struct SmallAllocationHeader {
    SmallAllocationHeader* next;
};

struct LargeAllocationHeader {
    uint64_t size;
    LargeAllocationHeader* next;
};

static Arena arena = {0, NULL, NULL};

// Sorted linked list
static LargeAllocationHeader free_large = {0, NULL};
static SmallAllocationHeader free_small[] = {{NULL},   // 8-byte allocations
                                             {NULL},   // 16
                                             {NULL},   // 32
                                             {NULL},   // 64
                                             {NULL},   // 128
                                             {NULL},   // 256
                                             {NULL},   // 512
                                             {NULL},   // 1024
                                             {NULL},   // 2048
                                             {NULL}};  // 4096

#ifdef GDSTK_ALLOCATOR_INFO
static uint64_t dbg_sys_alloc = 0;
static uint64_t dbg_sys_free = 0;
static uint64_t dbg_arena_alloc = 0;
static uint64_t dbg_alloc[COUNT(free_small) + 1] = {0};
static uint64_t dbg_realloc[COUNT(free_small) + 1] = {0};
static uint64_t dbg_reuse[COUNT(free_small) + 1] = {0};
static uint64_t dbg_free[COUNT(free_small) + 1] = {0};

static void print_status() {
    uint64_t t = 0;
    printf("System allocations/free: %lu/%lu\nArena allocations: %lu\nAllocations:", dbg_sys_alloc,
           dbg_sys_free, dbg_arena_alloc);
    for (uint64_t i = 0; i <= COUNT(free_small); i++) {
        t += dbg_alloc[i];
        printf(" %lu", dbg_alloc[i]);
    }
    printf("  (%lu)\nReallocations:", t);
    t = 0;
    for (uint64_t i = 0; i <= COUNT(free_small); i++) {
        t += dbg_realloc[i];
        printf(" %lu", dbg_realloc[i]);
    }
    printf("  (%lu)\nReuses:", t);
    t = 0;
    for (uint64_t i = 0; i <= COUNT(free_small); i++) {
        t += dbg_reuse[i];
        printf(" %lu", dbg_reuse[i]);
    }
    printf("  (%lu)\nFrees:", t);
    t = 0;
    for (uint64_t i = 0; i <= COUNT(free_small); i++) {
        t += dbg_free[i];
        printf(" %lu", dbg_free[i]);
    }
    printf("  (%lu)\n", t);

    printf("Waste:");
    uint64_t all_waste = 0;
    uint64_t waste;
    for (uint64_t i = 0; i < COUNT(free_small); i++) {
        waste = 0;
        for (SmallAllocationHeader* h = free_small + i; h; h = h->next) waste++;
        waste *= 1 << (i + 3);
        all_waste += waste;
        printf(" %lu", waste);
    }

    waste = 0;
    for (LargeAllocationHeader* h = &free_large; h; h = h->next) waste += h->size;
    all_waste += waste;
    printf(" %lu  (%lu)\n", waste, all_waste);

    waste = 0;
    for (Arena* a = &arena; a; a = a->next) waste += a->available_size;
    printf("Arena waste: %lu\n", waste);
}
#endif

static uint8_t* system_allocate(uint64_t size) {
#ifdef _WIN32
    void* result = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (result == NULL) {
        fprintf(stderr, "Error allocating memory.");
        return NULL;
    }
#else
    void* result = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (result == MAP_FAILED) {
        int err = errno;
        fprintf(stderr, "Error in mmap (errno %d).", err);
        return NULL;
    }
#endif
#ifdef GDSTK_ALLOCATOR_INFO
    dbg_sys_alloc++;
#endif
    return (uint8_t*)result;
}

static void system_deallocate(void* ptr, uint64_t size) {
#ifdef GDSTK_ALLOCATOR_INFO
    dbg_sys_free++;
#endif
#ifdef _WIN32
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    munmap(ptr, size);
#endif
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
    fprintf(stderr, "%s:%d: Unreachable\n", __FILE__, __LINE__);
    return 0;
}

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
    }
    void* result = (void*)a->cursor;
    a->cursor += size;
    a->available_size -= size;
#ifdef GDSTK_ALLOCATOR_INFO
    dbg_arena_alloc++;
#endif
    return result;
}

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
#ifdef GDSTK_ALLOCATOR_INFO
            dbg_reuse[COUNT(free_small)]++;
#endif
        } else {
            result = (LargeAllocationHeader*)arena_allocate(size + sizeof(LargeAllocationHeader));
            result->size = size;
#ifdef GDSTK_ALLOCATOR_INFO
            dbg_alloc[COUNT(free_small)]++;
#endif
        }
        result->next = NULL;
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
#ifdef GDSTK_ALLOCATOR_INFO
            dbg_reuse[small_index]++;
#endif
        } else {
            result =
                (SmallAllocationHeader*)arena_allocate(pow2_size + sizeof(SmallAllocationHeader));
#ifdef GDSTK_ALLOCATOR_INFO
            dbg_alloc[small_index]++;
#endif
        }
        result->next = free;
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

#ifdef GDSTK_ALLOCATOR_INFO
        dbg_realloc[small_index]++;
#endif
        void* new_ptr = allocate(size);
        memcpy(new_ptr, ptr, ptr_size);
        free_allocation(ptr);
        return new_ptr;
    } else {
        // Large allocation
        LargeAllocationHeader* lh =
            (LargeAllocationHeader*)((uint8_t*)ptr - sizeof(LargeAllocationHeader));
        if (size <= lh->size) return ptr;

#ifdef GDSTK_ALLOCATOR_INFO
        dbg_realloc[COUNT(free_small)]++;
#endif
        Arena* a = arena.next;
        // Find the arena to which this ptr belongs.
        while (a->cursor < (uint8_t*)lh || (uint8_t*)a > (uint8_t*)lh) a = a->next;

        // Check to see if this is the last allocation and there is enough space afterwards.
        uint64_t size_increase = size - lh->size;
        if (a->cursor == (uint8_t*)ptr + lh->size && a->available_size >= size_increase) {
            a->available_size -= size_increase;
            a->cursor += size_increase;
            lh->size += size_increase;
            return ptr;
        }

        void* new_ptr = allocate(size);
        memcpy(new_ptr, ptr, lh->size);
        free_allocation(ptr);
        return new_ptr;
    }
    return NULL;
}

void* allocate_clear(uint64_t size) {
    void* result = allocate(size);
    memset(result, 0, size);
    return result;
}

void free_allocation(void* ptr) {
    if (ptr == NULL) return;
    SmallAllocationHeader* sh =
        (SmallAllocationHeader*)((uint8_t*)ptr - sizeof(SmallAllocationHeader));
    if (sh->next) {
        // Small allocation
#ifdef GDSTK_ALLOCATOR_INFO
        uint8_t small_index = sh->next - free_small;
        dbg_free[small_index]++;
#endif
        SmallAllocationHeader* free = sh->next;
        sh->next = free->next;
        free->next = sh;
    } else {
#ifdef GDSTK_ALLOCATOR_INFO
        dbg_free[COUNT(free_small)]++;
#endif
        LargeAllocationHeader* lh =
            (LargeAllocationHeader*)((uint8_t*)ptr - sizeof(LargeAllocationHeader));
        LargeAllocationHeader* free = &free_large;
        while (free->next && free->next->size < lh->size) {
            free = free->next;
        }
        lh->next = free->next;
        free->next = lh;
    }
}

void gdstk_finalize() {
#ifdef GDSTK_ALLOCATOR_INFO
    print_status();
#endif
    Arena* a = arena.next;
    while (a) {
        Arena* b = a->next;
        system_deallocate(a, a->cursor + a->available_size - (uint8_t*)a);
        a = b;
    }
}

}  // namespace gdstk

#endif  // GDSTK_CUSTOM_ALLOCATOR
