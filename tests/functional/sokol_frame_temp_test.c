//------------------------------------------------------------------------------
//  sokol_frame_temp_test.c
//
//  LLM maintained.
//
//  Reference per-frame temporary allocator and its functional tests.
//  This file is test-only; the production API belongs in a human-written
//  sokol header.
//------------------------------------------------------------------------------
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "utest.h"

#define T(b) EXPECT_TRUE(b)

#define SFT_DEFAULT_CAPACITY (1024u * 1024u)
#define SFT_DEFAULT_FRAME_SLOTS 2u
#define SFT_MAX_FRAME_SLOTS 8u
#define SFT_MAX_ALIGNMENT 256u
#define SFT_NO_CLEAR (1u << 0)
#define SFT_FRAME_TEMP_INIT {0}

typedef enum sft_result {
    SFT_RESULT_OK = 0,
    SFT_RESULT_INVALID_ARGUMENT,
    SFT_RESULT_NOT_INITIALIZED,
    SFT_RESULT_ALREADY_INITIALIZED,
    SFT_RESULT_FRAME_ALREADY_OPEN,
    SFT_RESULT_FRAME_NOT_OPEN,
    SFT_RESULT_OVERFLOW,
    SFT_RESULT_GENERATION_EXHAUSTED,
    SFT_RESULT_OUT_OF_MEMORY,
} sft_result;

typedef struct sft_allocator {
    void* (*alloc_fn)(size_t size, void* user_data);
    void (*free_fn)(void* ptr, void* user_data);
    void* user_data;
} sft_allocator;

typedef struct sft_desc {
    size_t capacity;
    uint32_t frame_slots;
    uint32_t flags;
    sft_allocator allocator;
} sft_desc;

typedef struct sft_block {
    void* ptr;
    size_t size;
    uint32_t slot;
    uint32_t _reserved;
    uint64_t generation;
} sft_block;

typedef struct sft_stats {
    size_t capacity;
    size_t slot_capacity;
    size_t backing_bytes;
    size_t used;
    size_t peak_used;
    size_t payload;
    size_t padding;
    uint64_t frames_started;
    uint64_t frames_ended;
    uint64_t allocations;
    uint64_t overflows;
    uint64_t bytes_allocated;
    uint64_t bytes_cleared;
} sft_stats;

typedef struct sft_frame_temp {
    uint8_t* base;
    void* alloc_base;
    size_t capacity;
    size_t slot_capacity;
    size_t offset;
    size_t peak_used;
    size_t payload;
    size_t padding;
    uint32_t num_slots;
    uint32_t cur_slot;
    uint32_t flags;
    bool valid;
    bool frame_open;
    uint64_t generation;
    uint64_t frames_started;
    uint64_t frames_ended;
    uint64_t allocations;
    uint64_t overflows;
    uint64_t bytes_allocated;
    uint64_t bytes_cleared;
    sft_allocator allocator;
} sft_frame_temp;

static size_t sft_align_up(size_t value, size_t alignment) {
    return (value + (alignment - 1u)) & ~(alignment - 1u);
}

static bool sft_is_pow2(size_t value) {
    return (value != 0u) && ((value & (value - 1u)) == 0u);
}

static sft_result sft_init(sft_frame_temp* ft, const sft_desc* desc) {
    if (!ft || !desc) {
        return SFT_RESULT_INVALID_ARGUMENT;
    }
    if (ft->valid) {
        return SFT_RESULT_ALREADY_INITIALIZED;
    }
    memset(ft, 0, sizeof(*ft));

    const size_t capacity = (desc->capacity == 0u) ? SFT_DEFAULT_CAPACITY : desc->capacity;
    const uint32_t num_slots = (desc->frame_slots == 0u) ? SFT_DEFAULT_FRAME_SLOTS : desc->frame_slots;
    if ((num_slots > SFT_MAX_FRAME_SLOTS) || ((desc->flags & ~SFT_NO_CLEAR) != 0u)) {
        return SFT_RESULT_INVALID_ARGUMENT;
    }
    if (capacity > (SIZE_MAX - ((size_t)SFT_MAX_ALIGNMENT - 1u))) {
        return SFT_RESULT_INVALID_ARGUMENT;
    }
    const bool has_alloc = desc->allocator.alloc_fn != NULL;
    const bool has_free = desc->allocator.free_fn != NULL;
    if (has_alloc != has_free) {
        return SFT_RESULT_INVALID_ARGUMENT;
    }

    const size_t slot_capacity = (capacity / (size_t)num_slots) & ~((size_t)SFT_MAX_ALIGNMENT - 1u);
    if (slot_capacity == 0u) {
        return SFT_RESULT_INVALID_ARGUMENT;
    }
    const size_t usable = slot_capacity * (size_t)num_slots;
    const size_t backing = usable + (size_t)SFT_MAX_ALIGNMENT - 1u;
    void* alloc_base;
    if (has_alloc) {
        alloc_base = desc->allocator.alloc_fn(backing, desc->allocator.user_data);
    } else {
        alloc_base = malloc(backing);
    }
    if (!alloc_base) {
        return SFT_RESULT_OUT_OF_MEMORY;
    }

    ft->alloc_base = alloc_base;
    ft->base = (uint8_t*)(((uintptr_t)alloc_base + ((uintptr_t)SFT_MAX_ALIGNMENT - 1u))
        & ~((uintptr_t)SFT_MAX_ALIGNMENT - 1u));
    ft->capacity = usable;
    ft->slot_capacity = slot_capacity;
    ft->num_slots = num_slots;
    ft->flags = desc->flags;
    ft->allocator = desc->allocator;
    ft->valid = true;
    return SFT_RESULT_OK;
}

static void sft_shutdown(sft_frame_temp* ft) {
    if (!ft || !ft->valid) {
        return;
    }
    if (ft->allocator.free_fn) {
        ft->allocator.free_fn(ft->alloc_base, ft->allocator.user_data);
    } else {
        free(ft->alloc_base);
    }
    memset(ft, 0, sizeof(*ft));
}

static sft_result sft_begin_frame(sft_frame_temp* ft) {
    if (!ft || !ft->valid) {
        return SFT_RESULT_NOT_INITIALIZED;
    }
    if (ft->frame_open) {
        return SFT_RESULT_FRAME_ALREADY_OPEN;
    }
    if (ft->generation == UINT64_MAX) {
        return SFT_RESULT_GENERATION_EXHAUSTED;
    }
    ft->generation++;
    ft->cur_slot = (uint32_t)((ft->generation - 1u) % (uint64_t)ft->num_slots);
    ft->offset = 0u;
    ft->payload = 0u;
    ft->padding = 0u;
    ft->frame_open = true;
    ft->frames_started++;
    return SFT_RESULT_OK;
}

static sft_result sft_end_frame(sft_frame_temp* ft) {
    if (!ft || !ft->valid) {
        return SFT_RESULT_NOT_INITIALIZED;
    }
    if (!ft->frame_open) {
        return SFT_RESULT_FRAME_NOT_OPEN;
    }
    if (ft->offset > ft->peak_used) {
        ft->peak_used = ft->offset;
    }
    ft->frame_open = false;
    ft->frames_ended++;
    return SFT_RESULT_OK;
}

static sft_result sft_alloc(sft_frame_temp* ft, size_t size, size_t alignment, sft_block* out) {
    if (!ft || !out) {
        return SFT_RESULT_INVALID_ARGUMENT;
    }
    if (!ft->valid) {
        return SFT_RESULT_NOT_INITIALIZED;
    }
    if (!ft->frame_open) {
        return SFT_RESULT_FRAME_NOT_OPEN;
    }
    if ((size == 0u) || !sft_is_pow2(alignment) || (alignment > SFT_MAX_ALIGNMENT)) {
        return SFT_RESULT_INVALID_ARGUMENT;
    }

    const size_t aligned_offset = sft_align_up(ft->offset, alignment);
    if ((aligned_offset > ft->slot_capacity) || (size > (ft->slot_capacity - aligned_offset))) {
        ft->overflows++;
        return SFT_RESULT_OVERFLOW;
    }

    uint8_t* ptr = ft->base + ((size_t)ft->cur_slot * ft->slot_capacity) + aligned_offset;
    if ((ft->flags & SFT_NO_CLEAR) == 0u) {
        memset(ptr, 0, size);
        ft->bytes_cleared += (uint64_t)size;
    }

    out->ptr = ptr;
    out->size = size;
    out->slot = ft->cur_slot;
    out->_reserved = 0u;
    out->generation = ft->generation;
    ft->padding += aligned_offset - ft->offset;
    ft->offset = aligned_offset + size;
    ft->payload += size;
    ft->allocations++;
    ft->bytes_allocated += (uint64_t)size;
    return SFT_RESULT_OK;
}

static void* sft_block_ptr(const sft_frame_temp* ft, const sft_block* block) {
    if (!ft || !block || !ft->valid || !ft->frame_open) {
        return NULL;
    }
    if ((block->generation != ft->generation) || (block->slot != ft->cur_slot) || (block->size == 0u)) {
        return NULL;
    }
    const uintptr_t slot_base = (uintptr_t)(ft->base + ((size_t)block->slot * ft->slot_capacity));
    const uintptr_t ptr = (uintptr_t)block->ptr;
    if ((ptr < slot_base) || (ptr > (slot_base + ft->slot_capacity))) {
        return NULL;
    }
    const size_t offset = (size_t)(ptr - slot_base);
    if (block->size > (ft->slot_capacity - offset)) {
        return NULL;
    }
    return block->ptr;
}

static sft_stats sft_query_stats(const sft_frame_temp* ft) {
    sft_stats stats;
    memset(&stats, 0, sizeof(stats));
    if (!ft || !ft->valid) {
        return stats;
    }
    stats.capacity = ft->capacity;
    stats.slot_capacity = ft->slot_capacity;
    stats.backing_bytes = ft->capacity + (size_t)SFT_MAX_ALIGNMENT - 1u;
    stats.used = ft->offset;
    stats.peak_used = (ft->offset > ft->peak_used) ? ft->offset : ft->peak_used;
    stats.payload = ft->payload;
    stats.padding = ft->padding;
    stats.frames_started = ft->frames_started;
    stats.frames_ended = ft->frames_ended;
    stats.allocations = ft->allocations;
    stats.overflows = ft->overflows;
    stats.bytes_allocated = ft->bytes_allocated;
    stats.bytes_cleared = ft->bytes_cleared;
    return stats;
}

typedef struct sft_test_allocator {
    int allocs;
    int frees;
    int live;
    bool fail;
} sft_test_allocator;

static void* sft_test_alloc(size_t size, void* user_data) {
    sft_test_allocator* state = (sft_test_allocator*)user_data;
    if (state->fail) {
        return NULL;
    }
    void* ptr = malloc(size);
    if (ptr) {
        state->allocs++;
        state->live++;
    }
    return ptr;
}

static void sft_test_free(void* ptr, void* user_data) {
    sft_test_allocator* state = (sft_test_allocator*)user_data;
    state->frees++;
    state->live--;
    free(ptr);
}

static uint8_t sft_pattern(size_t index, uint8_t seed) {
    return (uint8_t)(seed ^ (uint8_t)(index * 131u));
}

static void sft_fill(void* ptr, size_t size, uint8_t seed) {
    uint8_t* bytes = (uint8_t*)ptr;
    for (size_t i = 0u; i < size; i++) {
        bytes[i] = sft_pattern(i, seed);
    }
}

static bool sft_verify(const void* ptr, size_t size, uint8_t seed) {
    const uint8_t* bytes = (const uint8_t*)ptr;
    for (size_t i = 0u; i < size; i++) {
        if (bytes[i] != sft_pattern(i, seed)) {
            return false;
        }
    }
    return true;
}

static bool sft_is_zero(const void* ptr, size_t size) {
    const uint8_t* bytes = (const uint8_t*)ptr;
    for (size_t i = 0u; i < size; i++) {
        if (bytes[i] != 0u) {
            return false;
        }
    }
    return true;
}

UTEST(sokol_frame_temp, lifecycle) {
    sft_frame_temp ft = SFT_FRAME_TEMP_INIT;
    T(sft_init(&ft, &(sft_desc){0}) == SFT_RESULT_OK);
    const sft_stats init_stats = sft_query_stats(&ft);
    T(init_stats.capacity == (SFT_DEFAULT_CAPACITY & ~((size_t)SFT_MAX_ALIGNMENT - 1u)));
    T(init_stats.slot_capacity == (init_stats.capacity / SFT_DEFAULT_FRAME_SLOTS));
    T(init_stats.backing_bytes == (init_stats.capacity + SFT_MAX_ALIGNMENT - 1u));
    T(sft_init(&ft, &(sft_desc){0}) == SFT_RESULT_ALREADY_INITIALIZED);

    T(sft_begin_frame(&ft) == SFT_RESULT_OK);
    T(sft_begin_frame(&ft) == SFT_RESULT_FRAME_ALREADY_OPEN);
    sft_block block;
    T(sft_alloc(&ft, 16u, 16u, &block) == SFT_RESULT_OK);
    T(block.ptr != NULL);
    T(sft_block_ptr(&ft, &block) == block.ptr);
    T(sft_end_frame(&ft) == SFT_RESULT_OK);
    T(sft_end_frame(&ft) == SFT_RESULT_FRAME_NOT_OPEN);
    T(sft_block_ptr(&ft, &block) == NULL);
    T(sft_alloc(&ft, 16u, 16u, &block) == SFT_RESULT_FRAME_NOT_OPEN);

    const sft_stats stats = sft_query_stats(&ft);
    T(stats.frames_started == 1u);
    T(stats.frames_ended == 1u);
    T(stats.allocations == 1u);
    T(stats.overflows == 0u);
    sft_shutdown(&ft);
    sft_shutdown(&ft);
    T(sft_query_stats(&ft).capacity == 0u);
}

UTEST(sokol_frame_temp, generation_exhaustion) {
    sft_frame_temp ft = SFT_FRAME_TEMP_INIT;
    T(sft_init(&ft, &(sft_desc){ .capacity = 4096u, .frame_slots = 1u }) == SFT_RESULT_OK);
    ft.generation = UINT64_MAX;
    T(sft_begin_frame(&ft) == SFT_RESULT_GENERATION_EXHAUSTED);
    T(!ft.frame_open);
    T(sft_alloc(&ft, 1u, 1u, &(sft_block){0}) == SFT_RESULT_FRAME_NOT_OPEN);
    sft_shutdown(&ft);
}

UTEST(sokol_frame_temp, invalid_init_and_args) {
    sft_frame_temp ft = SFT_FRAME_TEMP_INIT;
    T(sft_init(NULL, &(sft_desc){0}) == SFT_RESULT_INVALID_ARGUMENT);
    T(sft_init(&ft, NULL) == SFT_RESULT_INVALID_ARGUMENT);
    T(sft_init(&ft, &(sft_desc){ .capacity = 1u, .frame_slots = 1u }) == SFT_RESULT_INVALID_ARGUMENT);
    T(sft_init(&ft, &(sft_desc){ .frame_slots = SFT_MAX_FRAME_SLOTS + 1u }) == SFT_RESULT_INVALID_ARGUMENT);
    T(sft_init(&ft, &(sft_desc){ .flags = 2u }) == SFT_RESULT_INVALID_ARGUMENT);

    sft_test_allocator alloc_state = {0};
    T(sft_init(&ft, &(sft_desc){
        .allocator = { .alloc_fn = sft_test_alloc, .user_data = &alloc_state }
    }) == SFT_RESULT_INVALID_ARGUMENT);
    T(alloc_state.allocs == 0);

    alloc_state.fail = true;
    T(sft_init(&ft, &(sft_desc){
        .allocator = {
            .alloc_fn = sft_test_alloc,
            .free_fn = sft_test_free,
            .user_data = &alloc_state,
        }
    }) == SFT_RESULT_OUT_OF_MEMORY);
    T(alloc_state.allocs == 0);
    T(alloc_state.frees == 0);
    T(alloc_state.live == 0);

    T(sft_init(&ft, &(sft_desc){ .capacity = 4096u, .frame_slots = 1u }) == SFT_RESULT_OK);
    sft_block block;
    T(sft_alloc(&ft, 1u, 1u, &block) == SFT_RESULT_FRAME_NOT_OPEN);
    T(sft_begin_frame(&ft) == SFT_RESULT_OK);
    T(sft_alloc(&ft, 0u, 1u, &block) == SFT_RESULT_INVALID_ARGUMENT);
    T(sft_alloc(&ft, 1u, 0u, &block) == SFT_RESULT_INVALID_ARGUMENT);
    T(sft_alloc(&ft, 1u, 3u, &block) == SFT_RESULT_INVALID_ARGUMENT);
    T(sft_alloc(&ft, 1u, SFT_MAX_ALIGNMENT * 2u, &block) == SFT_RESULT_INVALID_ARGUMENT);
    T(sft_alloc(&ft, 1u, 1u, NULL) == SFT_RESULT_INVALID_ARGUMENT);
    T(sft_query_stats(&ft).allocations == 0u);
    T(sft_end_frame(&ft) == SFT_RESULT_OK);
    sft_shutdown(&ft);
}

UTEST(sokol_frame_temp, alignment_exact_fit_and_overflow) {
    sft_frame_temp ft = SFT_FRAME_TEMP_INIT;
    T(sft_init(&ft, &(sft_desc){ .capacity = 4096u, .frame_slots = 1u }) == SFT_RESULT_OK);
    T(sft_begin_frame(&ft) == SFT_RESULT_OK);

    sft_block a;
    sft_block b;
    T(sft_alloc(&ft, 1u, 1u, &a) == SFT_RESULT_OK);
    T(sft_alloc(&ft, 100u, 64u, &b) == SFT_RESULT_OK);
    T(((uintptr_t)b.ptr % 64u) == 0u);
    const sft_stats padded = sft_query_stats(&ft);
    T(padded.used == 164u);
    T(padded.payload == 101u);
    T(padded.padding == 63u);

    const size_t remaining = padded.slot_capacity - padded.used;
    sft_block exact;
    T(sft_alloc(&ft, remaining, 1u, &exact) == SFT_RESULT_OK);
    const sft_stats full = sft_query_stats(&ft);
    T(full.used == full.slot_capacity);

    sft_block untouched;
    memset(&untouched, 0xA5, sizeof(untouched));
    const sft_block before = untouched;
    T(sft_alloc(&ft, 1u, 1u, &untouched) == SFT_RESULT_OVERFLOW);
    T(memcmp(&untouched, &before, sizeof(untouched)) == 0);
    const sft_stats overflowed = sft_query_stats(&ft);
    T(overflowed.used == full.used);
    T(overflowed.payload == full.payload);
    T(overflowed.padding == full.padding);
    T(overflowed.allocations == full.allocations);
    T(overflowed.overflows == 1u);

    T(sft_end_frame(&ft) == SFT_RESULT_OK);
    T(sft_begin_frame(&ft) == SFT_RESULT_OK);
    sft_block next_frame;
    T(sft_alloc(&ft, 4096u, 1u, &next_frame) == SFT_RESULT_OK);
    T(sft_end_frame(&ft) == SFT_RESULT_OK);
    sft_shutdown(&ft);
}

UTEST(sokol_frame_temp, frame_isolation_and_ring_wrap) {
    sft_frame_temp ft = SFT_FRAME_TEMP_INIT;
    T(sft_init(&ft, &(sft_desc){ .capacity = 3u * 512u, .frame_slots = 3u }) == SFT_RESULT_OK);

    sft_block first;
    void* first_ptr = NULL;
    T(sft_begin_frame(&ft) == SFT_RESULT_OK);
    T(sft_alloc(&ft, 32u, 16u, &first) == SFT_RESULT_OK);
    first_ptr = first.ptr;
    T(sft_is_zero(first.ptr, first.size));
    sft_fill(first.ptr, first.size, 0x11u);
    T(sft_end_frame(&ft) == SFT_RESULT_OK);
    T(sft_block_ptr(&ft, &first) == NULL);

    T(sft_begin_frame(&ft) == SFT_RESULT_OK);
    sft_block second;
    T(sft_alloc(&ft, 32u, 16u, &second) == SFT_RESULT_OK);
    T(second.ptr != first_ptr);
    T(sft_is_zero(second.ptr, second.size));
    sft_fill(second.ptr, second.size, 0x22u);
    T(sft_end_frame(&ft) == SFT_RESULT_OK);

    T(sft_begin_frame(&ft) == SFT_RESULT_OK);
    sft_block third;
    T(sft_alloc(&ft, 32u, 16u, &third) == SFT_RESULT_OK);
    T(third.ptr != first_ptr);
    T(third.ptr != second.ptr);
    T(sft_is_zero(third.ptr, third.size));
    sft_fill(third.ptr, third.size, 0x33u);
    T(sft_end_frame(&ft) == SFT_RESULT_OK);

    T(sft_begin_frame(&ft) == SFT_RESULT_OK);
    sft_block wrapped;
    T(sft_alloc(&ft, 32u, 16u, &wrapped) == SFT_RESULT_OK);
    T(wrapped.ptr == first_ptr);
    T(sft_block_ptr(&ft, &first) == NULL);
    T(sft_block_ptr(&ft, &wrapped) == wrapped.ptr);
    T(sft_is_zero(wrapped.ptr, wrapped.size));
    sft_fill(wrapped.ptr, wrapped.size, 0x44u);
    T(sft_verify(wrapped.ptr, wrapped.size, 0x44u));
    T(sft_end_frame(&ft) == SFT_RESULT_OK);

    const sft_stats stats = sft_query_stats(&ft);
    T(stats.frames_started == 4u);
    T(stats.frames_ended == 4u);
    T(stats.allocations == 4u);
    T(stats.peak_used == 32u);
    sft_shutdown(&ft);
}

UTEST(sokol_frame_temp, overflow_recovery) {
    sft_frame_temp ft = SFT_FRAME_TEMP_INIT;
    sft_test_allocator alloc_state = {0};
    T(sft_init(&ft, &(sft_desc){
        .capacity = 2u * 256u,
        .frame_slots = 2u,
        .allocator = {
            .alloc_fn = sft_test_alloc,
            .free_fn = sft_test_free,
            .user_data = &alloc_state,
        },
    }) == SFT_RESULT_OK);
    T(alloc_state.allocs == 1);

    for (int frame = 0; frame < 1000; frame++) {
        T(sft_begin_frame(&ft) == SFT_RESULT_OK);
        sft_block full;
        T(sft_alloc(&ft, 256u, 1u, &full) == SFT_RESULT_OK);
        sft_block over;
        T(sft_alloc(&ft, 1u, 1u, &over) == SFT_RESULT_OVERFLOW);
        T(sft_end_frame(&ft) == SFT_RESULT_OK);
    }
    const sft_stats stats = sft_query_stats(&ft);
    T(stats.frames_started == 1000u);
    T(stats.allocations == 1000u);
    T(stats.overflows == 1000u);
    T(stats.peak_used == 256u);
    T(alloc_state.allocs == 1);
    T(alloc_state.frees == 0);
    T(alloc_state.live == 1);
    sft_shutdown(&ft);
    T(alloc_state.frees == 1);
    T(alloc_state.live == 0);
}

UTEST(sokol_frame_temp, capacity_rounding) {
    sft_frame_temp ft = SFT_FRAME_TEMP_INIT;
    T(sft_init(&ft, &(sft_desc){ .capacity = 1000u, .frame_slots = 3u }) == SFT_RESULT_OK);
    const sft_stats stats = sft_query_stats(&ft);
    T(stats.slot_capacity == 256u);
    T(stats.capacity == 768u);
    T(stats.backing_bytes == (768u + SFT_MAX_ALIGNMENT - 1u));
    sft_shutdown(&ft);
}

typedef struct sft_stress_block {
    size_t size;
    size_t alignment;
} sft_stress_block;

static bool sft_stress_frame(sft_frame_temp* ft, uint64_t frame_index, sft_block* previous) {
    if (sft_begin_frame(ft) != SFT_RESULT_OK) {
        return false;
    }
    if (previous->ptr && (sft_block_ptr(ft, previous) != NULL)) {
        return false;
    }

    static const sft_stress_block spec[] = {
        { 1u, 1u },
        { 33u, 16u },
        { 257u, 64u },
        { 1024u, 256u },
        { 4096u, 16u },
    };
    sft_block blocks[sizeof(spec) / sizeof(spec[0])];
    const uint8_t seed = (uint8_t)(frame_index * 17u);
    for (size_t i = 0u; i < (sizeof(spec) / sizeof(spec[0])); i++) {
        if (sft_alloc(ft, spec[i].size, spec[i].alignment, &blocks[i]) != SFT_RESULT_OK) {
            return false;
        }
        if (((uintptr_t)blocks[i].ptr % spec[i].alignment) != 0u) {
            return false;
        }
        if (sft_block_ptr(ft, &blocks[i]) != blocks[i].ptr) {
            return false;
        }
        if (!sft_is_zero(blocks[i].ptr, blocks[i].size)) {
            return false;
        }
        sft_fill(blocks[i].ptr, blocks[i].size, (uint8_t)(seed + (uint8_t)i));
        if (!sft_verify(blocks[i].ptr, blocks[i].size, (uint8_t)(seed + (uint8_t)i))) {
            return false;
        }
    }

    for (size_t i = 1u; i < (sizeof(spec) / sizeof(spec[0])); i++) {
        const uint8_t* prev_end = (const uint8_t*)blocks[i - 1u].ptr + blocks[i - 1u].size;
        if ((const uint8_t*)blocks[i].ptr < prev_end) {
            return false;
        }
    }

    *previous = blocks[0];
    return sft_end_frame(ft) == SFT_RESULT_OK;
}

UTEST(sokol_frame_temp, hundred_thousand_frames_no_leak_no_bleed) {
    enum { FRAME_COUNT = 100000, SLOT_BYTES = 16384 };
    sft_test_allocator alloc_state = {0};
    sft_frame_temp ft = SFT_FRAME_TEMP_INIT;
    T(sft_init(&ft, &(sft_desc){
        .capacity = 2u * SLOT_BYTES,
        .frame_slots = 2u,
        .allocator = {
            .alloc_fn = sft_test_alloc,
            .free_fn = sft_test_free,
            .user_data = &alloc_state,
        },
    }) == SFT_RESULT_OK);
    T(alloc_state.allocs == 1);
    T(alloc_state.frees == 0);
    T(alloc_state.live == 1);

    sft_block previous;
    memset(&previous, 0, sizeof(previous));
    bool ok = true;
    for (uint64_t frame = 1u; frame <= FRAME_COUNT; frame++) {
        if (!sft_stress_frame(&ft, frame, &previous)) {
            ok = false;
            break;
        }
        if (alloc_state.allocs != 1) {
            ok = false;
            break;
        }
    }
    T(ok);

    const sft_stats stats = sft_query_stats(&ft);
    T(stats.frames_started == FRAME_COUNT);
    T(stats.frames_ended == FRAME_COUNT);
    T(stats.allocations == (uint64_t)FRAME_COUNT * 5u);
    T(stats.overflows == 0u);
    T(stats.capacity == 2u * SLOT_BYTES);
    T(stats.slot_capacity == SLOT_BYTES);
    T(stats.peak_used == 5632u);
    T(stats.payload == 5411u);
    T(stats.padding == 221u);
    T(stats.bytes_allocated == (uint64_t)FRAME_COUNT * (1u + 33u + 257u + 1024u + 4096u));
    T(stats.bytes_cleared == stats.bytes_allocated);
    T(alloc_state.allocs == 1);
    T(alloc_state.frees == 0);
    T(alloc_state.live == 1);

    sft_shutdown(&ft);
    T(alloc_state.frees == 1);
    T(alloc_state.live == 0);
}
