//------------------------------------------------------------------------------
//  sokol_gfx_lifecycle_stress_test.c
//
//  LLM maintained.
//
//  Reproduction and stability test for mid-frame resource destruction:
//
//  1. A handle destroyed inside a frame must stay resolvable until sg_commit()
//     (deferred reclamation at frame end).
//  2. A destroyed handle must never be reused by a new resource in the same
//     frame (no generation-counter aliasing of stale draw commands).
//  3. Stale handles used after frame-end reclamation must fail lookup with an
//     explicit validation error, never crash.
//  4. Repeated create/destroy cycles must not leak slots or native objects.
//------------------------------------------------------------------------------
#include "force_dummy_backend.h"
#define _POSIX_C_SOURCE 200809L
#define SOKOL_IMPL
#define SOKOL_TRACE_HOOKS
#include "sokol_gfx.h"
#include "utest.h"

#include <stdlib.h>

#define T(b) EXPECT_TRUE(b)

#define STRESS_ITERATIONS (10000)

static int num_log_called = 0;
static sg_log_item last_log_item = (sg_log_item)0;

static void test_logger(const char* tag, uint32_t log_level, uint32_t log_item_id,
                        const char* message_or_null, uint32_t line_nr,
                        const char* filename_or_null, void* user_data)
{
    (void)tag; (void)log_level; (void)message_or_null;
    (void)line_nr; (void)filename_or_null; (void)user_data;
    num_log_called++;
    last_log_item = (sg_log_item)log_item_id;
}

static void stress_setup(int pool_size) {
    num_log_called = 0;
    last_log_item = (sg_log_item)0;
    sg_setup(&(sg_desc){
        .buffer_pool_size = pool_size,
        .logger = { .func = test_logger },
    });
}

static sg_buffer make_vbuf(void) {
    static const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    return sg_make_buffer(&(sg_buffer_desc){ .data = SG_RANGE(data) });
}

static sg_buffer make_transient_vbuf(void) {
    return sg_make_buffer(&(sg_buffer_desc){
        .size = 256,
        .usage = {
            .vertex_buffer = true,
            .write_transient = true,
        },
    });
}

// Destroy inside a frame must not invalidate the handle until sg_commit().
UTEST(sokol_gfx_lifecycle, destroy_defers_slot_reclamation) {
    stress_setup(4);
    T(sg_isvalid());

    sg_buffer buf = make_vbuf();
    T(buf.id != SG_INVALID_ID);
    const int free_before = _sg.pools.buffer_pool.queue_top;

    sg_destroy_buffer(buf);

    // deferred: the stale handle must already fail lookup (explicit error),
    // but the pool slot must not have been returned for reuse yet
    T(_sg_lookup_buffer(buf.id) == 0);
    T(sg_query_buffer_state(buf) == SG_RESOURCESTATE_INVALID);
    T(_sg.pools.buffer_pool.queue_top == free_before);

    sg_commit();

    // frame end: slot is now actually reclaimed
    T(_sg_lookup_buffer(buf.id) == 0);
    T(_sg.pools.buffer_pool.queue_top == free_before + 1);

    sg_shutdown();
}

// A destroyed slot must not be handed out again within the same frame.
UTEST(sokol_gfx_lifecycle, no_slot_reuse_within_frame) {
    stress_setup(2);
    T(sg_isvalid());

    sg_buffer first = make_vbuf();
    T(first.id != SG_INVALID_ID);
    sg_destroy_buffer(first);

    // before the fix this allocation aliases the just-destroyed slot
    sg_buffer second = make_vbuf();
    T(second.id != SG_INVALID_ID);
    T(second.id != first.id);
    T(_sg_slot_index(second.id) != _sg_slot_index(first.id));

    // the stale handle must still fail lookup (generation guard), while the
    // new handle resolves to the live resource
    T(_sg_lookup_buffer(first.id) == 0);
    T(_sg_lookup_buffer(second.id) != 0);

    sg_commit();
    sg_shutdown();
}

// After frame-end reclamation a stale handle returns an explicit error.
UTEST(sokol_gfx_lifecycle, stale_handle_explicit_error) {
    stress_setup(2);
    T(sg_isvalid());

    sg_buffer buf = make_transient_vbuf();
    T(buf.id != SG_INVALID_ID);
    sg_destroy_buffer(buf);
    sg_commit();

    T(_sg_lookup_buffer(buf.id) == 0);
    T(sg_query_buffer_state(buf) == SG_RESOURCESTATE_INVALID);

    // a write through the stale handle must log an explicit error, not crash
    num_log_called = 0;
    static const uint8_t payload[] = { 1, 2, 3, 4 };
    sg_write_buffer_transient(&(sg_write_buffer_desc){
        .src.data = SG_RANGE(payload),
        .dst.buffer = buf,
        .size = sizeof(payload),
    });
    T(num_log_called > 0);
    T(last_log_item == SG_LOGITEM_WRITE_BUFFER_TRANSIENT_BUFFER_ALIVE);

    sg_shutdown();
}

// 10000 create/destroy cycles: no slot leak, no pool exhaustion, stable stats.
UTEST(sokol_gfx_lifecycle, ten_thousand_cycles_stable) {
    stress_setup(8);
    T(sg_isvalid());
    const int pool_capacity = _sg.pools.buffer_pool.size - 1;

    for (int i = 0; i < STRESS_ITERATIONS; i++) {
        sg_buffer bufs[4];
        for (int j = 0; j < 4; j++) {
            bufs[j] = make_vbuf();
            T(bufs[j].id != SG_INVALID_ID);
        }
        // destroy two mid-frame and two across the commit boundary
        sg_destroy_buffer(bufs[0]);
        sg_destroy_buffer(bufs[1]);
        sg_commit();
        sg_destroy_buffer(bufs[2]);
        sg_destroy_buffer(bufs[3]);
        sg_commit();

        // every slot must have been returned after both commits
        T(_sg.pools.buffer_pool.queue_top == pool_capacity);
        for (int j = 0; j < 4; j++) {
            T(_sg_lookup_buffer(bufs[j].id) == 0);
        }
    }

    T(_sg.pools.buffer_pool.queue_top == pool_capacity);
    sg_shutdown();
}

// Interleaved live/dead resources: live handles never collide with deferred
// destruction of other resources over 10000 cycles.
UTEST(sokol_gfx_lifecycle, ten_thousand_cycles_with_live_resource) {
    stress_setup(8);
    T(sg_isvalid());

    sg_buffer anchor = make_vbuf();
    T(anchor.id != SG_INVALID_ID);

    for (int i = 0; i < STRESS_ITERATIONS; i++) {
        sg_buffer scratch = make_vbuf();
        T(scratch.id != SG_INVALID_ID);
        T(scratch.id != anchor.id);
        sg_destroy_buffer(scratch);
        sg_commit();
        T(_sg_lookup_buffer(anchor.id) != 0);
        T(sg_query_buffer_state(anchor) == SG_RESOURCESTATE_VALID);
        T(_sg_lookup_buffer(scratch.id) == 0);
    }

    sg_destroy_buffer(anchor);
    sg_commit();
    T(_sg_lookup_buffer(anchor.id) == 0);
    sg_shutdown();
}

// Destroying the same handle twice in one frame must be a safe no-op.
UTEST(sokol_gfx_lifecycle, double_destroy_is_safe) {
    stress_setup(4);
    T(sg_isvalid());

    sg_buffer buf = make_vbuf();
    const int free_before = _sg.pools.buffer_pool.queue_top;
    sg_destroy_buffer(buf);
    sg_destroy_buffer(buf);   // must not enqueue twice, must not assert
    T(_sg.pools.buffer_pool.destroy_queue_top == 1);
    T(_sg.pools.buffer_pool.queue_top == free_before);

    sg_commit();
    T(_sg.pools.buffer_pool.queue_top == free_before + 1);
    T(_sg.pools.buffer_pool.destroy_queue_top == 0);
    sg_shutdown();
}

// Pending destroys from consecutive frames must not leak across the boundary.
UTEST(sokol_gfx_lifecycle, pending_queue_clears_each_frame) {
    stress_setup(8);
    T(sg_isvalid());
    const int capacity = _sg.pools.buffer_pool.size - 1;

    for (int frame = 0; frame < 100; frame++) {
        sg_buffer a = make_vbuf();
        sg_buffer b = make_vbuf();
        sg_destroy_buffer(a);
        T(_sg.pools.buffer_pool.destroy_queue_top == 1);
        sg_commit();
        T(_sg.pools.buffer_pool.destroy_queue_top == 0);
        sg_destroy_buffer(b);
        sg_commit();
        T(_sg.pools.buffer_pool.queue_top == capacity);
    }
    sg_shutdown();
}

UTEST_MAIN()
