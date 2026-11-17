//------------------------------------------------------------------------------
//  LLM maintained.
//
//  sokol_render_target_pool_test.c
//------------------------------------------------------------------------------
#include "force_dummy_backend.h"
#include "sokol_gfx.h"
#include "capture_log.h"
#include "render_target_pool.h"
#include "utest.h"

#define T(b) EXPECT_TRUE(b)

static void init(void) {
    capture_log_reset();
    sg_setup(&(sg_desc){ .logger = { .func = capture_log } });
}

static void shutdown_pool_test(void) {
    sg_shutdown();
}

static void begin_and_end(srtp_pool* pool) {
    srtp_begin_frame(pool);
    srtp_end_frame(pool);
}

UTEST(sokol_render_target_pool, defaults_and_lifecycle) {
    init();
    srtp_pool* pool = srtp_create(0);
    T(pool != 0);
    srtp_pool_stats stats = srtp_query_stats(pool);
    T(stats.live == 0);
    T(stats.free_count == 0);
    T(stats.creates == 0);
    srtp_begin_frame(pool);
    srtp_target target = srtp_acquire(pool, &(srtp_target_desc){
        .width = 64,
        .height = 32,
    });
    T(target.id != SRTP_INVALID_ID);
    srtp_target_info info;
    T(srtp_query_target(pool, target, &info));
    T(info.width == 64);
    T(info.height == 32);
    T(info.color_format == SG_PIXELFORMAT_RGBA8);
    T(info.sample_count == 1);
    T(info.epoch == 1);
    T(info.image.id != SG_INVALID_ID);
    T(info.color_attachment.id != SG_INVALID_ID);
    T(info.texture.id != SG_INVALID_ID);
    srtp_release(pool, target);
    srtp_end_frame(pool);
    stats = srtp_query_stats(pool);
    T(stats.acquires == 1);
    T(stats.creates == 1);
    T(stats.releases == 1);
    T(stats.deferred_retires == 1);
    T(stats.live == 0);
    T(stats.retired == 1);
    srtp_destroy(pool);
    T(capture_log_num_items == 0);
    shutdown_pool_test();
}

UTEST(sokol_render_target_pool, exact_key_reuse) {
    init();
    srtp_pool* pool = srtp_create(&(srtp_pool_desc){
        .max_targets = 4,
        .frames_in_flight = 2,
    });
    srtp_begin_frame(pool);
    srtp_target first = srtp_acquire(pool, &(srtp_target_desc){ .width = 96, .height = 64 });
    srtp_target_info first_info;
    T(srtp_query_target(pool, first, &first_info));
    srtp_release(pool, first);
    srtp_end_frame(pool);
    begin_and_end(pool);
    begin_and_end(pool);
    srtp_target second = srtp_acquire(pool, &(srtp_target_desc){ .width = 96, .height = 64 });
    srtp_target_info second_info;
    T(srtp_query_target(pool, second, &second_info));
    T(second_info.image.id == first_info.image.id);
    T(second_info.epoch == first_info.epoch + 1);
    srtp_pool_stats stats = srtp_query_stats(pool);
    T(stats.creates == 1);
    T(stats.reuses == 1);
    T(stats.deferred_purges == 1);
    srtp_release(pool, second);
    srtp_end_frame(pool);
    srtp_destroy(pool);
    T(capture_log_num_items == 0);
    shutdown_pool_test();
}

UTEST(sokol_render_target_pool, deferred_reclaim_is_not_same_frame_reuse) {
    init();
    srtp_pool* pool = srtp_create(&(srtp_pool_desc){
        .max_targets = 4,
        .frames_in_flight = 2,
    });
    srtp_begin_frame(pool);
    srtp_target first = srtp_acquire(pool, &(srtp_target_desc){ .width = 32, .height = 32 });
    srtp_target_info first_info;
    T(srtp_query_target(pool, first, &first_info));
    srtp_release(pool, first);
    srtp_end_frame(pool);
    begin_and_end(pool);
    srtp_target second = srtp_acquire(pool, &(srtp_target_desc){ .width = 32, .height = 32 });
    srtp_target_info second_info;
    T(srtp_query_target(pool, second, &second_info));
    T(second.id != first.id);
    T(second_info.image.id != first_info.image.id);
    srtp_release(pool, second);
    srtp_end_frame(pool);
    begin_and_end(pool);
    srtp_target third = srtp_acquire(pool, &(srtp_target_desc){ .width = 32, .height = 32 });
    srtp_target_info third_info;
    T(srtp_query_target(pool, third, &third_info));
    T(third_info.image.id == first_info.image.id);
    srtp_pool_stats stats = srtp_query_stats(pool);
    T(stats.creates == 2);
    T(stats.reuses == 1);
    srtp_release(pool, third);
    srtp_end_frame(pool);
    srtp_destroy(pool);
    T(capture_log_num_items == 0);
    shutdown_pool_test();
}

UTEST(sokol_render_target_pool, size_and_format_are_partition_keys) {
    init();
    srtp_pool* pool = srtp_create(&(srtp_pool_desc){
        .max_targets = 8,
        .frames_in_flight = 1,
    });
    const srtp_target_desc keys[3] = {
        { .width = 64, .height = 64, .color_format = SG_PIXELFORMAT_RGBA8 },
        { .width = 64, .height = 64, .color_format = SG_PIXELFORMAT_RGBA16F },
        { .width = 128, .height = 64, .color_format = SG_PIXELFORMAT_RGBA8 },
    };
    srtp_target_info first_infos[3];
    for (int i = 0; i < 3; i++) {
        srtp_begin_frame(pool);
        srtp_target target = srtp_acquire(pool, &keys[i]);
        T(target.id != SRTP_INVALID_ID);
        T(srtp_query_target(pool, target, &first_infos[i]));
        srtp_release(pool, target);
        srtp_end_frame(pool);
        begin_and_end(pool);
    }
    for (int i = 0; i < 3; i++) {
        srtp_begin_frame(pool);
        srtp_target target = srtp_acquire(pool, &keys[i]);
        srtp_target_info info;
        T(srtp_query_target(pool, target, &info));
        T(info.image.id == first_infos[i].image.id);
        T(info.width == keys[i].width);
        T(info.height == keys[i].height);
        T(info.color_format == keys[i].color_format);
        srtp_release(pool, target);
        srtp_end_frame(pool);
    }
    srtp_pool_stats stats = srtp_query_stats(pool);
    T(stats.creates == 3);
    T(stats.reuses == 3);
    srtp_destroy(pool);
    T(capture_log_num_items == 0);
    shutdown_pool_test();
}

UTEST(sokol_render_target_pool, invalidate_rebuilds_on_next_acquire) {
    init();
    srtp_pool* pool = srtp_create(&(srtp_pool_desc){
        .max_targets = 4,
        .frames_in_flight = 2,
    });
    srtp_begin_frame(pool);
    srtp_target target = srtp_acquire(pool, &(srtp_target_desc){ .width = 48, .height = 24 });
    srtp_target_info old_info;
    T(srtp_query_target(pool, target, &old_info));
    srtp_invalidate(pool, target);
    T(!srtp_query_target(pool, target, &old_info));
    srtp_end_frame(pool);
    begin_and_end(pool);
    begin_and_end(pool);
    srtp_target rebuilt = srtp_acquire(pool, &(srtp_target_desc){ .width = 48, .height = 24 });
    srtp_target_info new_info;
    T(srtp_query_target(pool, rebuilt, &new_info));
    T(new_info.image.id != old_info.image.id);
    T(new_info.epoch == 1);
    srtp_pool_stats stats = srtp_query_stats(pool);
    T(stats.invalidations == 1);
    T(stats.creates == 2);
    T(stats.destroys == 1);
    T(stats.deferred_purges == 1);
    srtp_release(pool, rebuilt);
    srtp_end_frame(pool);
    srtp_destroy(pool);
    T(capture_log_num_items == 0);
    shutdown_pool_test();
}

UTEST(sokol_render_target_pool, invalidate_all_rebuilds_live_targets) {
    init();
    srtp_pool* pool = srtp_create(&(srtp_pool_desc){
        .max_targets = 4,
        .frames_in_flight = 2,
    });
    srtp_begin_frame(pool);
    const srtp_target_desc keys[2] = {
        { .width = 16, .height = 16 },
        { .width = 32, .height = 32 },
    };
    srtp_target old_targets[2];
    srtp_target_info old_infos[2];
    for (int i = 0; i < 2; i++) {
        old_targets[i] = srtp_acquire(pool, &keys[i]);
        T(srtp_query_target(pool, old_targets[i], &old_infos[i]));
    }
    srtp_invalidate_all(pool);
    for (int i = 0; i < 2; i++) {
        srtp_target_info info;
        T(!srtp_query_target(pool, old_targets[i], &info));
    }
    srtp_end_frame(pool);
    begin_and_end(pool);
    begin_and_end(pool);
    for (int i = 0; i < 2; i++) {
        srtp_target rebuilt = srtp_acquire(pool, &keys[i]);
        srtp_target_info info;
        T(srtp_query_target(pool, rebuilt, &info));
        T(info.image.id != old_infos[i].image.id);
        srtp_release(pool, rebuilt);
    }
    srtp_end_frame(pool);
    const srtp_pool_stats stats = srtp_query_stats(pool);
    T(stats.invalidations == 2);
    T(stats.creates == 4);
    T(stats.destroys == 2);
    T(stats.deferred_purges == 2);
    T(stats.stale_rejects == 2);
    srtp_destroy(pool);
    T(capture_log_num_items == 0);
    shutdown_pool_test();
}

UTEST(sokol_render_target_pool, stale_targets_are_rejected) {
    init();
    srtp_pool* pool = srtp_create(&(srtp_pool_desc){
        .max_targets = 2,
        .frames_in_flight = 1,
    });
    srtp_begin_frame(pool);
    srtp_target target = srtp_acquire(pool, &(srtp_target_desc){ .width = 16, .height = 16 });
    srtp_target_info info;
    T(srtp_query_target(pool, target, &info));
    srtp_release(pool, target);
    T(!srtp_query_target(pool, target, &info));
    srtp_release(pool, target);
    srtp_invalidate(pool, target);
    srtp_target invalid = { .id = 0xFFFFFFFFu };
    T(!srtp_query_target(pool, invalid, &info));
    srtp_pool_stats stats = srtp_query_stats(pool);
    T(stats.stale_rejects == 4);
    srtp_end_frame(pool);
    srtp_destroy(pool);
    T(capture_log_num_items == 0);
    shutdown_pool_test();
}

UTEST(sokol_render_target_pool, exhaustion_is_explicit) {
    init();
    srtp_pool* pool = srtp_create(&(srtp_pool_desc){
        .max_targets = 1,
        .frames_in_flight = 1,
    });
    srtp_begin_frame(pool);
    srtp_target first = srtp_acquire(pool, &(srtp_target_desc){ .width = 16, .height = 16 });
    T(first.id != SRTP_INVALID_ID);
    srtp_target second = srtp_acquire(pool, &(srtp_target_desc){ .width = 16, .height = 16 });
    T(second.id == SRTP_INVALID_ID);
    srtp_pool_stats stats = srtp_query_stats(pool);
    T(stats.exhausted == 1);
    T(stats.creates == 1);
    srtp_release(pool, first);
    srtp_end_frame(pool);
    begin_and_end(pool);
    srtp_target third = srtp_acquire(pool, &(srtp_target_desc){ .width = 16, .height = 16 });
    T(third.id != SRTP_INVALID_ID);
    stats = srtp_query_stats(pool);
    T(stats.reuses == 1);
    srtp_release(pool, third);
    srtp_end_frame(pool);
    srtp_destroy(pool);
    T(capture_log_num_items == 0);
    shutdown_pool_test();
}

UTEST(sokol_render_target_pool, invalid_size_is_rejected) {
    init();
    srtp_pool* pool = srtp_create(0);
    srtp_begin_frame(pool);
    T(srtp_acquire(pool, &(srtp_target_desc){ .width = 0, .height = 16 }).id == SRTP_INVALID_ID);
    T(srtp_acquire(pool, &(srtp_target_desc){ .width = 16, .height = 0 }).id == SRTP_INVALID_ID);
    T(srtp_acquire(pool, &(srtp_target_desc){ .width = 16, .height = 16, .color_format = (sg_pixel_format)9999 }).id == SRTP_INVALID_ID);
    T(srtp_acquire(pool, 0).id == SRTP_INVALID_ID);
    srtp_pool_stats stats = srtp_query_stats(pool);
    T(stats.invalid_desc == 4);
    srtp_end_frame(pool);
    srtp_destroy(pool);
    T(capture_log_num_items == 0);
    shutdown_pool_test();
}

UTEST(sokol_render_target_pool, thousand_scene_switch_no_leak) {
    init();
    srtp_pool* pool = srtp_create(&(srtp_pool_desc){
        .max_targets = 8,
        .frames_in_flight = 2,
        .color_format = SG_PIXELFORMAT_RGBA8,
        .sample_count = 1,
    });
    const srtp_target_desc sizes[4] = {
        { .width = 16, .height = 16 },
        { .width = 32, .height = 16 },
        { .width = 16, .height = 32 },
        { .width = 8, .height = 8 },
    };
    uint64_t last_epoch[4] = {0};
    for (int scene = 0; scene < 1000; scene++) {
        const int slot = scene & 3;
        srtp_begin_frame(pool);
        srtp_target target = srtp_acquire(pool, &sizes[slot]);
        T(target.id != SRTP_INVALID_ID);
        srtp_target_info info;
        T(srtp_query_target(pool, target, &info));
        T(info.epoch > last_epoch[slot]);
        last_epoch[slot] = info.epoch;
        const sg_pass pass = srtp_clear_pass(&info, (sg_color){
            .r = (float)(scene & 0xFF) / 255.0f,
            .g = (float)((scene * 3) & 0xFF) / 255.0f,
            .b = (float)((scene * 7) & 0xFF) / 255.0f,
            .a = 1.0f,
        });
        T(pass.action.colors[0].load_action == SG_LOADACTION_CLEAR);
        T(pass.action.colors[0].store_action == SG_STOREACTION_STORE);
        T(pass.attachments.colors[0].id == info.color_attachment.id);
        sg_begin_pass(&pass);
        sg_end_pass();
        srtp_release(pool, target);
        srtp_end_frame(pool);
        sg_commit();
    }
    begin_and_end(pool);
    begin_and_end(pool);
    srtp_pool_stats stats = srtp_query_stats(pool);
    T(stats.acquires == 1000);
    T(stats.reuses == 996);
    T(stats.creates == 4);
    T(stats.releases == 1000);
    T(stats.deferred_retires == 1000);
    T(stats.deferred_purges == 1000);
    T(stats.invalidations == 0);
    T(stats.stale_rejects == 0);
    T(stats.exhausted == 0);
    T(stats.invalid_desc == 0);
    T(stats.create_failures == 0);
    T(stats.live == 0);
    T(stats.retired == 0);
    T(stats.free_count == 4);
    T(stats.peak_live == 1);
    srtp_destroy(pool);
    T(capture_log_num_items == 0);
    shutdown_pool_test();
}
