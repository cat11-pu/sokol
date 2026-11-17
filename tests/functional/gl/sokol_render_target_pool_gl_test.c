//------------------------------------------------------------------------------
//  LLM maintained.
//
//  sokol_render_target_pool_gl_test.c
//
//  Pixel-level checks for the render-target pool against the GL mock.
//  The mock keeps an RGBA8 content shadow and applies glClearBufferfv() to the
//  currently attached color texture, so recycled targets can be read back.
//------------------------------------------------------------------------------
#define SOKOL_EXTERNAL_GL_LOADER
#include "gl_mock.h"
#include "sokol_gfx.h"
#include "render_target_pool.h"
#include "utest.h"
#include <string.h>

#define T(b) EXPECT_TRUE(b)

static int num_error_logs;

static void reset_log(void) {
    num_error_logs = 0;
}

static void capture_log(const char* tag, uint32_t log_level, uint32_t log_item_id, const char* msg, uint32_t line_nr, const char* file, void* ud) {
    (void)tag; (void)log_item_id; (void)msg; (void)line_nr; (void)file; (void)ud;
    if (log_level <= 1) {
        num_error_logs++;
    }
}

static bool no_errors(void) {
    return num_error_logs == 0;
}

static void setup(void) {
    if (sg_isvalid()) {
        sg_shutdown();
        gl_mock_shutdown();
    }
    reset_log();
    gl_mock_setup();
    sg_setup(&(sg_desc){
        .environment.defaults = {
            .color_format = SG_PIXELFORMAT_RGBA8,
            .depth_format = SG_PIXELFORMAT_DEPTH_STENCIL,
            .sample_count = 1,
        },
        .logger.func = capture_log,
    });
}

static bool teardown(void) {
    sg_shutdown();
    const bool no_leaks = gl_mock_live_objects_total() == 0;
    gl_mock_shutdown();
    return no_leaks;
}

static bool read_target_rgba8(sg_image image, uint8_t* pixels, size_t size) {
    const sg_gl_image_info info = sg_gl_query_image_info(image);
    if ((info.active_slot < 0) || (info.active_slot >= SG_NUM_INFLIGHT_FRAMES)) {
        return false;
    }
    return gl_mock_texture_read_rgba8(info.tex[info.active_slot], pixels, size);
}

UTEST(sokol_render_target_pool_gl, thousand_scene_pixel_isolation) {
    setup();
    srtp_pool* pool = srtp_create(&(srtp_pool_desc){
        .max_targets = 8,
        .frames_in_flight = 2,
        .color_format = SG_PIXELFORMAT_RGBA8,
        .sample_count = 1,
    });
    T(pool != 0);
    const srtp_target_desc sizes[4] = {
        { .width = 16, .height = 16 },
        { .width = 32, .height = 16 },
        { .width = 16, .height = 32 },
        { .width = 8, .height = 8 },
    };
    uint8_t readback[32 * 32 * 4];
    for (int scene = 0; scene < 1000; scene++) {
        const int slot = scene & 3;
        const uint8_t expected[4] = {
            (uint8_t)(scene & 0xFF),
            (uint8_t)((scene * 3) & 0xFF),
            (uint8_t)((scene * 7) & 0xFF),
            0xFF,
        };
        srtp_begin_frame(pool);
        srtp_target target = srtp_acquire(pool, &sizes[slot]);
        T(target.id != SRTP_INVALID_ID);
        srtp_target_info info;
        T(srtp_query_target(pool, target, &info));
        T(info.width == sizes[slot].width);
        T(info.height == sizes[slot].height);
        T(info.color_format == SG_PIXELFORMAT_RGBA8);
        sg_pass pass = srtp_clear_pass(&info, (sg_color){
            .r = (float)expected[0] / 255.0f,
            .g = (float)expected[1] / 255.0f,
            .b = (float)expected[2] / 255.0f,
            .a = 1.0f,
        });
        T(pass.action.colors[0].load_action == SG_LOADACTION_CLEAR);
        T(pass.action.colors[0].store_action == SG_STOREACTION_STORE);
        sg_begin_pass(&pass);
        sg_end_pass();
        const size_t num_bytes = (size_t)info.width * (size_t)info.height * 4;
        T(num_bytes <= sizeof(readback));
        T(read_target_rgba8(info.image, readback, num_bytes));
        bool pixels_match = true;
        for (size_t i = 0; i < num_bytes; i += 4) {
            if ((readback[i + 0] != expected[0])
                || (readback[i + 1] != expected[1])
                || (readback[i + 2] != expected[2])
                || (readback[i + 3] != expected[3])) {
                pixels_match = false;
                break;
            }
        }
        T(pixels_match);
        srtp_release(pool, target);
        srtp_end_frame(pool);
        sg_commit();
    }
    srtp_begin_frame(pool);
    srtp_end_frame(pool);
    srtp_begin_frame(pool);
    srtp_end_frame(pool);
    const srtp_pool_stats stats = srtp_query_stats(pool);
    T(stats.acquires == 1000);
    T(stats.reuses == 996);
    T(stats.creates == 4);
    T(stats.releases == 1000);
    T(stats.invalidations == 0);
    T(stats.stale_rejects == 0);
    T(stats.exhausted == 0);
    T(stats.live == 0);
    T(stats.retired == 0);
    T(stats.free_count == 4);
    T(stats.peak_live == 1);
    srtp_destroy(pool);
    sg_commit();
    T(gl_mock_live_objects(GL_MOCK_OBJ_TEXTURE) == 0);
    T(no_errors());
    T(teardown());
}

UTEST(sokol_render_target_pool_gl, invalidation_rebuild_pixels) {
    setup();
    srtp_pool* pool = srtp_create(&(srtp_pool_desc){
        .max_targets = 2,
        .frames_in_flight = 2,
    });
    srtp_begin_frame(pool);
    srtp_target first = srtp_acquire(pool, &(srtp_target_desc){ .width = 7, .height = 5 });
    srtp_target_info first_info;
    T(srtp_query_target(pool, first, &first_info));
    sg_pass first_pass = srtp_clear_pass(&first_info, (sg_color){ .r = 1.0f, .g = 0.0f, .b = 0.0f, .a = 1.0f });
    sg_begin_pass(&first_pass);
    sg_end_pass();
    uint8_t pixels[7 * 5 * 4];
    T(read_target_rgba8(first_info.image, pixels, sizeof(pixels)));
    T(pixels[0] == 255 && pixels[1] == 0 && pixels[2] == 0 && pixels[3] == 255);
    srtp_invalidate(pool, first);
    srtp_end_frame(pool);
    srtp_begin_frame(pool);
    srtp_end_frame(pool);
    srtp_begin_frame(pool);
    srtp_end_frame(pool);
    srtp_target rebuilt = srtp_acquire(pool, &(srtp_target_desc){ .width = 7, .height = 5 });
    srtp_target_info rebuilt_info;
    T(srtp_query_target(pool, rebuilt, &rebuilt_info));
    T(rebuilt_info.image.id != first_info.image.id);
    sg_pass rebuilt_pass = srtp_clear_pass(&rebuilt_info, (sg_color){ .r = 0.0f, .g = 1.0f, .b = 0.0f, .a = 1.0f });
    sg_begin_pass(&rebuilt_pass);
    sg_end_pass();
    T(read_target_rgba8(rebuilt_info.image, pixels, sizeof(pixels)));
    T(pixels[0] == 0 && pixels[1] == 255 && pixels[2] == 0 && pixels[3] == 255);
    srtp_release(pool, rebuilt);
    srtp_end_frame(pool);
    const srtp_pool_stats stats = srtp_query_stats(pool);
    T(stats.invalidations == 1);
    T(stats.creates == 2);
    T(stats.destroys == 1);
    srtp_destroy(pool);
    T(no_errors());
    T(teardown());
}
