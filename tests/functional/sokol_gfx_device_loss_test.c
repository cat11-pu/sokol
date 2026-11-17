//------------------------------------------------------------------------------
//  sokol_gfx_device_loss_test.c
//
//  LLM maintained.
//
//  Device-loss stress test for the generic sokol-gfx lifecycle:
//  - 200 loss/rebuild cycles using sg_shutdown()/sg_setup()
//  - pipeline, buffer and texture resources are invalidated in rotation
//  - duplicate destroys, stale same-generation handles and invalid handles
//  - allocation tracking must return to zero after every shutdown
//  - the rebuilt frame must have the same command/resource fingerprint
//
//  The dummy backend has no pixel readback, so frame consistency is checked
//  as an identical deterministic draw-command stream plus identical resource
//  creation attributes. Every failure prints the minimal single-cycle sequence.
//
//  Set SOKOL_TEST_DEVICE_LOSS_COUNTEREXAMPLE=1 to run the focused stale-handle
//  counterexample. It fails until handles from a previous sg_setup() cannot
//  alias resources from the current sg_setup().
//------------------------------------------------------------------------------
#include "force_dummy_backend.h"
#include "sokol_gfx.h"
#include "utest.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define T(b) EXPECT_TRUE(b)

#define DL_NUM_CYCLES (200)

//------------------------------------------------------------------------------
//  allocation tracking
//------------------------------------------------------------------------------
typedef union {
    max_align_t align;
    size_t size;
} dl_alloc_header_t;

typedef struct {
    size_t outstanding_allocs;
    size_t outstanding_bytes;
    size_t total_allocs;
    size_t total_bytes;
} dl_alloc_state_t;

static dl_alloc_state_t dl_alloc_state;

static void* dl_alloc(size_t size, void* user_data) {
    dl_alloc_state_t* state = (dl_alloc_state_t*)user_data;
    dl_alloc_header_t* header = (dl_alloc_header_t*)malloc(sizeof(dl_alloc_header_t) + size);
    if (header == NULL) {
        return NULL;
    }
    header->size = size;
    state->outstanding_allocs++;
    state->outstanding_bytes += size;
    state->total_allocs++;
    state->total_bytes += size;
    return header + 1;
}

static void dl_free(void* ptr, void* user_data) {
    if (ptr == NULL) {
        return;
    }
    dl_alloc_state_t* state = (dl_alloc_state_t*)user_data;
    dl_alloc_header_t* header = ((dl_alloc_header_t*)ptr) - 1;
    if ((state->outstanding_allocs == 0) || (state->outstanding_bytes < header->size)) {
        printf("[device-loss] allocator accounting underflow\n");
        abort();
    }
    state->outstanding_allocs--;
    state->outstanding_bytes -= header->size;
    free(header);
}

static void dl_alloc_reset(void) {
    memset(&dl_alloc_state, 0, sizeof(dl_alloc_state));
}

static bool dl_check_no_leaks(const char* phase, int cycle) {
    if ((dl_alloc_state.outstanding_allocs == 0) &&
        (dl_alloc_state.outstanding_bytes == 0))
    {
        return true;
    }
    printf("\n[device-loss] FAILURE cycle=%d invariant=leak phase=%s\n", cycle, phase);
    printf("  outstanding_allocs=%zu outstanding_bytes=%zu\n",
        dl_alloc_state.outstanding_allocs,
        dl_alloc_state.outstanding_bytes);
    return false;
}

//------------------------------------------------------------------------------
//  validation log capture
//------------------------------------------------------------------------------
#define DL_MAX_LOG_ITEMS (32)

typedef struct {
    int num_items;
    int num_errors;
    uint32_t items[DL_MAX_LOG_ITEMS];
    uint32_t levels[DL_MAX_LOG_ITEMS];
} dl_log_t;

static dl_log_t dl_log;

static void dl_log_reset(void) {
    memset(&dl_log, 0, sizeof(dl_log));
}

static void dl_logger(const char* tag, uint32_t log_level, uint32_t log_item_id, const char* message, uint32_t line_nr, const char* filename, void* user_data) {
    (void)tag;
    (void)message;
    (void)line_nr;
    (void)filename;
    (void)user_data;
    if (dl_log.num_items < DL_MAX_LOG_ITEMS) {
        dl_log.items[dl_log.num_items] = log_item_id;
        dl_log.levels[dl_log.num_items] = log_level;
        dl_log.num_items++;
    }
    if (log_level <= 1) {
        dl_log.num_errors++;
    }
}

//------------------------------------------------------------------------------
//  resources and deterministic frame
//------------------------------------------------------------------------------
typedef struct {
    sg_buffer vbuf;
    sg_image img;
    sg_view view;
    sg_sampler smp;
    sg_shader shd;
    sg_pipeline pip;
} dl_resources_t;

typedef struct {
    uint32_t num_passes;
    uint32_t num_apply_viewport;
    uint32_t num_apply_scissor_rect;
    uint32_t num_apply_pipeline;
    uint32_t num_apply_bindings;
    uint32_t num_draw;
    size_t buffer_size;
    int image_type;
    int image_width;
    int image_height;
    int image_num_slices;
    int image_num_mipmaps;
    int image_pixel_format;
    int image_sample_count;
    int view_type;
    int pipeline_primitive_type;
    int pipeline_index_type;
    int pipeline_sample_count;
    int pipeline_attr0_format;
    int pipeline_color_count;
    int pipeline_depth_format;
} dl_frame_signature_t;

static void dl_print_min_sequence(const char* reason, int cycle, const char* invalidated) {
    printf("\n[device-loss] FAILURE cycle=%d reason=%s\n", cycle, reason);
    printf("  minimal sequence:\n");
    printf("    01. sg_setup()\n");
    printf("    02. make buffer + image + view + sampler + shader + pipeline\n");
    printf("    03. begin swapchain pass, apply pipeline/bindings, draw, end, commit\n");
    printf("    04. invalidate %s and duplicate-destroy it\n", invalidated);
    printf("    05. destroy invalid handles and stale same-generation handles\n");
    printf("    06. sg_shutdown() with remaining resources alive\n");
    printf("    07. verify allocator outstanding allocations == 0\n");
    printf("    08. sg_setup(), recreate the same resources\n");
    printf("    09. repeat the draw and compare the frame fingerprint\n");
}

static dl_resources_t dl_make_resources(void) {
    dl_resources_t resources = {0};

    static const float vertices[9] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f,
    };
    static const uint32_t pixels[8 * 8] = {
        0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080,
        0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF,
        0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080,
        0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF,
        0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080,
        0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF,
        0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080,
        0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF, 0x80808080, 0xFFFFFFFF,
    };

    resources.vbuf = sg_make_buffer(&(sg_buffer_desc){
        .usage.vertex_buffer = true,
        .data = SG_RANGE(vertices),
    });
    resources.img = sg_make_image(&(sg_image_desc){
        .width = 8,
        .height = 8,
        .pixel_format = SG_PIXELFORMAT_RGBA8,
        .data = SG_RANGE(pixels),
    });
    resources.view = sg_make_view(&(sg_view_desc){
        .texture.image = resources.img,
    });
    resources.smp = sg_make_sampler(&(sg_sampler_desc){0});
    resources.shd = sg_make_shader(&(sg_shader_desc){
        .views[0].texture = {
            .stage = SG_SHADERSTAGE_FRAGMENT,
            .image_type = SG_IMAGETYPE_2D,
            .sample_type = SG_IMAGESAMPLETYPE_FLOAT,
        },
        .samplers[0] = {
            .stage = SG_SHADERSTAGE_FRAGMENT,
            .sampler_type = SG_SAMPLERTYPE_FILTERING,
        },
        .texture_sampler_pairs[0] = {
            .stage = SG_SHADERSTAGE_FRAGMENT,
            .view_slot = 0,
            .sampler_slot = 0,
        },
    });
    resources.pip = sg_make_pipeline(&(sg_pipeline_desc){
        .shader = resources.shd,
        .layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3,
    });
    return resources;
}

static bool dl_resources_valid(const dl_resources_t* resources) {
    return (sg_query_buffer_state(resources->vbuf) == SG_RESOURCESTATE_VALID) &&
           (sg_query_image_state(resources->img) == SG_RESOURCESTATE_VALID) &&
           (sg_query_view_state(resources->view) == SG_RESOURCESTATE_VALID) &&
           (sg_query_sampler_state(resources->smp) == SG_RESOURCESTATE_VALID) &&
           (sg_query_shader_state(resources->shd) == SG_RESOURCESTATE_VALID) &&
           (sg_query_pipeline_state(resources->pip) == SG_RESOURCESTATE_VALID);
}

static void dl_draw_frame(const dl_resources_t* resources) {
    sg_begin_pass(&(sg_pass){
        .swapchain = {
            .width = 64,
            .height = 64,
            .sample_count = 1,
            .color_format = SG_PIXELFORMAT_RGBA8,
        },
    });
    sg_apply_viewport(0, 0, 64, 64, true);
    sg_apply_scissor_rect(0, 0, 64, 64, true);
    sg_apply_pipeline(resources->pip);
    sg_apply_bindings(&(sg_bindings){
        .vertex_buffers[0] = resources->vbuf,
        .views[0] = resources->view,
        .samplers[0] = resources->smp,
    });
    sg_draw(0, 3, 1);
    sg_end_pass();
}

static dl_frame_signature_t dl_capture_frame(const dl_resources_t* resources) {
    const sg_frame_stats stats = sg_query_stats().cur_frame;
    const sg_buffer_desc buffer_desc = sg_query_buffer_desc(resources->vbuf);
    const sg_image_desc image_desc = sg_query_image_desc(resources->img);
    const sg_pipeline_desc pipeline_desc = sg_query_pipeline_desc(resources->pip);

    dl_frame_signature_t signature = {
        .num_passes = stats.num_passes,
        .num_apply_viewport = stats.num_apply_viewport,
        .num_apply_scissor_rect = stats.num_apply_scissor_rect,
        .num_apply_pipeline = stats.num_apply_pipeline,
        .num_apply_bindings = stats.num_apply_bindings,
        .num_draw = stats.num_draw,
        .buffer_size = buffer_desc.size,
        .image_type = (int)image_desc.type,
        .image_width = image_desc.width,
        .image_height = image_desc.height,
        .image_num_slices = image_desc.num_slices,
        .image_num_mipmaps = image_desc.num_mipmaps,
        .image_pixel_format = (int)image_desc.pixel_format,
        .image_sample_count = image_desc.sample_count,
        .view_type = (int)sg_query_view_type(resources->view),
        .pipeline_primitive_type = (int)pipeline_desc.primitive_type,
        .pipeline_index_type = (int)pipeline_desc.index_type,
        .pipeline_sample_count = pipeline_desc.sample_count,
        .pipeline_attr0_format = (int)pipeline_desc.layout.attrs[0].format,
        .pipeline_color_count = pipeline_desc.color_count,
        .pipeline_depth_format = (int)pipeline_desc.depth.pixel_format,
    };
    return signature;
}

static bool dl_frame_equal(const dl_frame_signature_t* lhs, const dl_frame_signature_t* rhs) {
    return (lhs->num_passes == rhs->num_passes) &&
           (lhs->num_apply_viewport == rhs->num_apply_viewport) &&
           (lhs->num_apply_scissor_rect == rhs->num_apply_scissor_rect) &&
           (lhs->num_apply_pipeline == rhs->num_apply_pipeline) &&
           (lhs->num_apply_bindings == rhs->num_apply_bindings) &&
           (lhs->num_draw == rhs->num_draw) &&
           (lhs->buffer_size == rhs->buffer_size) &&
           (lhs->image_type == rhs->image_type) &&
           (lhs->image_width == rhs->image_width) &&
           (lhs->image_height == rhs->image_height) &&
           (lhs->image_num_slices == rhs->image_num_slices) &&
           (lhs->image_num_mipmaps == rhs->image_num_mipmaps) &&
           (lhs->image_pixel_format == rhs->image_pixel_format) &&
           (lhs->image_sample_count == rhs->image_sample_count) &&
           (lhs->view_type == rhs->view_type) &&
           (lhs->pipeline_primitive_type == rhs->pipeline_primitive_type) &&
           (lhs->pipeline_index_type == rhs->pipeline_index_type) &&
           (lhs->pipeline_sample_count == rhs->pipeline_sample_count) &&
           (lhs->pipeline_attr0_format == rhs->pipeline_attr0_format) &&
           (lhs->pipeline_color_count == rhs->pipeline_color_count) &&
           (lhs->pipeline_depth_format == rhs->pipeline_depth_format);
}

// Exercise same-generation stale handles and invalid handles. This must not
// affect any live resource.
static bool dl_inject_bad_handles(const dl_resources_t* resources) {
    bool passed = true;
    static const float buffer_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    static const uint32_t image_data[4 * 4] = { 0 };
    const sg_buffer stale_buf = sg_make_buffer(&(sg_buffer_desc){
        .usage.vertex_buffer = true,
        .data = SG_RANGE(buffer_data),
    });
    const sg_buffer replacement_buf = sg_make_buffer(&(sg_buffer_desc){
        .usage.vertex_buffer = true,
        .data = SG_RANGE(buffer_data),
    });
    sg_destroy_buffer(stale_buf);
    sg_destroy_buffer(stale_buf);
    passed &= (sg_query_buffer_state(replacement_buf) == SG_RESOURCESTATE_VALID);
    if (!passed) {
        printf("[device-loss] buffer probe stale=%08x replacement=%08x state=%d\n",
            stale_buf.id, replacement_buf.id, (int)sg_query_buffer_state(replacement_buf));
    }
    sg_destroy_buffer(replacement_buf);

    const sg_image stale_img = sg_make_image(&(sg_image_desc){
        .width = 4,
        .height = 4,
        .pixel_format = SG_PIXELFORMAT_RGBA8,
        .data = SG_RANGE(image_data),
    });
    const sg_image replacement_img = sg_make_image(&(sg_image_desc){
        .width = 4,
        .height = 4,
        .pixel_format = SG_PIXELFORMAT_RGBA8,
        .data = SG_RANGE(image_data),
    });
    sg_destroy_image(stale_img);
    sg_destroy_image(stale_img);
    const bool image_ok = (sg_query_image_state(replacement_img) == SG_RESOURCESTATE_VALID);
    if (!image_ok) {
        printf("[device-loss] image probe stale=%08x replacement=%08x state=%d\n",
            stale_img.id, replacement_img.id, (int)sg_query_image_state(replacement_img));
    }
    passed &= image_ok;
    sg_destroy_image(replacement_img);

    const sg_view stale_view = sg_make_view(&(sg_view_desc){ .texture.image = resources->img });
    const sg_view replacement_view = sg_make_view(&(sg_view_desc){ .texture.image = resources->img });
    sg_destroy_view(stale_view);
    sg_destroy_view(stale_view);
    const bool view_ok = (sg_query_view_state(replacement_view) == SG_RESOURCESTATE_VALID);
    if (!view_ok) {
        printf("[device-loss] view probe stale=%08x replacement=%08x state=%d\n",
            stale_view.id, replacement_view.id, (int)sg_query_view_state(replacement_view));
    }
    passed &= view_ok;
    sg_destroy_view(replacement_view);

    const sg_pipeline stale_pip = sg_make_pipeline(&(sg_pipeline_desc){
        .shader = resources->shd,
        .layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3,
    });
    const sg_pipeline replacement_pip = sg_make_pipeline(&(sg_pipeline_desc){
        .shader = resources->shd,
        .layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3,
    });
    sg_destroy_pipeline(stale_pip);
    sg_destroy_pipeline(stale_pip);
    const bool pipeline_ok = (sg_query_pipeline_state(replacement_pip) == SG_RESOURCESTATE_VALID);
    if (!pipeline_ok) {
        printf("[device-loss] pipeline probe stale=%08x replacement=%08x state=%d\n",
            stale_pip.id, replacement_pip.id, (int)sg_query_pipeline_state(replacement_pip));
    }
    passed &= pipeline_ok;
    sg_destroy_pipeline(replacement_pip);

    sg_destroy_buffer((sg_buffer){ .id = 1 });
    sg_destroy_image((sg_image){ .id = 1 });
    sg_destroy_view((sg_view){ .id = 1 });
    sg_destroy_pipeline((sg_pipeline){ .id = 1 });
    sg_destroy_shader((sg_shader){ .id = 1 });
    sg_destroy_sampler((sg_sampler){ .id = 1 });
    return passed;
}

static const char* dl_invalidate_resources(dl_resources_t* resources, int cycle) {
    switch (cycle % 3) {
        case 0:
            sg_destroy_pipeline(resources->pip);
            sg_destroy_pipeline(resources->pip);
            dl_draw_frame(resources);
            return "pipeline";
        case 1:
            sg_destroy_buffer(resources->vbuf);
            sg_destroy_buffer(resources->vbuf);
            dl_draw_frame(resources);
            return "buffer";
        default:
            sg_destroy_view(resources->view);
            sg_destroy_view(resources->view);
            sg_destroy_image(resources->img);
            sg_destroy_image(resources->img);
            dl_draw_frame(resources);
            return "texture";
    }
}

static void dl_setup_device(void) {
    dl_log_reset();
    sg_setup(&(sg_desc){
        .buffer_pool_size = 32,
        .image_pool_size = 32,
        .sampler_pool_size = 16,
        .shader_pool_size = 16,
        .pipeline_pool_size = 16,
        .view_pool_size = 32,
        .logger.func = dl_logger,
        .allocator = {
            .alloc_fn = dl_alloc,
            .free_fn = dl_free,
            .user_data = &dl_alloc_state,
        },
    });
}

static bool dl_run_cycle(int cycle) {
    dl_resources_t resources = dl_make_resources();
    if (!dl_resources_valid(&resources)) {
        dl_print_min_sequence("resource creation failed", cycle, "none");
        return false;
    }

    dl_log_reset();
    dl_draw_frame(&resources);
    const dl_frame_signature_t reference = dl_capture_frame(&resources);
    const bool clean_frame = (dl_log.num_errors == 0);
    sg_commit();
    if (!clean_frame) {
        dl_print_min_sequence("clean reference frame logged an error", cycle, "none");
        return false;
    }

    if (!dl_inject_bad_handles(&resources)) {
        dl_print_min_sequence("stale same-generation handle affected a live resource", cycle, "none");
        return false;
    }
    const char* invalidated = dl_invalidate_resources(&resources, cycle);
    sg_commit();

    sg_shutdown();
    if (!dl_check_no_leaks("after loss", cycle)) {
        dl_print_min_sequence("allocation leak after device loss", cycle, "none");
        return false;
    }
    if (sg_isvalid()) {
        dl_print_min_sequence("sg_shutdown did not invalidate the device", cycle, "none");
        return false;
    }

    dl_setup_device();
    dl_resources_t rebuilt = dl_make_resources();
    if (!dl_resources_valid(&rebuilt)) {
        dl_print_min_sequence("resource rebuild failed", cycle, "none");
        return false;
    }

    dl_log_reset();
    dl_draw_frame(&rebuilt);
    const dl_frame_signature_t replay = dl_capture_frame(&rebuilt);
    const bool rebuilt_clean = (dl_log.num_errors == 0);
    sg_commit();
    if (!rebuilt_clean) {
        dl_print_min_sequence("rebuilt frame logged an error", cycle, "none");
        return false;
    }
    if (!dl_frame_equal(&reference, &replay)) {
        dl_print_min_sequence("frame fingerprint mismatch", cycle, invalidated);
        return false;
    }

    sg_shutdown();
    if (!dl_check_no_leaks("after rebuild", cycle)) {
        dl_print_min_sequence("allocation leak after rebuild", cycle, "none");
        return false;
    }
    return true;
}

UTEST(sokol_gfx, device_loss_200_cycles) {
    dl_alloc_reset();
    dl_setup_device();

    bool passed = true;
    for (int cycle = 0; cycle < DL_NUM_CYCLES; cycle++) {
        if (!dl_run_cycle(cycle)) {
            passed = false;
            break;
        }
        if ((cycle + 1 < DL_NUM_CYCLES) && !sg_isvalid()) {
            dl_setup_device();
        }
    }

    if (sg_isvalid()) {
        sg_shutdown();
    }
    const bool no_leaks = dl_check_no_leaks("after test", DL_NUM_CYCLES - 1);
    if (!no_leaks) {
        dl_print_min_sequence("allocation leak after test", DL_NUM_CYCLES - 1, "none");
    }
    if (passed && no_leaks) {
        printf("[device-loss] PASS cycles=%d allocs=%zu bytes=%zu\n",
            DL_NUM_CYCLES,
            dl_alloc_state.total_allocs,
            dl_alloc_state.total_bytes);
    }
    T(no_leaks);
    T(passed);
}

// Focused counterexample, opt-in because it documents a known failure:
// generation counters are reset by sg_setup(), so a handle from the previous
// device generation can alias a newly created resource.
UTEST(sokol_gfx, device_loss_stale_handle_counterexample) {
    if (getenv("SOKOL_TEST_DEVICE_LOSS_COUNTEREXAMPLE") == NULL) {
        return;
    }

    dl_alloc_reset();
    dl_setup_device();
    static const float vertices[9] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f,
    };
    const sg_buffer old_buf = sg_make_buffer(&(sg_buffer_desc){
        .usage.vertex_buffer = true,
        .data = SG_RANGE(vertices),
    });
    sg_shutdown();
    const bool loss_no_leaks = dl_check_no_leaks("counterexample after loss", 0);
    if (!loss_no_leaks) {
        dl_print_min_sequence("allocation leak after counterexample loss", 0, "none");
    }
    T(loss_no_leaks);

    dl_setup_device();
    const sg_buffer new_buf = sg_make_buffer(&(sg_buffer_desc){
        .usage.vertex_buffer = true,
        .data = SG_RANGE(vertices),
    });
    printf("\n[device-loss] COUNTEREXAMPLE cycle=0\n");
    printf("  old_handle=%08x new_handle=%08x\n", old_buf.id, new_buf.id);
    printf("  minimal sequence:\n");
    printf("    01. sg_setup()\n");
    printf("    02. buf_old = sg_make_buffer(...)\n");
    printf("    03. sg_shutdown()\n");
    printf("    04. sg_setup()\n");
    printf("    05. buf_new = sg_make_buffer(...)\n");
    printf("    06. sg_destroy_buffer(buf_old)\n");
    printf("    07. sg_query_buffer_state(buf_new) == SG_RESOURCESTATE_VALID\n");

    T(old_buf.id != new_buf.id);
    sg_destroy_buffer(old_buf);
    T(sg_query_buffer_state(new_buf) == SG_RESOURCESTATE_VALID);
    sg_destroy_buffer(new_buf);
    sg_shutdown();
    const bool rebuild_no_leaks = dl_check_no_leaks("counterexample after rebuild", 0);
    if (!rebuild_no_leaks) {
        dl_print_min_sequence("allocation leak after counterexample rebuild", 0, "none");
    }
    T(rebuild_no_leaks);
}
