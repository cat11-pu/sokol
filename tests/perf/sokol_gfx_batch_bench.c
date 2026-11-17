// LLM maintained.
//
// sokol_gfx_batch_bench.c
//
// Measures three application-level batching strategies with the real GL
// backend:
//
//   raw:     one pipeline- and one bindings-update per draw, random order
//   sorted:  draws sorted by pipeline and binding, redundant updates skipped
//   merged:  sorted draws with contiguous same-binding ranges merged
//
// The scene is a fixed 64x64 grid of non-overlapping quads. All strategies
// render the same object->pipeline/texture mapping, so the final frame must
// be byte-identical.
//
// Linux/GLCORE only. The executable creates an EGL pbuffer, runs each strategy
// for a number of frames with a glFinish() after every frame, and writes the
// per-frame timings as CSV.

#define _POSIX_C_SOURCE 200809L
#define SOKOL_IMPL
#define SOKOL_GLCORE

#include "sokol_gfx.h"

#include <EGL/egl.h>
#include <GL/gl.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BENCH_WIDTH (512)
#define BENCH_HEIGHT (512)
#define GRID_WIDTH (64)
#define GRID_HEIGHT (64)
#define NUM_OBJECTS (GRID_WIDTH * GRID_HEIGHT)
#define NUM_SHADERS (4)
#define NUM_BLEND_STATES (4)
#define NUM_PIPELINES (NUM_SHADERS * NUM_BLEND_STATES)
#define NUM_TEXTURES (32)
#define NUM_GROUPS (NUM_PIPELINES * NUM_TEXTURES)
#define OBJECTS_PER_GROUP (NUM_OBJECTS / NUM_GROUPS)
#define DEFAULT_FRAMES (120)
#define DEFAULT_WARMUP_FRAMES (16)

_Static_assert((NUM_OBJECTS % NUM_GROUPS) == 0, "object count must fill all batch groups");

typedef enum bench_mode {
    BENCH_MODE_RAW,
    BENCH_MODE_SORTED,
    BENCH_MODE_MERGED,
    BENCH_MODE_NUM,
} bench_mode;

typedef struct bench_result {
    double frame_ms[DEFAULT_FRAMES];
    int num_frames;
    uint32_t draw_calls;
    uint32_t pipeline_applies;
    uint32_t binding_applies;
    uint64_t pixel_hash;
    uint8_t* pixels;
    sg_frame_stats stats;
} bench_result;

typedef struct pipeline_entry {
    int shader_index;
    int blend_state_index;
    sg_pipeline pip;
} pipeline_entry;

typedef struct asset_state {
    sg_buffer vbuf;
    sg_buffer ibuf;
    sg_shader shaders[NUM_SHADERS];
    sg_image images[NUM_TEXTURES];
    sg_view views[NUM_TEXTURES];
    sg_sampler sampler;
    sg_bindings bindings[NUM_TEXTURES];
    sg_pipeline object_pipeline[NUM_OBJECTS];
    int object_texture[NUM_OBJECTS];
    int object_slot[NUM_OBJECTS];
    int sorted_object[NUM_OBJECTS];
    int raw_order[NUM_OBJECTS];
    int group_start[NUM_GROUPS];
    int group_count[NUM_GROUPS];
    pipeline_entry pipeline_cache[NUM_PIPELINES];
    int pipeline_cache_count;
    int pipeline_cache_hits;
    int pipeline_cache_misses;
    sg_swapchain swapchain;
    sg_pass pass;
} asset_state;

static asset_state assets;
static EGLDisplay egl_display = EGL_NO_DISPLAY;
static EGLSurface egl_surface = EGL_NO_SURFACE;
static EGLContext egl_context = EGL_NO_CONTEXT;
static int error_count;
static uint32_t frame_draw_calls;
static uint32_t frame_pipeline_applies;
static uint32_t frame_binding_applies;

static void bench_logger(const char* tag, uint32_t log_level, uint32_t log_item_id,
                         const char* message_or_null, uint32_t line_nr,
                         const char* filename_or_null, void* user_data)
{
    (void)tag;
    (void)log_item_id;
    (void)user_data;
    if (log_level <= 1) {
        error_count++;
        fprintf(stderr, "sg error: %s (%s:%u)\n",
                message_or_null ? message_or_null : "unknown",
                filename_or_null ? filename_or_null : "sokol_gfx.h",
                (unsigned)line_nr);
    }
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((double)ts.tv_sec * 1000.0) + ((double)ts.tv_nsec * 0.000001);
}

static bool init_egl(void)
{
    egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_display == EGL_NO_DISPLAY) {
        fprintf(stderr, "eglGetDisplay failed\n");
        return false;
    }
    if (!eglInitialize(egl_display, NULL, NULL)) {
        fprintf(stderr, "eglInitialize failed (0x%x)\n", (unsigned)eglGetError());
        return false;
    }
    if (!eglBindAPI(EGL_OPENGL_API)) {
        fprintf(stderr, "eglBindAPI(EGL_OPENGL_API) failed\n");
        return false;
    }

    const EGLint config_attrs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 0,
        EGL_STENCIL_SIZE, 0,
        EGL_NONE,
    };
    EGLConfig config = NULL;
    EGLint num_configs = 0;
    if (!eglChooseConfig(egl_display, config_attrs, &config, 1, &num_configs) || (num_configs < 1)) {
        fprintf(stderr, "eglChooseConfig failed\n");
        return false;
    }

    const EGLint surface_attrs[] = {
        EGL_WIDTH, BENCH_WIDTH,
        EGL_HEIGHT, BENCH_HEIGHT,
        EGL_NONE,
    };
    egl_surface = eglCreatePbufferSurface(egl_display, config, surface_attrs);
    if (egl_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "eglCreatePbufferSurface failed (0x%x)\n", (unsigned)eglGetError());
        return false;
    }

    const EGLint context_attrs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE,
    };
    egl_context = eglCreateContext(egl_display, config, EGL_NO_CONTEXT, context_attrs);
    if (egl_context == EGL_NO_CONTEXT) {
        fprintf(stderr, "eglCreateContext failed (0x%x)\n", (unsigned)eglGetError());
        return false;
    }
    if (!eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)) {
        fprintf(stderr, "eglMakeCurrent failed (0x%x)\n", (unsigned)eglGetError());
        return false;
    }
    return true;
}

static void shutdown_egl(void)
{
    if (egl_display != EGL_NO_DISPLAY) {
        eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (egl_context != EGL_NO_CONTEXT) {
            eglDestroyContext(egl_display, egl_context);
        }
        if (egl_surface != EGL_NO_SURFACE) {
            eglDestroySurface(egl_display, egl_surface);
        }
        eglTerminate(egl_display);
    }
}

static sg_shader make_shader(int shader_index)
{
    static const char* vs_source =
        "#version 430 core\n"
        "layout(location = 0) in vec2 pos;\n"
        "layout(location = 1) in vec2 uv;\n"
        "out vec2 v_uv;\n"
        "void main() {\n"
        "    gl_Position = vec4(pos, 0.0, 1.0);\n"
        "    v_uv = uv;\n"
        "}\n";
    static const char* fs_source =
        "#version 430 core\n"
        "in vec2 v_uv;\n"
        "out vec4 frag_color;\n"
        "uniform sampler2D tex;\n"
        "void main() {\n"
        "    vec4 c = texture(tex, v_uv);\n"
        "    frag_color = vec4(c.rgb, c.a);\n"
        "}\n";
    (void)shader_index;
    return sg_make_shader(&(sg_shader_desc){
        .vertex_func = {
            .source = vs_source,
            .entry = "main",
        },
        .fragment_func = {
            .source = fs_source,
            .entry = "main",
        },
        .attrs = {
            [0] = { .base_type = SG_SHADERATTRBASETYPE_FLOAT, .glsl_name = "pos" },
            [1] = { .base_type = SG_SHADERATTRBASETYPE_FLOAT, .glsl_name = "uv" },
        },
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
            .glsl_name = "tex",
        },
    });
}

static sg_pipeline pipeline_cache_get(asset_state* state, int shader_index, int blend_state_index)
{
    for (int i = 0; i < state->pipeline_cache_count; i++) {
        const pipeline_entry* entry = &state->pipeline_cache[i];
        if ((entry->shader_index == shader_index) && (entry->blend_state_index == blend_state_index)) {
            state->pipeline_cache_hits++;
            return entry->pip;
        }
    }

    const float color_scale = (float)(blend_state_index + 1) / (float)(NUM_BLEND_STATES + 1);
    const sg_pipeline_desc desc = {
        .shader = state->shaders[shader_index],
        .layout = {
            .attrs = {
                [0] = { .format = SG_VERTEXFORMAT_FLOAT2 },
                [1] = { .format = SG_VERTEXFORMAT_FLOAT2 },
            },
        },
        .depth = {
            .pixel_format = SG_PIXELFORMAT_NONE,
            .compare = SG_COMPAREFUNC_ALWAYS,
            .bias = (float)blend_state_index * 0.25f,
            .bias_slope_scale = (float)blend_state_index * 0.5f,
        },
        .color_count = 1,
        .colors[0] = {
            .pixel_format = SG_PIXELFORMAT_RGBA8,
            .blend = { .enabled = false },
        },
        .primitive_type = SG_PRIMITIVETYPE_TRIANGLES,
        .index_type = SG_INDEXTYPE_UINT16,
        .cull_mode = SG_CULLMODE_NONE,
        .blend_color = { color_scale, color_scale, color_scale, 1.0f },
    };
    const sg_pipeline pip = sg_make_pipeline(&desc);
    if (state->pipeline_cache_count < NUM_PIPELINES) {
        pipeline_entry* entry = &state->pipeline_cache[state->pipeline_cache_count++];
        entry->shader_index = shader_index;
        entry->blend_state_index = blend_state_index;
        entry->pip = pip;
    }
    state->pipeline_cache_misses++;
    return pip;
}

static void make_scene(asset_state* state)
{
    typedef struct vertex {
        float x, y;
        float u, v;
    } vertex;
    vertex vertices[NUM_OBJECTS * 4];
    uint16_t indices[NUM_OBJECTS * 6];
    memset(vertices, 0, sizeof(vertices));
    memset(indices, 0, sizeof(indices));

    for (int i = 0; i < NUM_SHADERS; i++) {
        state->shaders[i] = make_shader(i);
    }

    for (int i = 0; i < NUM_TEXTURES; i++) {
        const uint8_t pixel[4] = {
            (uint8_t)(17 + i * 7),
            (uint8_t)(43 + i * 11),
            (uint8_t)(79 + i * 13),
            255,
        };
        state->images[i] = sg_make_image(&(sg_image_desc){
            .width = 1,
            .height = 1,
            .pixel_format = SG_PIXELFORMAT_RGBA8,
            .data = { .ptr = pixel, .size = sizeof(pixel) },
        });
        state->views[i] = sg_make_view(&(sg_view_desc){
            .texture.image = state->images[i],
        });
    }
    state->sampler = sg_make_sampler(&(sg_sampler_desc){
        .min_filter = SG_FILTER_NEAREST,
        .mag_filter = SG_FILTER_NEAREST,
        .mipmap_filter = SG_FILTER_NEAREST,
        .wrap_u = SG_WRAP_CLAMP_TO_EDGE,
        .wrap_v = SG_WRAP_CLAMP_TO_EDGE,
    });

    for (int i = 0; i < NUM_OBJECTS; i++) {
        const int group = i % NUM_GROUPS;
        const int pipeline_index = group / NUM_TEXTURES;
        const int shader_index = pipeline_index / NUM_BLEND_STATES;
        const int blend_state_index = pipeline_index % NUM_BLEND_STATES;
        state->object_pipeline[i] = pipeline_cache_get(state, shader_index, blend_state_index);
        state->object_texture[i] = group % NUM_TEXTURES;
    }

    const float cell_w = 2.0f / (float)GRID_WIDTH;
    const float cell_h = 2.0f / (float)GRID_HEIGHT;
    int cursor = 0;
    for (int group = 0; group < NUM_GROUPS; group++) {
        state->group_start[group] = cursor;
        state->group_count[group] = 0;
        for (int i = 0; i < NUM_OBJECTS; i++) {
            if ((i % NUM_GROUPS) == group) {
                state->object_slot[i] = cursor;
                state->sorted_object[cursor++] = i;
                state->group_count[group]++;
            }
        }
    }
    if (cursor != NUM_OBJECTS) {
        fprintf(stderr, "invalid sorted object count: %d\n", cursor);
        error_count++;
    }

    for (int slot = 0; slot < NUM_OBJECTS; slot++) {
        const int object = state->sorted_object[slot];
        const float x0 = -1.0f + (float)(object % GRID_WIDTH) * cell_w + cell_w * 0.125f;
        const float y0 = -1.0f + (float)(object / GRID_WIDTH) * cell_h + cell_h * 0.125f;
        const float x1 = x0 + cell_w * 0.75f;
        const float y1 = y0 + cell_h * 0.75f;
        vertex* v = &vertices[slot * 4];
        v[0] = (vertex){ x0, y0, 0.0f, 0.0f };
        v[1] = (vertex){ x1, y0, 1.0f, 0.0f };
        v[2] = (vertex){ x1, y1, 1.0f, 1.0f };
        v[3] = (vertex){ x0, y1, 0.0f, 1.0f };
        const uint16_t base = (uint16_t)(slot * 4);
        indices[slot * 6 + 0] = base;
        indices[slot * 6 + 1] = (uint16_t)(base + 1);
        indices[slot * 6 + 2] = (uint16_t)(base + 2);
        indices[slot * 6 + 3] = base;
        indices[slot * 6 + 4] = (uint16_t)(base + 2);
        indices[slot * 6 + 5] = (uint16_t)(base + 3);
    }
    state->vbuf = sg_make_buffer(&(sg_buffer_desc){
        .data = { .ptr = vertices, .size = sizeof(vertices) },
        .label = "batch-bench-vertices",
    });
    state->ibuf = sg_make_buffer(&(sg_buffer_desc){
        .usage = { .index_buffer = true },
        .data = { .ptr = indices, .size = sizeof(indices) },
        .label = "batch-bench-indices",
    });
    for (int i = 0; i < NUM_TEXTURES; i++) {
        state->bindings[i] = (sg_bindings){
            .vertex_buffers[0] = state->vbuf,
            .index_buffer = state->ibuf,
            .views[0] = state->views[i],
            .samplers[0] = state->sampler,
        };
    }

    uint32_t rng = 0x12345678u;
    for (int i = 0; i < NUM_OBJECTS; i++) {
        state->raw_order[i] = i;
    }
    for (int i = NUM_OBJECTS - 1; i > 0; i--) {
        rng = rng * 1664525u + 1013904223u;
        const int j = (int)(rng % (uint32_t)(i + 1));
        const int tmp = state->raw_order[i];
        state->raw_order[i] = state->raw_order[j];
        state->raw_order[j] = tmp;
    }

    state->swapchain = (sg_swapchain){
        .width = BENCH_WIDTH,
        .height = BENCH_HEIGHT,
        .sample_count = 1,
        .color_format = SG_PIXELFORMAT_RGBA8,
        .depth_format = SG_PIXELFORMAT_NONE,
        .gl.framebuffer = 0,
    };
    state->pass = (sg_pass){
        .action = {
            .colors[0] = {
                .load_action = SG_LOADACTION_CLEAR,
                .store_action = SG_STOREACTION_STORE,
                .clear_value = { 0.0f, 0.0f, 0.0f, 1.0f },
            },
            .depth.load_action = SG_LOADACTION_DONTCARE,
            .stencil.load_action = SG_LOADACTION_DONTCARE,
        },
        .swapchain = state->swapchain,
    };
}

static void apply_pipeline(sg_pipeline pip)
{
    frame_pipeline_applies++;
    sg_apply_pipeline(pip);
}

static void apply_bindings(const sg_bindings* bindings)
{
    frame_binding_applies++;
    sg_apply_bindings(bindings);
}

static void draw_frame(const asset_state* state, bench_mode mode)
{
    frame_draw_calls = 0;
    frame_pipeline_applies = 0;
    frame_binding_applies = 0;

    sg_begin_pass(&state->pass);
    if (mode == BENCH_MODE_RAW) {
        for (int i = 0; i < NUM_OBJECTS; i++) {
            const int object = state->raw_order[i];
            apply_pipeline(state->object_pipeline[object]);
            const sg_bindings current = state->bindings[state->object_texture[object]];
            apply_bindings(&current);
            sg_draw(state->object_slot[object] * 6, 6, 1);
            frame_draw_calls++;
        }
    } else if (mode == BENCH_MODE_SORTED) {
        sg_pipeline last_pipeline = { 0 };
        int last_texture = -1;
        for (int i = 0; i < NUM_OBJECTS; i++) {
            const int object = state->sorted_object[i];
            const sg_pipeline pip = state->object_pipeline[object];
            const int texture = state->object_texture[object];
            const bool pipeline_changed = pip.id != last_pipeline.id;
            if (pipeline_changed) {
                apply_pipeline(pip);
                last_pipeline = pip;
            }
            if (pipeline_changed || (texture != last_texture)) {
                apply_bindings(&state->bindings[texture]);
                last_texture = texture;
            }
            sg_draw(i * 6, 6, 1);
            frame_draw_calls++;
        }
    } else {
        sg_pipeline last_pipeline = { 0 };
        int last_texture = -1;
        for (int group = 0; group < NUM_GROUPS; group++) {
            const int first = state->group_start[group];
            const int object = state->sorted_object[first];
            const sg_pipeline pip = state->object_pipeline[object];
            const int texture = state->object_texture[object];
            const bool pipeline_changed = pip.id != last_pipeline.id;
            if (pipeline_changed) {
                apply_pipeline(pip);
                last_pipeline = pip;
            }
            if (pipeline_changed || (texture != last_texture)) {
                apply_bindings(&state->bindings[texture]);
                last_texture = texture;
            }
            sg_draw(first * 6, state->group_count[group] * 6, 1);
            frame_draw_calls++;
        }
    }
    sg_end_pass();
    sg_commit();
    glFinish();
}

static uint64_t pixel_hash(const uint8_t* pixels, size_t num_bytes)
{
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < num_bytes; i++) {
        hash ^= (uint64_t)pixels[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static void read_pixels(uint8_t* pixels)
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glReadBuffer(GL_BACK);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, BENCH_WIDTH, BENCH_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glFinish();
}

static bench_result run_mode(const asset_state* state, bench_mode mode, int num_frames, int warmup_frames)
{
    bench_result result;
    memset(&result, 0, sizeof(result));
    result.num_frames = num_frames;
    result.pixels = (uint8_t*)malloc((size_t)BENCH_WIDTH * (size_t)BENCH_HEIGHT * 4);
    if (!result.pixels) {
        fprintf(stderr, "pixel buffer allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < warmup_frames; i++) {
        draw_frame(state, mode);
    }
    for (int i = 0; i < num_frames; i++) {
        const double t0 = now_ms();
        draw_frame(state, mode);
        const double t1 = now_ms();
        result.frame_ms[i] = t1 - t0;
    }
    result.draw_calls = frame_draw_calls;
    result.pipeline_applies = frame_pipeline_applies;
    result.binding_applies = frame_binding_applies;

    sg_enable_stats();
    draw_frame(state, mode);
    sg_disable_stats();
    result.stats = sg_query_stats().prev_frame;

    read_pixels(result.pixels);
    result.pixel_hash = pixel_hash(result.pixels, (size_t)BENCH_WIDTH * (size_t)BENCH_HEIGHT * 4);
    return result;
}

static int compare_double(const void* lhs, const void* rhs)
{
    const double a = *(const double*)lhs;
    const double b = *(const double*)rhs;
    return (a > b) - (a < b);
}

static double percentile(const double* values, int num_values, double p)
{
    double* sorted = (double*)malloc((size_t)num_values * sizeof(double));
    if (!sorted) {
        return 0.0;
    }
    memcpy(sorted, values, (size_t)num_values * sizeof(double));
    qsort(sorted, (size_t)num_values, sizeof(double), compare_double);
    int index = (int)(p * (double)(num_values - 1) + 0.5);
    if (index < 0) {
        index = 0;
    }
    if (index >= num_values) {
        index = num_values - 1;
    }
    const double value = sorted[index];
    free(sorted);
    return value;
}

static double mean_ms(const bench_result* result)
{
    double sum = 0.0;
    for (int i = 0; i < result->num_frames; i++) {
        sum += result->frame_ms[i];
    }
    return sum / (double)result->num_frames;
}

static void write_csv(const char* path, const bench_result results[BENCH_MODE_NUM], int num_frames)
{
    FILE* file = fopen(path, "w");
    if (!file) {
        fprintf(stderr, "failed to open %s\n", path);
        return;
    }
    fprintf(file, "frame,raw_ms,sorted_ms,merged_ms\n");
    for (int i = 0; i < num_frames; i++) {
        fprintf(file, "%d,%.6f,%.6f,%.6f\n",
                i + 1,
                results[BENCH_MODE_RAW].frame_ms[i],
                results[BENCH_MODE_SORTED].frame_ms[i],
                results[BENCH_MODE_MERGED].frame_ms[i]);
    }
    fclose(file);
}

int main(int argc, char** argv)
{
    int num_frames = DEFAULT_FRAMES;
    int warmup_frames = DEFAULT_WARMUP_FRAMES;
    if (argc > 1) {
        num_frames = atoi(argv[1]);
    }
    if (num_frames <= 0 || num_frames > DEFAULT_FRAMES) {
        fprintf(stderr, "frame count must be 1..%d\n", DEFAULT_FRAMES);
        return 2;
    }
    if (argc > 2) {
        warmup_frames = atoi(argv[2]);
    }
    if (warmup_frames < 0) {
        warmup_frames = 0;
    }
    const char* csv_path = (argc > 3) ? argv[3] : NULL;

    if (!init_egl()) {
        shutdown_egl();
        return 1;
    }

    sg_setup(&(sg_desc){
        .buffer_pool_size = 8,
        .image_pool_size = NUM_TEXTURES + 8,
        .sampler_pool_size = 4,
        .shader_pool_size = NUM_SHADERS + 4,
        .pipeline_pool_size = NUM_PIPELINES + 8,
        .view_pool_size = NUM_TEXTURES + 8,
        .environment.defaults = {
            .color_format = SG_PIXELFORMAT_RGBA8,
            .depth_format = SG_PIXELFORMAT_NONE,
            .sample_count = 1,
        },
        .logger.func = bench_logger,
    });
    if (!sg_isvalid()) {
        fprintf(stderr, "sg_setup failed\n");
        shutdown_egl();
        return 1;
    }

    memset(&assets, 0, sizeof(assets));
    make_scene(&assets);
    const GLubyte* gl_version = glGetString(GL_VERSION);
    const GLubyte* gl_renderer = glGetString(GL_RENDERER);
    printf("sokol-gfx batch benchmark\n");
    printf("renderer=%s\n", gl_renderer ? (const char*)gl_renderer : "unknown");
    printf("version=%s\n", gl_version ? (const char*)gl_version : "unknown");
    printf("objects=%d pipelines=%d shaders=%d textures=%d groups=%d objects_per_group=%d frames=%d warmup=%d\n",
           NUM_OBJECTS, NUM_PIPELINES, NUM_SHADERS, NUM_TEXTURES,
           NUM_GROUPS, OBJECTS_PER_GROUP, num_frames, warmup_frames);
    printf("pipeline_cache_hits=%d misses=%d\n",
           assets.pipeline_cache_hits, assets.pipeline_cache_misses);

    bench_result results[BENCH_MODE_NUM];
    results[BENCH_MODE_RAW] = run_mode(&assets, BENCH_MODE_RAW, num_frames, warmup_frames);
    results[BENCH_MODE_SORTED] = run_mode(&assets, BENCH_MODE_SORTED, num_frames, warmup_frames);
    results[BENCH_MODE_MERGED] = run_mode(&assets, BENCH_MODE_MERGED, num_frames, warmup_frames);

    const uint8_t* raw_pixels = results[BENCH_MODE_RAW].pixels;
    int sorted_mismatch = 0;
    int merged_mismatch = 0;
    const size_t pixel_bytes = (size_t)BENCH_WIDTH * (size_t)BENCH_HEIGHT * 4;
    for (size_t i = 0; i < pixel_bytes; i++) {
        sorted_mismatch += raw_pixels[i] != results[BENCH_MODE_SORTED].pixels[i];
        merged_mismatch += raw_pixels[i] != results[BENCH_MODE_MERGED].pixels[i];
    }

    static const char* mode_names[BENCH_MODE_NUM] = { "raw", "sorted", "merged" };
    printf("\nmode,median_ms,mean_ms,p95_ms,min_ms,max_ms,draws,pipeline_applies,binding_applies,use_program,bind_texture,render_state,hash\n");
    for (int i = 0; i < BENCH_MODE_NUM; i++) {
        const bench_result* r = &results[i];
        printf("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%u,%u,%u,%u,%u,%u,0x%016llx\n",
               mode_names[i],
               percentile(r->frame_ms, r->num_frames, 0.50),
               mean_ms(r),
               percentile(r->frame_ms, r->num_frames, 0.95),
               percentile(r->frame_ms, r->num_frames, 0.0),
               percentile(r->frame_ms, r->num_frames, 1.0),
               (unsigned)r->draw_calls,
               (unsigned)r->pipeline_applies,
               (unsigned)r->binding_applies,
               (unsigned)r->stats.gl.num_use_program,
               (unsigned)r->stats.gl.num_bind_texture,
               (unsigned)r->stats.gl.num_render_state,
               (unsigned long long)r->pixel_hash);
    }

    const double raw_median = percentile(results[BENCH_MODE_RAW].frame_ms, num_frames, 0.50);
    const double sorted_median = percentile(results[BENCH_MODE_SORTED].frame_ms, num_frames, 0.50);
    const double merged_median = percentile(results[BENCH_MODE_MERGED].frame_ms, num_frames, 0.50);
    const double sorted_saving = (raw_median > 0.0) ? (1.0 - sorted_median / raw_median) * 100.0 : 0.0;
    const double merged_saving = (raw_median > 0.0) ? (1.0 - merged_median / raw_median) * 100.0 : 0.0;
    printf("\nsorted_saving=%.2f%% merged_saving=%.2f%%\n", sorted_saving, merged_saving);
    printf("pixel_mismatch_sorted=%d pixel_mismatch_merged=%d\n", sorted_mismatch, merged_mismatch);

    if (num_frames == DEFAULT_FRAMES) {
        printf("\nframe,raw_ms,sorted_ms,merged_ms\n");
        for (int i = 0; i < num_frames; i++) {
            printf("%d,%.6f,%.6f,%.6f\n", i + 1,
                   results[BENCH_MODE_RAW].frame_ms[i],
                   results[BENCH_MODE_SORTED].frame_ms[i],
                   results[BENCH_MODE_MERGED].frame_ms[i]);
        }
    }

    if (csv_path) {
        write_csv(csv_path, results, num_frames);
    }

    const bool pixel_equal = (sorted_mismatch == 0) && (merged_mismatch == 0);
    const bool target_met = (merged_saving >= 30.0);
    printf("\nresult=%s target_30=%s\n", pixel_equal ? "PIXEL_EQUAL" : "PIXEL_MISMATCH",
           target_met ? "PASS" : "FAIL");

    for (int i = 0; i < BENCH_MODE_NUM; i++) {
        free(results[i].pixels);
    }
    sg_shutdown();
    shutdown_egl();
    return (pixel_equal && (error_count == 0)) ? 0 : 1;
}
