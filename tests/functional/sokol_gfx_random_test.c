//------------------------------------------------------------------------------
//  LLM maintained.
//
//  sokol_gfx_random_test.c
//
//  Randomized property tests for interleaved resource operations.
//  The test runs 10000 deterministic resource-operation sequences by default.
//  Each sequence is checked for validation errors, crashes (panics), resource
//  leaks, and divergence from a reference model. A failing sequence is
//  delta-debug minimized before it is printed.
//------------------------------------------------------------------------------
#include "sokol_gfx.h"
#include "utest.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define T(b) EXPECT_TRUE(b)

#define TEST_MAX_BUFFERS (6)
#define TEST_MAX_PIPELINES (4)
#define TEST_MAX_OPS (512)
#define TEST_MAX_BUFFER_SIZE (4096)
#define TEST_DEFAULT_CASES (10000)
#define TEST_DEFAULT_STEPS (64)
#define TEST_DEFAULT_SEED (0x5EED1234ULL)

typedef enum {
    OP_CREATE_BUFFER,
    OP_DESTROY_BUFFER,
    OP_WRITE_BUFFER,
    OP_CREATE_PIPELINE,
    OP_DESTROY_PIPELINE,
    OP_APPLY_PIPELINE,
    OP_BEGIN_PASS,
    OP_END_PASS,
    OP_COMMIT,
    OP_DRAW,
    OP_COUNT,
} op_kind_t;

typedef struct {
    uint8_t kind;
    uint8_t a;
    uint8_t b;
    uint8_t reserved;
    int32_t x;
    int32_t y;
    int32_t z;
    uint32_t value;
} test_op_t;

typedef struct {
    test_op_t ops[TEST_MAX_OPS];
    int count;
} test_sequence_t;

typedef enum {
    RUN_OK,
    RUN_INVALID_SCRIPT,
    RUN_LOG_ERROR,
    RUN_PANIC,
    RUN_MODEL_MISMATCH,
    RUN_LEAK,
    RUN_SETUP_FAILED,
    RUN_SEQUENCE_OVERFLOW,
} run_status_t;

typedef struct {
    run_status_t status;
    uint32_t log_item;
    int op_index;
    const char* failure_id;
    char detail[192];
} run_result_t;

typedef struct {
    bool alive;
    sg_buffer handle;
    int size;
    int num_slots;
    int active_slot;
    bool written_this_frame;
    bool bound_this_frame;
    uint32_t write_count;
} model_buffer_t;

typedef struct {
    bool alive;
    sg_pipeline handle;
    int variant;
} model_pipeline_t;

typedef struct {
    model_buffer_t buffers[TEST_MAX_BUFFERS];
    model_pipeline_t pipelines[TEST_MAX_PIPELINES];
    sg_shader shader;
    bool in_pass;
    int applied_pipeline;
} model_t;

typedef struct {
    uint64_t state;
} test_rng_t;

typedef struct {
    int64_t outstanding;
    int64_t total_allocations;
    int64_t total_frees;
    int64_t bytes_outstanding;
} test_allocator_state_t;

typedef struct {
    int errors;
    int warnings;
    int panics;
    uint32_t first_error_item;
    char first_error_message[128];
} test_log_state_t;

typedef struct {
    test_sequence_t* sequence;
    test_rng_t* rng;
    model_t model;
    int frame;
} run_context_t;

typedef enum {
    ALLOC_MODE_NONE,
    ALLOC_MODE_BYTE,
    ALLOC_MODE_HEADER,
} alloc_mode_t;

typedef union {
    size_t size;
    max_align_t alignment;
} test_alloc_header_t;

static alloc_mode_t g_alloc_mode;
static test_allocator_state_t g_allocator;
static test_log_state_t g_log;

static uint64_t g_op_counts[OP_COUNT];
static uint64_t g_total_ops;
static uint64_t g_sequence_count;
static uint64_t g_min_ops = UINT64_MAX;
static uint64_t g_max_ops;
static uint64_t g_draw_ops;
static uint64_t g_write_ops;
static uint64_t g_pipeline_switch_ops;

static void* test_alloc(size_t size, void* user_data) {
    (void)user_data;
    void* ptr;
    if (g_alloc_mode == ALLOC_MODE_HEADER) {
        test_alloc_header_t* header = (test_alloc_header_t*)malloc(sizeof(test_alloc_header_t) + size);
        if (!header) {
            return NULL;
        }
        header->size = size;
        ptr = (void*)(header + 1);
    } else {
        ptr = malloc(size);
        if (!ptr) {
            return NULL;
        }
    }
    g_allocator.outstanding++;
    g_allocator.total_allocations++;
    g_allocator.bytes_outstanding += (int64_t)size;
    return ptr;
}

static void test_free(void* ptr, void* user_data) {
    (void)user_data;
    if (!ptr) {
        return;
    }
    size_t size = 0;
    void* base = ptr;
    if (g_alloc_mode == ALLOC_MODE_HEADER) {
        test_alloc_header_t* header = ((test_alloc_header_t*)ptr) - 1;
        size = header->size;
        base = (void*)header;
    }
    g_allocator.outstanding--;
    g_allocator.total_frees++;
    g_allocator.bytes_outstanding -= (int64_t)size;
    free(base);
}

static void reset_allocator(alloc_mode_t mode) {
    g_alloc_mode = mode;
    memset(&g_allocator, 0, sizeof(g_allocator));
}

static void test_logger(const char* tag, uint32_t log_level, uint32_t log_item_id, const char* message_or_null, uint32_t line_nr, const char* filename_or_null, void* user_data) {
    (void)tag;
    (void)line_nr;
    (void)filename_or_null;
    (void)user_data;
    if (log_level <= 1) {
        if (g_log.errors == 0) {
            g_log.first_error_item = log_item_id;
            snprintf(g_log.first_error_message, sizeof(g_log.first_error_message), "%s", message_or_null ? message_or_null : "");
        }
        g_log.errors++;
    }
    if (log_level == 2) {
        g_log.warnings++;
    }
    if (log_level == 0) {
        g_log.panics++;
    }
}

static void reset_log(void) {
    memset(&g_log, 0, sizeof(g_log));
}

static uint64_t rng_next(test_rng_t* rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * UINT64_C(2685821657736338717);
}

static uint32_t rng_bounded(test_rng_t* rng, uint32_t bound) {
    if (bound == 0) {
        return 0;
    }
    return (uint32_t)(rng_next(rng) % (uint64_t)bound);
}

static void set_failure(run_result_t* result, run_status_t status, const char* failure_id, int op_index, const char* detail) {
    if (result->status != RUN_OK) {
        return;
    }
    result->status = status;
    result->failure_id = failure_id;
    result->op_index = op_index;
    result->log_item = g_log.first_error_item;
    snprintf(result->detail, sizeof(result->detail), "%s", detail ? detail : "");
}

static void set_failuref(run_result_t* result, run_status_t status, const char* failure_id, int op_index, const char* fmt, ...) {
    if (result->status != RUN_OK) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    result->status = status;
    result->failure_id = failure_id;
    result->op_index = op_index;
    result->log_item = g_log.first_error_item;
    vsnprintf(result->detail, sizeof(result->detail), fmt, args);
    va_end(args);
}

static bool check_log(run_result_t* result, int op_index) {
    if (g_log.panics != 0) {
        set_failuref(result, RUN_PANIC, "panic", op_index, "panic item=%" PRIu32 " message=%s", g_log.first_error_item, g_log.first_error_message);
        return false;
    }
    if (g_log.errors != 0) {
        set_failuref(result, RUN_LOG_ERROR, "log_error", op_index, "error item=%" PRIu32 " message=%s", g_log.first_error_item, g_log.first_error_message);
        return false;
    }
    return true;
}

static int find_free_buffer(const model_t* model) {
    for (int i = 0; i < TEST_MAX_BUFFERS; i++) {
        if (!model->buffers[i].alive) {
            return i;
        }
    }
    return -1;
}

static int find_free_pipeline(const model_t* model) {
    for (int i = 0; i < TEST_MAX_PIPELINES; i++) {
        if (!model->pipelines[i].alive) {
            return i;
        }
    }
    return -1;
}

static int count_live_buffers(const model_t* model) {
    int count = 0;
    for (int i = 0; i < TEST_MAX_BUFFERS; i++) {
        if (model->buffers[i].alive) {
            count++;
        }
    }
    return count;
}

static int count_live_pipelines(const model_t* model) {
    int count = 0;
    for (int i = 0; i < TEST_MAX_PIPELINES; i++) {
        if (model->pipelines[i].alive) {
            count++;
        }
    }
    return count;
}

static int random_live_buffer(const model_t* model, test_rng_t* rng) {
    const int count = count_live_buffers(model);
    if (count == 0) {
        return -1;
    }
    int target = (int)rng_bounded(rng, (uint32_t)count);
    for (int i = 0; i < TEST_MAX_BUFFERS; i++) {
        if (model->buffers[i].alive) {
            if (target == 0) {
                return i;
            }
            target--;
        }
    }
    return -1;
}

static int random_live_pipeline(const model_t* model, test_rng_t* rng) {
    const int count = count_live_pipelines(model);
    if (count == 0) {
        return -1;
    }
    int target = (int)rng_bounded(rng, (uint32_t)count);
    for (int i = 0; i < TEST_MAX_PIPELINES; i++) {
        if (model->pipelines[i].alive) {
            if (target == 0) {
                return i;
            }
            target--;
        }
    }
    return -1;
}

static int random_unbound_buffer(const model_t* model, test_rng_t* rng) {
    int count = 0;
    for (int i = 0; i < TEST_MAX_BUFFERS; i++) {
        if (model->buffers[i].alive && !model->buffers[i].bound_this_frame) {
            count++;
        }
    }
    if (count == 0) {
        return -1;
    }
    int target = (int)rng_bounded(rng, (uint32_t)count);
    for (int i = 0; i < TEST_MAX_BUFFERS; i++) {
        if (model->buffers[i].alive && !model->buffers[i].bound_this_frame) {
            if (target == 0) {
                return i;
            }
            target--;
        }
    }
    return -1;
}

static bool emit_op(run_context_t* ctx, test_op_t op, run_result_t* result);

static bool close_pass(run_context_t* ctx, run_result_t* result) {
    if (!ctx->model.in_pass) {
        return true;
    }
    test_op_t op = { .kind = OP_END_PASS };
    return emit_op(ctx, op, result);
}

static bool commit_frame(run_context_t* ctx, run_result_t* result) {
    if (!close_pass(ctx, result)) {
        return false;
    }
    test_op_t op = { .kind = OP_COMMIT };
    return emit_op(ctx, op, result);
}

static bool force_create_buffer(run_context_t* ctx, run_result_t* result) {
    const int slot = find_free_buffer(&ctx->model);
    if (slot < 0) {
        set_failure(result, RUN_INVALID_SCRIPT, "no_free_buffer_slot", -1, "all buffer model slots are live");
        return false;
    }
    if (!close_pass(ctx, result)) {
        return false;
    }
    static const int sizes[] = { 64, 128, 256, 1024, 4096 };
    const test_op_t op = {
        .kind = OP_CREATE_BUFFER,
        .a = (uint8_t)slot,
        .x = sizes[rng_bounded(ctx->rng, (uint32_t)(sizeof(sizes) / sizeof(sizes[0])))],
    };
    return emit_op(ctx, op, result);
}

static bool force_create_pipeline(run_context_t* ctx, run_result_t* result) {
    const int slot = find_free_pipeline(&ctx->model);
    if (slot < 0) {
        set_failure(result, RUN_INVALID_SCRIPT, "no_free_pipeline_slot", -1, "all pipeline model slots are live");
        return false;
    }
    if (!close_pass(ctx, result)) {
        return false;
    }
    const test_op_t op = {
        .kind = OP_CREATE_PIPELINE,
        .a = (uint8_t)slot,
        .b = (uint8_t)rng_bounded(ctx->rng, 3),
    };
    return emit_op(ctx, op, result);
}

static bool ensure_live_buffer(run_context_t* ctx, run_result_t* result) {
    if (count_live_buffers(&ctx->model) == 0) {
        return force_create_buffer(ctx, result);
    }
    return true;
}

static bool ensure_live_pipeline(run_context_t* ctx, run_result_t* result) {
    if (count_live_pipelines(&ctx->model) == 0) {
        return force_create_pipeline(ctx, result);
    }
    return true;
}

static bool ensure_pass(run_context_t* ctx, run_result_t* result) {
    if (ctx->model.in_pass) {
        return true;
    }
    const test_op_t op = { .kind = OP_BEGIN_PASS };
    return emit_op(ctx, op, result);
}

static bool model_write_buffer(run_context_t* ctx, int slot, size_t offset, size_t size, uint32_t pattern, run_result_t* result) {
    model_buffer_t* buf = &ctx->model.buffers[slot];
    if (!buf->alive) {
        set_failuref(result, RUN_INVALID_SCRIPT, "write_dead_buffer", -1, "buffer slot %d is not live", slot);
        return false;
    }
    if (buf->bound_this_frame) {
        set_failuref(result, RUN_INVALID_SCRIPT, "write_bound_buffer", -1, "buffer slot %d is already bound in this frame", slot);
        return false;
    }
    if ((offset + size) > (size_t)buf->size) {
        set_failuref(result, RUN_INVALID_SCRIPT, "write_overflow", -1, "write [%zu,%zu) exceeds buffer size %d", offset, offset + size, buf->size);
        return false;
    }

    uint8_t data[64];
    if (size > sizeof(data)) {
        set_failuref(result, RUN_INVALID_SCRIPT, "write_too_large", -1, "test write size %zu is too large", size);
        return false;
    }
    for (size_t i = 0; i < size; i++) {
        data[i] = (uint8_t)((pattern + (uint32_t)i) & 0xFFu);
    }
    const sg_write_buffer_desc write_desc = {
        .src = {
            .data = { .ptr = data, .size = size },
            .offset = 0,
        },
        .dst = {
            .buffer = buf->handle,
            .offset = offset,
        },
        .size = size,
    };
    sg_write_buffer_transient(&write_desc);
    if (!check_log(result, -1)) {
        return false;
    }
    if (buf->num_slots > 1 && !buf->written_this_frame) {
        buf->active_slot = (buf->active_slot + 1) % buf->num_slots;
    }
    buf->written_this_frame = true;
    buf->write_count++;

    const sg_buffer_info info = sg_query_buffer_info(buf->handle);
    if (info.active_slot != buf->active_slot) {
        set_failuref(result, RUN_MODEL_MISMATCH, "buffer_active_slot", -1, "buffer slot %d active_slot expected=%d actual=%d", slot, buf->active_slot, info.active_slot);
        return false;
    }
    return true;
}

static bool exec_op(run_context_t* ctx, const test_op_t* op, run_result_t* result) {
    switch ((op_kind_t)op->kind) {
        case OP_CREATE_BUFFER: {
            const int slot = op->a;
            if ((slot < 0) || (slot >= TEST_MAX_BUFFERS) || ctx->model.buffers[slot].alive) {
                set_failuref(result, RUN_INVALID_SCRIPT, "create_buffer_slot", result->op_index, "invalid or live buffer slot %d", slot);
                return false;
            }
            const int size = op->x;
            sg_buffer buffer = sg_make_buffer(&(sg_buffer_desc){
                .size = (size_t)size,
                .usage = {
                    .vertex_buffer = true,
                    .write_transient = true,
                },
            });
            if (!check_log(result, result->op_index)) {
                return false;
            }
            if (sg_query_buffer_state(buffer) != SG_RESOURCESTATE_VALID) {
                set_failuref(result, RUN_MODEL_MISMATCH, "buffer_state_create", result->op_index, "buffer slot %d not VALID after create", slot);
                return false;
            }
            if ((int)sg_query_buffer_size(buffer) != size) {
                set_failuref(result, RUN_MODEL_MISMATCH, "buffer_size", result->op_index, "buffer slot %d size expected=%d actual=%zu", slot, size, sg_query_buffer_size(buffer));
                return false;
            }
            const sg_buffer_usage usage = sg_query_buffer_usage(buffer);
            if (!usage.vertex_buffer || !usage.write_transient) {
                set_failuref(result, RUN_MODEL_MISMATCH, "buffer_usage", result->op_index, "buffer slot %d usage mismatch", slot);
                return false;
            }
            const sg_buffer_info info = sg_query_buffer_info(buffer);
            if (info.slot.state != SG_RESOURCESTATE_VALID) {
                set_failuref(result, RUN_MODEL_MISMATCH, "buffer_info_state", result->op_index, "buffer slot %d info state=%d", slot, (int)info.slot.state);
                return false;
            }
            model_buffer_t* model_buf = &ctx->model.buffers[slot];
            memset(model_buf, 0, sizeof(*model_buf));
            model_buf->alive = true;
            model_buf->handle = buffer;
            model_buf->size = size;
            model_buf->num_slots = info.num_slots;
            model_buf->active_slot = info.active_slot;
            return true;
        }
        case OP_DESTROY_BUFFER: {
            const int slot = op->a;
            if ((slot < 0) || (slot >= TEST_MAX_BUFFERS) || !ctx->model.buffers[slot].alive) {
                set_failuref(result, RUN_INVALID_SCRIPT, "destroy_buffer_slot", result->op_index, "invalid or dead buffer slot %d", slot);
                return false;
            }
            model_buffer_t* model_buf = &ctx->model.buffers[slot];
            sg_destroy_buffer(model_buf->handle);
            if (!check_log(result, result->op_index)) {
                return false;
            }
            if (sg_query_buffer_state(model_buf->handle) != SG_RESOURCESTATE_INVALID) {
                set_failuref(result, RUN_MODEL_MISMATCH, "buffer_state_destroy", result->op_index, "buffer slot %d not INVALID after destroy", slot);
                return false;
            }
            model_buf->alive = false;
            model_buf->written_this_frame = false;
            model_buf->bound_this_frame = false;
            return true;
        }
        case OP_WRITE_BUFFER: {
            const int slot = op->a;
            const size_t offset = (size_t)op->x;
            const size_t size = (size_t)op->y;
            if (!model_write_buffer(ctx, slot, offset, size, op->value, result)) {
                return false;
            }
            return check_log(result, result->op_index);
        }
        case OP_CREATE_PIPELINE: {
            const int slot = op->a;
            if ((slot < 0) || (slot >= TEST_MAX_PIPELINES) || ctx->model.pipelines[slot].alive) {
                set_failuref(result, RUN_INVALID_SCRIPT, "create_pipeline_slot", result->op_index, "invalid or live pipeline slot %d", slot);
                return false;
            }
            const int variant = op->b;
            const sg_primitive_type primitive_type = (variant == 1) ? SG_PRIMITIVETYPE_LINES
                                                                  : (variant == 2) ? SG_PRIMITIVETYPE_POINTS
                                                                                   : SG_PRIMITIVETYPE_TRIANGLES;
            sg_pipeline pipeline = sg_make_pipeline(&(sg_pipeline_desc){
                .shader = ctx->model.shader,
                .layout = {
                    .attrs[0].format = SG_VERTEXFORMAT_FLOAT3,
                },
                .primitive_type = primitive_type,
                .cull_mode = (variant == 1) ? SG_CULLMODE_BACK : SG_CULLMODE_NONE,
            });
            if (!check_log(result, result->op_index)) {
                return false;
            }
            if (sg_query_pipeline_state(pipeline) != SG_RESOURCESTATE_VALID) {
                set_failuref(result, RUN_MODEL_MISMATCH, "pipeline_state_create", result->op_index, "pipeline slot %d not VALID after create", slot);
                return false;
            }
            const sg_pipeline_desc pip_desc = sg_query_pipeline_desc(pipeline);
            if (pip_desc.primitive_type != primitive_type) {
                set_failuref(result, RUN_MODEL_MISMATCH, "pipeline_primitive", result->op_index, "pipeline slot %d primitive expected=%d actual=%d", slot, (int)primitive_type, (int)pip_desc.primitive_type);
                return false;
            }
            model_pipeline_t* model_pip = &ctx->model.pipelines[slot];
            model_pip->alive = true;
            model_pip->handle = pipeline;
            model_pip->variant = variant;
            return true;
        }
        case OP_DESTROY_PIPELINE: {
            const int slot = op->a;
            if ((slot < 0) || (slot >= TEST_MAX_PIPELINES) || !ctx->model.pipelines[slot].alive) {
                set_failuref(result, RUN_INVALID_SCRIPT, "destroy_pipeline_slot", result->op_index, "invalid or dead pipeline slot %d", slot);
                return false;
            }
            model_pipeline_t* model_pip = &ctx->model.pipelines[slot];
            sg_destroy_pipeline(model_pip->handle);
            if (!check_log(result, result->op_index)) {
                return false;
            }
            if (sg_query_pipeline_state(model_pip->handle) != SG_RESOURCESTATE_INVALID) {
                set_failuref(result, RUN_MODEL_MISMATCH, "pipeline_state_destroy", result->op_index, "pipeline slot %d not INVALID after destroy", slot);
                return false;
            }
            model_pip->alive = false;
            return true;
        }
        case OP_APPLY_PIPELINE: {
            const int slot = op->a;
            if (!ctx->model.in_pass) {
                set_failuref(result, RUN_INVALID_SCRIPT, "apply_pipeline_outside_pass", result->op_index, "pipeline slot %d applied outside pass", slot);
                return false;
            }
            if ((slot < 0) || (slot >= TEST_MAX_PIPELINES) || !ctx->model.pipelines[slot].alive) {
                set_failuref(result, RUN_INVALID_SCRIPT, "apply_pipeline_slot", result->op_index, "invalid or dead pipeline slot %d", slot);
                return false;
            }
            sg_apply_pipeline(ctx->model.pipelines[slot].handle);
            if (!check_log(result, result->op_index)) {
                return false;
            }
            ctx->model.applied_pipeline = slot;
            return true;
        }
        case OP_BEGIN_PASS: {
            if (ctx->model.in_pass) {
                set_failuref(result, RUN_INVALID_SCRIPT, "begin_pass_twice", result->op_index, "begin pass while already in a pass");
                return false;
            }
            sg_begin_pass(&(sg_pass){
                .swapchain = {
                    .width = 64,
                    .height = 64,
                    .sample_count = 1,
                    .color_format = SG_PIXELFORMAT_RGBA8,
                },
            });
            if (!check_log(result, result->op_index)) {
                return false;
            }
            if (sg_query_pass_state() != SG_PASSSTATE_RENDER) {
                set_failuref(result, RUN_MODEL_MISMATCH, "pass_state_begin", result->op_index, "pass state is not RENDER after begin_pass");
                return false;
            }
            ctx->model.in_pass = true;
            ctx->model.applied_pipeline = -1;
            return true;
        }
        case OP_END_PASS: {
            if (!ctx->model.in_pass) {
                set_failuref(result, RUN_INVALID_SCRIPT, "end_pass_outside", result->op_index, "end pass while not in a pass");
                return false;
            }
            sg_end_pass();
            if (!check_log(result, result->op_index)) {
                return false;
            }
            if (sg_query_pass_state() != SG_PASSSTATE_NONE) {
                set_failuref(result, RUN_MODEL_MISMATCH, "pass_state_end", result->op_index, "pass state is not NONE after end_pass");
                return false;
            }
            ctx->model.in_pass = false;
            ctx->model.applied_pipeline = -1;
            return true;
        }
        case OP_COMMIT: {
            if (ctx->model.in_pass) {
                sg_end_pass();
                if (!check_log(result, result->op_index)) {
                    return false;
                }
                ctx->model.in_pass = false;
                ctx->model.applied_pipeline = -1;
            }
            sg_commit();
            if (!check_log(result, result->op_index)) {
                return false;
            }
            for (int i = 0; i < TEST_MAX_BUFFERS; i++) {
                ctx->model.buffers[i].written_this_frame = false;
                ctx->model.buffers[i].bound_this_frame = false;
            }
            ctx->frame++;
            return true;
        }
        case OP_DRAW: {
            const int buf_slot = op->b;
            const int pip_slot = op->a;
            if (!ctx->model.in_pass) {
                set_failuref(result, RUN_INVALID_SCRIPT, "draw_outside_pass", result->op_index, "draw outside pass");
                return false;
            }
            if ((pip_slot < 0) || (pip_slot >= TEST_MAX_PIPELINES) || !ctx->model.pipelines[pip_slot].alive) {
                set_failuref(result, RUN_INVALID_SCRIPT, "draw_pipeline_slot", result->op_index, "invalid or dead pipeline slot %d", pip_slot);
                return false;
            }
            if ((buf_slot < 0) || (buf_slot >= TEST_MAX_BUFFERS) || !ctx->model.buffers[buf_slot].alive) {
                set_failuref(result, RUN_INVALID_SCRIPT, "draw_buffer_slot", result->op_index, "invalid or dead buffer slot %d", buf_slot);
                return false;
            }
            model_buffer_t* model_buf = &ctx->model.buffers[buf_slot];
            if (!model_buf->written_this_frame) {
                set_failuref(result, RUN_INVALID_SCRIPT, "draw_unwritten_buffer", result->op_index, "buffer slot %d is not written in this frame", buf_slot);
                return false;
            }
            sg_apply_pipeline(ctx->model.pipelines[pip_slot].handle);
            if (!check_log(result, result->op_index)) {
                return false;
            }
            ctx->model.applied_pipeline = pip_slot;
            sg_apply_bindings(&(sg_bindings){
                .vertex_buffers[0] = model_buf->handle,
            });
            if (!check_log(result, result->op_index)) {
                return false;
            }
            sg_draw(op->x, op->y, op->z);
            if (!check_log(result, result->op_index)) {
                return false;
            }
            model_buf->bound_this_frame = true;
            return true;
        }
        case OP_COUNT:
        default:
            set_failuref(result, RUN_INVALID_SCRIPT, "invalid_op_kind", result->op_index, "invalid op kind %d", (int)op->kind);
            return false;
    }
}

static bool emit_op(run_context_t* ctx, test_op_t op, run_result_t* result) {
    if (ctx->sequence->count >= TEST_MAX_OPS) {
        set_failure(result, RUN_SEQUENCE_OVERFLOW, "sequence_overflow", -1, "operation sequence overflow");
        return false;
    }
    result->op_index = ctx->sequence->count;
    ctx->sequence->ops[ctx->sequence->count++] = op;
    return exec_op(ctx, &op, result);
}

static bool setup_run(run_context_t* ctx, test_sequence_t* sequence, run_result_t* result) {
    memset(ctx, 0, sizeof(*ctx));
    memset(result, 0, sizeof(*result));
    result->status = RUN_OK;
    ctx->sequence = sequence;
    ctx->model.applied_pipeline = -1;
    reset_log();
    reset_allocator(ALLOC_MODE_HEADER);

    sg_setup(&(sg_desc){
        .buffer_pool_size = 16,
        .shader_pool_size = 4,
        .pipeline_pool_size = 8,
        .logger = { .func = test_logger },
        .allocator = {
            .alloc_fn = test_alloc,
            .free_fn = test_free,
        },
    });
    if (!sg_isvalid()) {
        set_failure(result, RUN_SETUP_FAILED, "sg_setup", -1, "sg_setup did not initialize sokol-gfx");
        return false;
    }
    if (!check_log(result, -1)) {
        return false;
    }
    ctx->model.shader = sg_make_shader(&(sg_shader_desc){0});
    if (!check_log(result, -1)) {
        return false;
    }
    if (sg_query_shader_state(ctx->model.shader) != SG_RESOURCESTATE_VALID) {
        set_failure(result, RUN_MODEL_MISMATCH, "shader_state", -1, "test shader is not VALID");
        return false;
    }
    return true;
}

static void cleanup_resources(run_context_t* ctx, run_result_t* result) {
    if (sg_isvalid()) {
        if (sg_query_pass_state() != SG_PASSSTATE_NONE) {
            sg_end_pass();
        }
        for (int i = 0; i < TEST_MAX_BUFFERS; i++) {
            if (ctx->model.buffers[i].alive) {
                sg_destroy_buffer(ctx->model.buffers[i].handle);
                ctx->model.buffers[i].alive = false;
            }
        }
        for (int i = 0; i < TEST_MAX_PIPELINES; i++) {
            if (ctx->model.pipelines[i].alive) {
                sg_destroy_pipeline(ctx->model.pipelines[i].handle);
                ctx->model.pipelines[i].alive = false;
            }
        }
        if (ctx->model.shader.id != SG_INVALID_ID) {
            sg_destroy_shader(ctx->model.shader);
            ctx->model.shader.id = SG_INVALID_ID;
        }
        if (result->status == RUN_OK) {
            if (!check_log(result, -1)) {
                return;
            }
            const sg_stats stats = sg_query_stats();
            if ((stats.total.buffers.alive != 0) ||
                (stats.total.pipelines.alive != 0) ||
                (stats.total.shaders.alive != 0) ||
                (stats.total.images.alive != 0) ||
                (stats.total.samplers.alive != 0) ||
                (stats.total.views.alive != 0))
            {
                set_failure(result, RUN_LEAK, "resource_slots_after_cleanup", -1, "resource slots still alive after cleanup");
            }
        }
    }
}

static void shutdown_run(run_context_t* ctx) {
    (void)ctx;
    if (sg_isvalid()) {
        sg_shutdown();
    }
}

static void verify_final_model(run_context_t* ctx, run_result_t* result) {
    if (ctx->model.in_pass) {
        sg_end_pass();
        ctx->model.in_pass = false;
    }
    sg_commit();
    if (!check_log(result, -1)) {
        return;
    }
    if (sg_query_pass_state() != SG_PASSSTATE_NONE) {
        set_failure(result, RUN_MODEL_MISMATCH, "final_pass_state", -1, "pass state is not NONE at end of sequence");
        return;
    }

    int live_buffers = 0;
    int live_pipelines = 0;
    for (int i = 0; i < TEST_MAX_BUFFERS; i++) {
        const model_buffer_t* model_buf = &ctx->model.buffers[i];
        if (model_buf->alive) {
            live_buffers++;
            if (sg_query_buffer_state(model_buf->handle) != SG_RESOURCESTATE_VALID) {
                set_failuref(result, RUN_MODEL_MISMATCH, "final_buffer_state", -1, "buffer slot %d state mismatch", i);
                return;
            }
            if ((int)sg_query_buffer_size(model_buf->handle) != model_buf->size) {
                set_failuref(result, RUN_MODEL_MISMATCH, "final_buffer_size", -1, "buffer slot %d size mismatch", i);
                return;
            }
        } else if (model_buf->handle.id != SG_INVALID_ID) {
            if (sg_query_buffer_state(model_buf->handle) != SG_RESOURCESTATE_INVALID) {
                set_failuref(result, RUN_MODEL_MISMATCH, "final_dead_buffer_state", -1, "destroyed buffer slot %d is not INVALID", i);
                return;
            }
        }
    }
    for (int i = 0; i < TEST_MAX_PIPELINES; i++) {
        const model_pipeline_t* model_pip = &ctx->model.pipelines[i];
        if (model_pip->alive) {
            live_pipelines++;
            if (sg_query_pipeline_state(model_pip->handle) != SG_RESOURCESTATE_VALID) {
                set_failuref(result, RUN_MODEL_MISMATCH, "final_pipeline_state", -1, "pipeline slot %d state mismatch", i);
                return;
            }
        } else if (model_pip->handle.id != SG_INVALID_ID) {
            if (sg_query_pipeline_state(model_pip->handle) != SG_RESOURCESTATE_INVALID) {
                set_failuref(result, RUN_MODEL_MISMATCH, "final_dead_pipeline_state", -1, "destroyed pipeline slot %d is not INVALID", i);
                return;
            }
        }
    }

    const sg_stats stats = sg_query_stats();
    if ((int)stats.total.buffers.alive != live_buffers) {
        set_failuref(result, RUN_MODEL_MISMATCH, "stats_buffers_alive", -1, "buffer alive count expected=%d actual=%" PRIu32, live_buffers, stats.total.buffers.alive);
        return;
    }
    if ((int)stats.total.pipelines.alive != live_pipelines) {
        set_failuref(result, RUN_MODEL_MISMATCH, "stats_pipelines_alive", -1, "pipeline alive count expected=%d actual=%" PRIu32, live_pipelines, stats.total.pipelines.alive);
        return;
    }
    if (stats.total.shaders.alive != 1) {
        set_failuref(result, RUN_MODEL_MISMATCH, "stats_shaders_alive", -1, "shader alive count expected=1 actual=%" PRIu32, stats.total.shaders.alive);
        return;
    }
    if ((stats.total.images.alive != 0) || (stats.total.samplers.alive != 0) || (stats.total.views.alive != 0)) {
        set_failure(result, RUN_MODEL_MISMATCH, "stats_unused_resources", -1, "unexpected live image/sampler/view resources");
        return;
    }
    if ((stats.total.buffers.allocated - stats.total.buffers.deallocated) != stats.total.buffers.alive) {
        set_failure(result, RUN_MODEL_MISMATCH, "stats_buffer_accounting", -1, "buffer allocation accounting mismatch");
        return;
    }
    if ((stats.total.pipelines.allocated - stats.total.pipelines.deallocated) != stats.total.pipelines.alive) {
        set_failure(result, RUN_MODEL_MISMATCH, "stats_pipeline_accounting", -1, "pipeline allocation accounting mismatch");
    }
}

static void finish_run(run_context_t* ctx, run_result_t* result) {
    if (result->status == RUN_OK) {
        verify_final_model(ctx, result);
    }
    cleanup_resources(ctx, result);
    shutdown_run(ctx);

    if (g_log.panics != 0 && result->status == RUN_OK) {
        set_failure(result, RUN_PANIC, "panic_shutdown", -1, "panic during shutdown");
    } else if (g_log.errors != 0 && result->status == RUN_OK) {
        set_failure(result, RUN_LOG_ERROR, "log_error_shutdown", -1, "validation error during shutdown");
    }
    if (result->status == RUN_OK) {
        if (g_allocator.outstanding != 0) {
            set_failuref(result, RUN_LEAK, "allocator_outstanding", -1, "allocator outstanding=%" PRId64, g_allocator.outstanding);
        } else if (g_allocator.bytes_outstanding != 0) {
            set_failuref(result, RUN_LEAK, "allocator_bytes", -1, "allocator bytes_outstanding=%" PRId64, g_allocator.bytes_outstanding);
        }
    }
}

static void run_sequence(const test_sequence_t* sequence, run_result_t* result) {
    run_context_t ctx;
    test_sequence_t scratch = *sequence;
    if (!setup_run(&ctx, &scratch, result)) {
        shutdown_run(&ctx);
        finish_run(&ctx, result);
        return;
    }
    for (int i = 0; i < scratch.count; i++) {
        result->op_index = i;
        if (!exec_op(&ctx, &scratch.ops[i], result)) {
            break;
        }
    }
    finish_run(&ctx, result);
}

static bool choose_random_op(run_context_t* ctx, run_result_t* result) {
    const uint32_t roll = rng_bounded(ctx->rng, 100);
    if (roll < 15) {
        if (find_free_buffer(&ctx->model) >= 0) {
            return force_create_buffer(ctx, result);
        }
        return true;
    }
    if (roll < 25) {
        if (count_live_buffers(&ctx->model) > 0) {
            const int slot = random_live_buffer(&ctx->model, ctx->rng);
            if (!commit_frame(ctx, result)) {
                return false;
            }
            const test_op_t op = { .kind = OP_DESTROY_BUFFER, .a = (uint8_t)slot };
            return emit_op(ctx, op, result);
        }
        return force_create_buffer(ctx, result);
    }
    if (roll < 48) {
        if (!ensure_live_buffer(ctx, result)) {
            return false;
        }
        int slot = random_unbound_buffer(&ctx->model, ctx->rng);
        if (slot < 0) {
            if (!commit_frame(ctx, result)) {
                return false;
            }
            slot = random_live_buffer(&ctx->model, ctx->rng);
        }
        if (slot < 0) {
            set_failure(result, RUN_INVALID_SCRIPT, "write_no_buffer", -1, "no live buffer for write");
            return false;
        }
        const int size = ctx->model.buffers[slot].size;
        const int len = 4 * (1 + (int)rng_bounded(ctx->rng, 16));
        const int max_offset = (size - len) / 4;
        const test_op_t op = {
            .kind = OP_WRITE_BUFFER,
            .a = (uint8_t)slot,
            .x = 4 * (int)rng_bounded(ctx->rng, (uint32_t)(max_offset + 1)),
            .y = len,
            .value = (uint32_t)rng_next(ctx->rng),
        };
        return emit_op(ctx, op, result);
    }
    if (roll < 55) {
        if (find_free_pipeline(&ctx->model) >= 0) {
            return force_create_pipeline(ctx, result);
        }
        return true;
    }
    if (roll < 62) {
        if (count_live_pipelines(&ctx->model) > 0) {
            const int slot = random_live_pipeline(&ctx->model, ctx->rng);
            if (!commit_frame(ctx, result)) {
                return false;
            }
            const test_op_t op = { .kind = OP_DESTROY_PIPELINE, .a = (uint8_t)slot };
            return emit_op(ctx, op, result);
        }
        return force_create_pipeline(ctx, result);
    }
    if (roll < 74) {
        if (!ensure_live_pipeline(ctx, result) || !ensure_pass(ctx, result)) {
            return false;
        }
        const int slot = random_live_pipeline(&ctx->model, ctx->rng);
        const test_op_t op = { .kind = OP_APPLY_PIPELINE, .a = (uint8_t)slot };
        return emit_op(ctx, op, result);
    }
    if (roll < 94) {
        if (!ensure_live_pipeline(ctx, result) || !ensure_live_buffer(ctx, result)) {
            return false;
        }
        int buf_slot = random_live_buffer(&ctx->model, ctx->rng);
        bool has_unwritten_unbound = false;
        for (int i = 0; i < TEST_MAX_BUFFERS; i++) {
            if (ctx->model.buffers[i].alive && !ctx->model.buffers[i].written_this_frame && !ctx->model.buffers[i].bound_this_frame) {
                has_unwritten_unbound = true;
            }
        }
        if (ctx->model.buffers[buf_slot].bound_this_frame) {
            if (!ensure_pass(ctx, result)) {
                return false;
            }
        } else if (!ctx->model.buffers[buf_slot].written_this_frame) {
            if (!has_unwritten_unbound) {
                if (!commit_frame(ctx, result)) {
                    return false;
                }
                buf_slot = random_live_buffer(&ctx->model, ctx->rng);
            }
            const int size = ctx->model.buffers[buf_slot].size;
            const int len = 4 * (1 + (int)rng_bounded(ctx->rng, 16));
            const int max_offset = (size - len) / 4;
            const test_op_t write_op = {
                .kind = OP_WRITE_BUFFER,
                .a = (uint8_t)buf_slot,
                .x = 4 * (int)rng_bounded(ctx->rng, (uint32_t)(max_offset + 1)),
                .y = len,
                .value = (uint32_t)rng_next(ctx->rng),
            };
            if (!emit_op(ctx, write_op, result)) {
                return false;
            }
        }
        if (!ensure_pass(ctx, result)) {
            return false;
        }
        const int pip_slot = random_live_pipeline(&ctx->model, ctx->rng);
        const test_op_t draw_op = {
            .kind = OP_DRAW,
            .a = (uint8_t)pip_slot,
            .b = (uint8_t)buf_slot,
            .x = (int32_t)rng_bounded(ctx->rng, 4),
            .y = (int32_t)rng_bounded(ctx->rng, 7),
            .z = (int32_t)rng_bounded(ctx->rng, 3),
        };
        return emit_op(ctx, draw_op, result);
    }
    if (roll < 97) {
        return ensure_pass(ctx, result);
    }
    if (roll < 99) {
        return close_pass(ctx, result);
    }
    return commit_frame(ctx, result);
}

static void generate_sequence(uint32_t case_index, uint64_t base_seed, int steps, test_sequence_t* sequence, run_result_t* result) {
    memset(sequence, 0, sizeof(*sequence));
    run_context_t ctx;
    if (!setup_run(&ctx, sequence, result)) {
        shutdown_run(&ctx);
        finish_run(&ctx, result);
        return;
    }
    test_rng_t rng = { .state = base_seed + (uint64_t)case_index * UINT64_C(0x9E3779B97F4A7C15) };
    ctx.rng = &rng;

    if (!force_create_buffer(&ctx, result) || !force_create_pipeline(&ctx, result)) {
        finish_run(&ctx, result);
        return;
    }
    for (int i = 2; i < steps; i++) {
        if (!choose_random_op(&ctx, result)) {
            break;
        }
    }
    finish_run(&ctx, result);
}

static const char* status_name(run_status_t status) {
    switch (status) {
        case RUN_OK: return "ok";
        case RUN_INVALID_SCRIPT: return "invalid_script";
        case RUN_LOG_ERROR: return "log_error";
        case RUN_PANIC: return "panic";
        case RUN_MODEL_MISMATCH: return "model_mismatch";
        case RUN_LEAK: return "leak";
        case RUN_SETUP_FAILED: return "setup_failed";
        case RUN_SEQUENCE_OVERFLOW: return "sequence_overflow";
        default: return "unknown";
    }
}

static void print_sequence(const char* title, const test_sequence_t* sequence, uint32_t case_index, uint64_t seed) {
    printf("%s case=%" PRIu32 " seed=%" PRIu64 " ops=%d\n", title, case_index, seed, sequence->count);
    for (int i = 0; i < sequence->count; i++) {
        const test_op_t* op = &sequence->ops[i];
        switch ((op_kind_t)op->kind) {
            case OP_CREATE_BUFFER:
                printf("  %02d create_buffer slot=%u size=%d\n", i, (unsigned)op->a, op->x);
                break;
            case OP_DESTROY_BUFFER:
                printf("  %02d destroy_buffer slot=%u\n", i, (unsigned)op->a);
                break;
            case OP_WRITE_BUFFER:
                printf("  %02d write_buffer slot=%u offset=%d size=%d pattern=0x%08" PRIx32 "\n", i, (unsigned)op->a, op->x, op->y, op->value);
                break;
            case OP_CREATE_PIPELINE:
                printf("  %02d create_pipeline slot=%u variant=%u\n", i, (unsigned)op->a, (unsigned)op->b);
                break;
            case OP_DESTROY_PIPELINE:
                printf("  %02d destroy_pipeline slot=%u\n", i, (unsigned)op->a);
                break;
            case OP_APPLY_PIPELINE:
                printf("  %02d apply_pipeline slot=%u\n", i, (unsigned)op->a);
                break;
            case OP_BEGIN_PASS:
                printf("  %02d begin_pass\n", i);
                break;
            case OP_END_PASS:
                printf("  %02d end_pass\n", i);
                break;
            case OP_COMMIT:
                printf("  %02d commit\n", i);
                break;
            case OP_DRAW:
                printf("  %02d draw pipeline=%u buffer=%u base=%d count=%d instances=%d\n", i, (unsigned)op->a, (unsigned)op->b, op->x, op->y, op->z);
                break;
            default:
                printf("  %02d invalid\n", i);
                break;
        }
    }
}

static bool same_failure(const run_result_t* a, const run_result_t* b) {
    if (a->status != b->status) {
        return false;
    }
    const char* a_id = a->failure_id ? a->failure_id : "";
    const char* b_id = b->failure_id ? b->failure_id : "";
    return strcmp(a_id, b_id) == 0;
}

static bool replay_reproduces(const test_sequence_t* sequence, const run_result_t* expected) {
    run_result_t actual;
    run_sequence(sequence, &actual);
    return same_failure(expected, &actual);
}

static void minimize_sequence(const test_sequence_t* input, const run_result_t* failure, test_sequence_t* output) {
    *output = *input;
    if (output->count == 0) {
        return;
    }
    int granularity = 2;
    while (output->count > 1) {
        const int chunk_size = (output->count + granularity - 1) / granularity;
        bool reduced = false;
        for (int start = 0; start < output->count; start += chunk_size) {
            int end = start + chunk_size;
            if (end > output->count) {
                end = output->count;
            }
            test_sequence_t candidate;
            memset(&candidate, 0, sizeof(candidate));
            candidate.count = 0;
            for (int i = 0; i < output->count; i++) {
                if ((i < start) || (i >= end)) {
                    candidate.ops[candidate.count++] = output->ops[i];
                }
            }
            if (replay_reproduces(&candidate, failure)) {
                *output = candidate;
                reduced = true;
                if (granularity > 2) {
                    granularity--;
                }
                break;
            }
        }
        if (!reduced) {
            if (granularity >= output->count) {
                break;
            }
            granularity *= 2;
            if (granularity > output->count) {
                granularity = output->count;
            }
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < output->count; i++) {
            test_sequence_t candidate;
            memset(&candidate, 0, sizeof(candidate));
            candidate.count = 0;
            for (int j = 0; j < output->count; j++) {
                if (j != i) {
                    candidate.ops[candidate.count++] = output->ops[j];
                }
            }
            if (replay_reproduces(&candidate, failure)) {
                *output = candidate;
                changed = true;
                break;
            }
        }
    }

    test_sequence_t empty;
    memset(&empty, 0, sizeof(empty));
    if ((output->count > 0) && replay_reproduces(&empty, failure)) {
        *output = empty;
    }
}

static int env_int(const char* name, int fallback) {
    const char* value = getenv(name);
    if (!value || !value[0]) {
        return fallback;
    }
    const long parsed = strtol(value, NULL, 10);
    if (parsed <= 0) {
        return fallback;
    }
    return (int)parsed;
}

static uint64_t env_u64(const char* name, uint64_t fallback) {
    const char* value = getenv(name);
    if (!value || !value[0]) {
        return fallback;
    }
    char* end = NULL;
    const unsigned long long parsed = strtoull(value, &end, 0);
    if (!end || (end == value)) {
        return fallback;
    }
    return (uint64_t)parsed;
}

static void record_sequence_stats(const test_sequence_t* sequence) {
    g_sequence_count++;
    g_total_ops += (uint64_t)sequence->count;
    if ((uint64_t)sequence->count < g_min_ops) {
        g_min_ops = (uint64_t)sequence->count;
    }
    if ((uint64_t)sequence->count > g_max_ops) {
        g_max_ops = (uint64_t)sequence->count;
    }
    for (int i = 0; i < sequence->count; i++) {
        const uint8_t kind = sequence->ops[i].kind;
        if (kind < OP_COUNT) {
            g_op_counts[kind]++;
        }
        if (kind == OP_DRAW) {
            g_draw_ops++;
        } else if (kind == OP_WRITE_BUFFER) {
            g_write_ops++;
        } else if (kind == OP_APPLY_PIPELINE) {
            g_pipeline_switch_ops++;
        }
    }
}

UTEST(sokol_gfx_random, model_sequences) {
    const int cases = env_int("SG_RANDOM_CASES", TEST_DEFAULT_CASES);
    const int steps = env_int("SG_RANDOM_STEPS", TEST_DEFAULT_STEPS);
    const uint64_t base_seed = env_u64("SG_RANDOM_SEED", TEST_DEFAULT_SEED);

    memset(g_op_counts, 0, sizeof(g_op_counts));
    g_total_ops = 0;
    g_sequence_count = 0;
    g_min_ops = UINT64_MAX;
    g_max_ops = 0;
    g_draw_ops = 0;
    g_write_ops = 0;
    g_pipeline_switch_ops = 0;

    int failed_case = -1;
    test_sequence_t failed_sequence;
    run_result_t failed_result;
    memset(&failed_sequence, 0, sizeof(failed_sequence));
    memset(&failed_result, 0, sizeof(failed_result));

    for (int case_index = 0; case_index < cases; case_index++) {
        test_sequence_t sequence;
        run_result_t result;
        generate_sequence((uint32_t)case_index, base_seed, steps, &sequence, &result);
        record_sequence_stats(&sequence);
        if (result.status != RUN_OK) {
            failed_case = case_index;
            failed_sequence = sequence;
            failed_result = result;
            break;
        }
    }

    if (failed_case >= 0) {
        const uint64_t seed = base_seed + (uint64_t)failed_case * UINT64_C(0x9E3779B97F4A7C15);
        printf("\nPROPERTY FAILURE case=%d status=%s id=%s op=%d item=%" PRIu32 " detail=%s\n",
               failed_case, status_name(failed_result.status), failed_result.failure_id ? failed_result.failure_id : "",
               failed_result.op_index, failed_result.log_item, failed_result.detail);
        print_sequence("FAILING_SEQUENCE", &failed_sequence, (uint32_t)failed_case, seed);

        test_sequence_t minimized;
        minimize_sequence(&failed_sequence, &failed_result, &minimized);
        print_sequence("MINIMIZED_SEQUENCE", &minimized, (uint32_t)failed_case, seed);

        run_result_t replay;
        run_sequence(&minimized, &replay);
        printf("MINIMIZED_REPLAY status=%s id=%s detail=%s\n",
               status_name(replay.status), replay.failure_id ? replay.failure_id : "", replay.detail);
    }

    const uint64_t average_ops = g_sequence_count ? (g_total_ops / g_sequence_count) : 0;
    const uint64_t minimum_ops = (g_min_ops == UINT64_MAX) ? 0 : g_min_ops;
    printf("\nrandom resource property summary:\n");
    printf("  sequences=%" PRIu64 " requested=%d steps=%d seed=%" PRIu64 "\n", g_sequence_count, cases, steps, base_seed);
    printf("  operations=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64 " avg=%" PRIu64 "\n", g_total_ops, minimum_ops, g_max_ops, average_ops);
    printf("  create_buffer=%" PRIu64 " destroy_buffer=%" PRIu64 " write_buffer=%" PRIu64 "\n", g_op_counts[OP_CREATE_BUFFER], g_op_counts[OP_DESTROY_BUFFER], g_write_ops);
    printf("  create_pipeline=%" PRIu64 " destroy_pipeline=%" PRIu64 " apply_pipeline=%" PRIu64 "\n", g_op_counts[OP_CREATE_PIPELINE], g_op_counts[OP_DESTROY_PIPELINE], g_pipeline_switch_ops);
    printf("  begin_pass=%" PRIu64 " end_pass=%" PRIu64 " commit=%" PRIu64 " draw=%" PRIu64 "\n", g_op_counts[OP_BEGIN_PASS], g_op_counts[OP_END_PASS], g_op_counts[OP_COMMIT], g_draw_ops);
    const int leak_failure = (failed_case >= 0) && (failed_result.status == RUN_LEAK);
    const int panic_failure = (failed_case >= 0) && (failed_result.status == RUN_PANIC);
    printf("  failures=%d failure_status=%s leaks=%d panics=%d\n",
           failed_case >= 0 ? 1 : 0,
           failed_case >= 0 ? status_name(failed_result.status) : "none",
           leak_failure,
           panic_failure);

    T(failed_case < 0);
}
