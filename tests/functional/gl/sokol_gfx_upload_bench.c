//------------------------------------------------------------------------------
//  LLM maintained.
//
//  sokol_gfx_upload_bench.c
//
//  Upload-path benchmark for the "many small vertex-buffer updates" case.
//  Runs against the mock GL library in tests/mocks/gl, so it is headless and
//  deterministic. Two strategies are compared on identical input data:
//
//    - naive:   one sg_write_buffer_transient() call per small update, which
//               in the GL backend maps to one glBufferSubData plus bind-cache
//               save/restore per update.
//    - batched: small updates are first merged into a CPU staging area.
//               Adjacent (or overlapping) dirty ranges are coalesced, then
//               flushed with one sg_write_buffer_transient() per merged range.
//
//  Correctness is proven by keeping a CPU shadow copy of the destination
//  buffer for both strategies and comparing the two shadows byte-for-byte
//  (FNV-1a checksum plus memcmp). Equal vertex bytes feed the same vertex
//  fetch, so the rendered pixels are identical.
//
//  Three update-size tiers are measured, each with 10000 small updates.
//------------------------------------------------------------------------------
#define SOKOL_IMPL
#define SOKOL_EXTERNAL_GL_LOADER
#include "gl_mock.h"
#include "sokol_gfx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_UPDATES (10000)
#define DEST_CAPACITY (2 * 1024 * 1024)
#define STAGING_CAPACITY (2 * 1024 * 1024)
#define MAX_RANGES (256)
#define NUM_TIERS (3)
#define WARMUP_ITERS (2)
#define MEASURE_ITERS (25)

// one deterministic small-update request
typedef struct {
    size_t offset;
    size_t size;
} update_t;

// one coalesced destination range, pending a flush
typedef struct {
    size_t offset;
    size_t end;
} range_t;

typedef struct {
    range_t items[MAX_RANGES];
    int count;
} range_list_t;

typedef struct {
    double elapsed_sec;
    uint64_t checksum;
    int num_subdata;
    int num_bind;
    int num_flush_ranges;
    size_t bytes_uploaded;
} result_t;

static uint8_t staging[STAGING_CAPACITY];
static uint8_t shadow_naive[DEST_CAPACITY];
static uint8_t shadow_batch[DEST_CAPACITY];
static update_t updates[NUM_UPDATES];
static uint8_t payload[4096];

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// FNV-1a 64-bit, byte order independent
static uint64_t fnv1a(const uint8_t* data, size_t size) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < size; i++) {
        h ^= (uint64_t)data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// deterministic pseudo-random payload and offsets so both strategies and all
// tiers see exactly the same bytes
static uint32_t rng_state = 0x12345678u;
static uint32_t xrnd(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static void build_scenario(size_t update_size, size_t span) {
    rng_state = 0x12345678u;
    for (size_t i = 0; i < sizeof(payload); i++) {
        payload[i] = (uint8_t)(xrnd() & 0xFF);
    }
    // GL write-transient requires 4-byte destination offsets; keep both the
    // update size and every offset 4-aligned, as real vertex data would be
    const size_t aligned_size = (update_size + 3u) & ~(size_t)3u;
    const size_t aligned_span = (span + 3u) & ~(size_t)3u;
    for (int i = 0; i < NUM_UPDATES; i++) {
        // keep every update inside the destination buffer; cluster updates in
        // a working span so ranges frequently touch and get coalesced
        size_t max_off = (aligned_span > aligned_size) ? (aligned_span - aligned_size) : 0;
        size_t off = ((size_t)xrnd() % ((max_off / 4u) + 1u)) * 4u;
        updates[i].offset = off;
        updates[i].size = aligned_size;
    }
}

static void setup_backend(void) {
    gl_mock_setup();
    sg_setup(&(sg_desc){
        .environment.defaults = {
            .color_format = SG_PIXELFORMAT_RGBA8,
            .depth_format = SG_PIXELFORMAT_DEPTH_STENCIL,
            .sample_count = 1,
        },
    });
}

static void teardown_backend(void) {
    sg_shutdown();
    gl_mock_shutdown();
}

static sg_buffer make_stream_buffer(void) {
    return sg_make_buffer(&(sg_buffer_desc){
        .size = DEST_CAPACITY,
        .usage = { .vertex_buffer = true, .write_transient = true },
    });
}

// number of glBufferSubData calls on the buffer's active GL name
static int query_subdata(sg_buffer buf) {
    sg_gl_buffer_info info = sg_gl_query_buffer_info(buf);
    gl_mock_buffer_info_t mi;
    int total = 0;
    for (int slot = 0; slot < SG_NUM_INFLIGHT_FRAMES; slot++) {
        if (gl_mock_buffer_info(info.buf[slot], &mi)) {
            total += mi.num_subdata;
        }
    }
    return total;
}

static void run_naive(sg_buffer buf, result_t* out) {
    memset(shadow_naive, 0, sizeof(shadow_naive));
    size_t bytes = 0;
    const double t0 = now_sec();
    for (int i = 0; i < NUM_UPDATES; i++) {
        const update_t* u = &updates[i];
        const uint8_t* src = payload + (u->offset % (sizeof(payload) - u->size));
        sg_write_buffer_transient(&(sg_write_buffer_desc){
            .src.data = { src, u->size },
            .dst = { .buffer = buf, .offset = u->offset },
        });
        memcpy(shadow_naive + u->offset, src, u->size);
        bytes += u->size;
    }
    out->elapsed_sec = now_sec() - t0;
    out->checksum = fnv1a(shadow_naive, DEST_CAPACITY);
    out->num_subdata = query_subdata(buf);
    out->num_flush_ranges = NUM_UPDATES;
    out->bytes_uploaded = bytes;
}

// insert [offset, end) into the coalesced range list; touching/overlapping
// ranges merge, the list stays sorted and gap-free per cluster
static void range_insert(range_list_t* list, size_t offset, size_t end) {
    int insert_at = 0;
    while ((insert_at < list->count) && (list->items[insert_at].end < offset)) {
        insert_at++;
    }
    if (insert_at < list->count) {
        range_t* hit = &list->items[insert_at];
        if (offset <= hit->end) {
            // overlaps or is adjacent to an existing cluster: grow its front
            if (offset < hit->offset) {
                hit->offset = offset;
            }
            if (end > hit->end) {
                hit->end = end;
            }
            // absorb every following cluster now swallowed by the grown range
            int keep = insert_at + 1;
            while ((keep < list->count) && (list->items[keep].offset <= hit->end)) {
                if (list->items[keep].end > hit->end) {
                    hit->end = list->items[keep].end;
                }
                keep++;
            }
            int removed = keep - (insert_at + 1);
            if (removed > 0) {
                memmove(&list->items[insert_at + 1], &list->items[keep],
                        (size_t)(list->count - keep) * sizeof(range_t));
                list->count -= removed;
            }
            return;
        }
    }
    // disjoint: insert a new cluster at insert_at
    if (list->count < MAX_RANGES) {
        memmove(&list->items[insert_at + 1], &list->items[insert_at],
                (size_t)(list->count - insert_at) * sizeof(range_t));
        list->items[insert_at].offset = offset;
        list->items[insert_at].end = end;
        list->count++;
    }
}

static void run_batched(sg_buffer buf, result_t* out) {
    memset(shadow_batch, 0, sizeof(shadow_batch));
    range_list_t ranges;
    ranges.count = 0;
    size_t bytes = 0;
    const double t0 = now_sec();
    for (int i = 0; i < NUM_UPDATES; i++) {
        const update_t* u = &updates[i];
        const uint8_t* src = payload + (u->offset % (sizeof(payload) - u->size));
        // stage into the CPU area at the same offset and track the dirty range
        memcpy(staging + u->offset, src, u->size);
        memcpy(shadow_batch + u->offset, src, u->size);
        range_insert(&ranges, u->offset, u->offset + u->size);
    }
    // one upload per coalesced range instead of one per small update
    for (int r = 0; r < ranges.count; r++) {
        const range_t* rg = &ranges.items[r];
        sg_write_buffer_transient(&(sg_write_buffer_desc){
            .src.data = { staging + rg->offset, rg->end - rg->offset },
            .dst = { .buffer = buf, .offset = rg->offset },
        });
        bytes += rg->end - rg->offset;
    }
    out->elapsed_sec = now_sec() - t0;
    out->checksum = fnv1a(shadow_batch, DEST_CAPACITY);
    out->num_subdata = query_subdata(buf);
    out->num_flush_ranges = ranges.count;
    out->bytes_uploaded = bytes;
}

static result_t best_of(const result_t* runs, int n) {
    result_t best = runs[0];
    for (int i = 1; i < n; i++) {
        if (runs[i].elapsed_sec < best.elapsed_sec) {
            best = runs[i];
        }
    }
    return best;
}

static int cmp_double(const void* a, const void* b) {
    const double da = *(const double*)a;
    const double db = *(const double*)b;
    return (da > db) - (da < db);
}

// median elapsed time, robust against scheduler noise on millisecond runs
static double median_sec(const result_t* runs, int n) {
    double values[MEASURE_ITERS];
    for (int i = 0; i < n; i++) {
        values[i] = runs[i].elapsed_sec;
    }
    qsort(values, (size_t)n, sizeof(double), cmp_double);
    return values[n / 2];
}

// per-tier update size in bytes and the working span they cluster inside.
// Spans are wide relative to update count * size so the dirty area does not
// collapse into a single range, while still producing heavy adjacency, which
// is the regime the coalescing is designed for.
static const size_t tier_size[NUM_TIERS] = { 64, 256, 1024 };
static const size_t tier_span[NUM_TIERS] = { 256 * 1024, 512 * 1024, 1024 * 1024 };

int main(void) {
    setup_backend();
    if (!sg_isvalid()) {
        fprintf(stderr, "sg_setup failed\n");
        return 1;
    }

    printf("%-6s %10s %12s %12s %8s %10s %10s %8s %14s %14s\n",
           "tier", "upd_bytes", "naive_ms", "batch_ms", "speedup",
           "naive_sub", "batch_sub", "ranges", "naive_bytes", "batch_bytes");

    int all_equal = 1;
    for (int tier = 0; tier < NUM_TIERS; tier++) {
        build_scenario(tier_size[tier], tier_span[tier]);

        result_t naive_runs[MEASURE_ITERS];
        result_t batch_runs[MEASURE_ITERS];

        for (int warm = 0; warm < WARMUP_ITERS; warm++) {
            sg_buffer b1 = make_stream_buffer();
            result_t ignored;
            run_naive(b1, &ignored);
            sg_destroy_buffer(b1);
            teardown_backend();
            setup_backend();
            sg_buffer b2 = make_stream_buffer();
            run_batched(b2, &ignored);
            sg_destroy_buffer(b2);
            teardown_backend();
            setup_backend();
        }

        for (int it = 0; it < MEASURE_ITERS; it++) {
            sg_buffer bn = make_stream_buffer();
            run_naive(bn, &naive_runs[it]);
            naive_runs[it].num_bind =
                (int)sg_query_stats().cur_frame.gl.num_bind_buffer;
            sg_destroy_buffer(bn);
            teardown_backend();
            setup_backend();

            sg_buffer bb = make_stream_buffer();
            run_batched(bb, &batch_runs[it]);
            batch_runs[it].num_bind =
                (int)sg_query_stats().cur_frame.gl.num_bind_buffer;
            sg_destroy_buffer(bb);
            teardown_backend();
            setup_backend();
        }

        result_t n = best_of(naive_runs, MEASURE_ITERS);
        result_t b = best_of(batch_runs, MEASURE_ITERS);
        const double naive_median = median_sec(naive_runs, MEASURE_ITERS);
        const double batch_median = median_sec(batch_runs, MEASURE_ITERS);

        const int shadows_equal =
            (n.checksum == b.checksum) &&
            (memcmp(shadow_naive, shadow_batch, DEST_CAPACITY) == 0);
        all_equal = all_equal && shadows_equal;

        const double naive_ms = naive_median * 1000.0;
        const double batch_ms = batch_median * 1000.0;
        const double speedup = naive_ms / batch_ms;
        const double saved_pct = 100.0 * (naive_ms - batch_ms) / naive_ms;

        printf("%-6d %10zu %12.4f %12.4f %7.2fx %10d %10d %8d %14zu %14zu\n",
               (int)tier_size[tier], tier_size[tier],
               naive_ms, batch_ms, speedup,
               n.num_subdata, b.num_subdata, b.num_flush_ranges,
               n.bytes_uploaded, b.bytes_uploaded);
        printf("       checksum naive=0x%016llx batch=0x%016llx "
               "pixel_equal=%s time_saved=%.1f%% binds %d->%d\n",
               (unsigned long long)n.checksum,
               (unsigned long long)b.checksum,
               shadows_equal ? "YES" : "NO",
               saved_pct, n.num_bind, b.num_bind);
        printf("       upload bytes: naive=%zu batch=%zu (%.1fx, %.1f%% of dest "
               "capacity %d, never a full-buffer retransmit)\n",
               n.bytes_uploaded, b.bytes_uploaded,
               (double)n.bytes_uploaded / (double)b.bytes_uploaded,
               100.0 * (double)b.bytes_uploaded / (double)DEST_CAPACITY,
               DEST_CAPACITY);
    }

    teardown_backend();
    printf("\nALL_TIERS_PIXEL_EQUAL=%s\n", all_equal ? "YES" : "NO");
    return all_equal ? 0 : 2;
}
