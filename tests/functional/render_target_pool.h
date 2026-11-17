//------------------------------------------------------------------------------
//  LLM maintained.
//
//  render_target_pool.h
//
//  Test reference for a cross-frame render-target pool:
//  - reuse is keyed by exact width, height, pixel format and sample count
//  - released targets are reclaimed after 'frames_in_flight' frame starts
//  - invalidated targets are rebuilt on the next acquire after reclamation
//  - every acquire advances the target epoch and requires a clearing pass
//------------------------------------------------------------------------------
#ifndef RENDER_TARGET_POOL_H
#define RENDER_TARGET_POOL_H

#include "sokol_gfx.h"
#include <stdbool.h>
#include <stdint.h>

#define SRTP_INVALID_ID (0)
#define SRTP_DEFAULT_MAX_TARGETS (8)
#define SRTP_DEFAULT_FRAMES_IN_FLIGHT (2)
#define SRTP_MAX_TARGETS (4096)

typedef struct srtp_target {
    uint32_t id;
} srtp_target;

#define SRTP_INVALID_TARGET ((srtp_target){ .id = SRTP_INVALID_ID })

typedef struct srtp_pool_desc {
    int max_targets;            // default: SRTP_DEFAULT_MAX_TARGETS
    int frames_in_flight;       // default: SRTP_DEFAULT_FRAMES_IN_FLIGHT
    sg_pixel_format color_format;   // default: SG_PIXELFORMAT_RGBA8
    int sample_count;           // default: 1
} srtp_pool_desc;

typedef struct srtp_target_desc {
    int width;
    int height;
    sg_pixel_format color_format;   // 0: use the pool default
    int sample_count;               // 0: use the pool default
    const char* label;
} srtp_target_desc;

typedef struct srtp_target_info {
    sg_image image;
    sg_view color_attachment;
    sg_view texture;            // invalid for multisampled targets
    int width;
    int height;
    sg_pixel_format color_format;
    int sample_count;
    uint64_t epoch;             // changes on every acquire
} srtp_target_info;

typedef struct srtp_pool_stats {
    uint64_t acquires;
    uint64_t reuses;
    uint64_t creates;
    uint64_t destroys;
    uint64_t releases;
    uint64_t invalidations;
    uint64_t stale_rejects;
    uint64_t exhausted;
    uint64_t invalid_desc;
    uint64_t create_failures;
    uint64_t deferred_retires;
    uint64_t deferred_purges;
    int live;
    int free_count;
    int retired;
    int peak_live;
} srtp_pool_stats;

typedef struct srtp_pool srtp_pool;

srtp_pool* srtp_create(const srtp_pool_desc* desc);
void srtp_destroy(srtp_pool* pool);

// Call once per frame before acquiring targets.
void srtp_begin_frame(srtp_pool* pool);
// Call once per frame after releasing targets. No resources are destroyed here.
void srtp_end_frame(srtp_pool* pool);

srtp_target srtp_acquire(srtp_pool* pool, const srtp_target_desc* desc);
void srtp_release(srtp_pool* pool, srtp_target target);
void srtp_invalidate(srtp_pool* pool, srtp_target target);
void srtp_invalidate_all(srtp_pool* pool);

bool srtp_query_target(srtp_pool* pool, srtp_target target, srtp_target_info* out_info);
srtp_pool_stats srtp_query_stats(const srtp_pool* pool);

// Always returns a pass with SG_LOADACTION_CLEAR for the target's color
// attachment. Use this instead of retaining content from a recycled target.
sg_pass srtp_clear_pass(const srtp_target_info* info, sg_color clear_color);

#endif // RENDER_TARGET_POOL_H
