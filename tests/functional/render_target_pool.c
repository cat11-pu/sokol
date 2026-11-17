//------------------------------------------------------------------------------
//  LLM maintained.
//
//  render_target_pool.c
//------------------------------------------------------------------------------
#include "render_target_pool.h"
#include <stdlib.h>
#include <string.h>

typedef enum srtp_entry_state {
    SRTP_ENTRY_FREE,
    SRTP_ENTRY_LIVE,
    SRTP_ENTRY_RETIRED,
} srtp_entry_state;

typedef struct srtp_entry {
    uint32_t id;
    uint16_t generation;
    srtp_entry_state state;
    int width;
    int height;
    sg_pixel_format color_format;
    int sample_count;
    sg_image image;
    sg_view attachment;
    sg_view texture;
    uint64_t epoch;
    uint64_t retire_frame;
    bool invalidated;
} srtp_entry;

struct srtp_pool {
    srtp_pool_desc desc;
    srtp_entry* entries;
    int num_entries;
    uint64_t frame;
    srtp_pool_stats stats;
};

static int _srtp_def(int value, int fallback) {
    return value > 0 ? value : fallback;
}

static sg_pixel_format _srtp_def_format(sg_pixel_format value, sg_pixel_format fallback) {
    return ((int)value == 0) ? fallback : value;
}

static bool _srtp_view_valid(sg_view view) {
    return view.id != SG_INVALID_ID;
}

static bool _srtp_image_valid(sg_image image) {
    return image.id != SG_INVALID_ID;
}

static void _srtp_destroy_resources(srtp_pool* pool, srtp_entry* entry) {
    if (_srtp_view_valid(entry->texture)) {
        sg_destroy_view(entry->texture);
        entry->texture = (sg_view){0};
    }
    if (_srtp_view_valid(entry->attachment)) {
        sg_destroy_view(entry->attachment);
        entry->attachment = (sg_view){0};
    }
    if (_srtp_image_valid(entry->image)) {
        sg_destroy_image(entry->image);
        entry->image = (sg_image){0};
    }
    pool->stats.destroys++;
}

static void _srtp_reset_entry(srtp_entry* entry) {
    const uint16_t generation = entry->generation;
    memset(entry, 0, sizeof(*entry));
    entry->generation = generation;
    entry->state = SRTP_ENTRY_FREE;
}

static bool _srtp_create_resources(srtp_pool* pool, srtp_entry* entry, const srtp_target_desc* desc) {
    entry->image = sg_make_image(&(sg_image_desc){
        .usage.color_attachment = true,
        .width = desc->width,
        .height = desc->height,
        .pixel_format = desc->color_format,
        .sample_count = desc->sample_count,
        .label = desc->label ? desc->label : "srtp-color-image",
    });
    if (!_srtp_image_valid(entry->image)) {
        return false;
    }
    if (sg_query_image_state(entry->image) != SG_RESOURCESTATE_VALID) {
        _srtp_destroy_resources(pool, entry);
        return false;
    }
    entry->attachment = sg_make_view(&(sg_view_desc){
        .color_attachment.image = entry->image,
        .label = desc->label ? desc->label : "srtp-color-attachment",
    });
    if (!_srtp_view_valid(entry->attachment) || (sg_query_view_state(entry->attachment) != SG_RESOURCESTATE_VALID)) {
        _srtp_destroy_resources(pool, entry);
        return false;
    }
    if (desc->sample_count == 1) {
        entry->texture = sg_make_view(&(sg_view_desc){
            .texture.image = entry->image,
            .label = desc->label ? desc->label : "srtp-color-texture",
        });
        if (!_srtp_view_valid(entry->texture) || (sg_query_view_state(entry->texture) != SG_RESOURCESTATE_VALID)) {
            _srtp_destroy_resources(pool, entry);
            return false;
        }
    }
    entry->width = desc->width;
    entry->height = desc->height;
    entry->color_format = desc->color_format;
    entry->sample_count = desc->sample_count;
    pool->stats.creates++;
    return true;
}

static bool _srtp_key_match(const srtp_entry* entry, const srtp_target_desc* desc) {
    return (entry->state == SRTP_ENTRY_FREE)
        && (entry->width == desc->width)
        && (entry->height == desc->height)
        && (entry->color_format == desc->color_format)
        && (entry->sample_count == desc->sample_count);
}

static void _srtp_next_id(srtp_entry* entry, int index) {
    entry->generation++;
    if (entry->generation == 0) {
        entry->generation = 1;
    }
    entry->id = ((uint32_t)entry->generation << 16) | (uint32_t)(index + 1);
}

static srtp_entry* _srtp_lookup(srtp_pool* pool, srtp_target target) {
    if (target.id == SRTP_INVALID_ID) {
        return 0;
    }
    const int index = (int)(target.id & 0xFFFFu) - 1;
    if ((index < 0) || (index >= pool->num_entries)) {
        return 0;
    }
    srtp_entry* entry = &pool->entries[index];
    if (entry->id != target.id) {
        return 0;
    }
    return entry;
}

static void _srtp_purge_expired(srtp_pool* pool) {
    for (int i = 0; i < pool->num_entries; i++) {
        srtp_entry* entry = &pool->entries[i];
        if ((entry->state == SRTP_ENTRY_RETIRED) && (entry->retire_frame <= pool->frame)) {
            if (entry->invalidated) {
                _srtp_destroy_resources(pool, entry);
                _srtp_reset_entry(entry);
            } else {
                entry->state = SRTP_ENTRY_FREE;
                entry->retire_frame = 0;
            }
            pool->stats.deferred_purges++;
        }
    }
}

static void _srtp_update_counts(srtp_pool* pool) {
    int live = 0;
    int free_count = 0;
    int retired = 0;
    for (int i = 0; i < pool->num_entries; i++) {
        switch (pool->entries[i].state) {
            case SRTP_ENTRY_LIVE:
                live++;
                break;
            case SRTP_ENTRY_RETIRED:
                retired++;
                break;
            case SRTP_ENTRY_FREE:
                if (_srtp_image_valid(pool->entries[i].image)) {
                    free_count++;
                }
                break;
        }
    }
    pool->stats.live = live;
    pool->stats.free_count = free_count;
    pool->stats.retired = retired;
    if (live > pool->stats.peak_live) {
        pool->stats.peak_live = live;
    }
}

srtp_pool* srtp_create(const srtp_pool_desc* desc) {
    const srtp_pool_desc defaults = {
        .max_targets = SRTP_DEFAULT_MAX_TARGETS,
        .frames_in_flight = SRTP_DEFAULT_FRAMES_IN_FLIGHT,
        .color_format = SG_PIXELFORMAT_RGBA8,
        .sample_count = 1,
    };
    const srtp_pool_desc* source = desc ? desc : &defaults;
    const int max_targets = _srtp_def(source->max_targets, SRTP_DEFAULT_MAX_TARGETS);
    const int frames_in_flight = _srtp_def(source->frames_in_flight, SRTP_DEFAULT_FRAMES_IN_FLIGHT);
    const int sample_count = _srtp_def(source->sample_count, 1);
    if ((max_targets > SRTP_MAX_TARGETS) || (sample_count < 1)) {
        return 0;
    }
    srtp_pool* pool = (srtp_pool*)calloc(1, sizeof(srtp_pool));
    if (!pool) {
        return 0;
    }
    pool->entries = (srtp_entry*)calloc((size_t)max_targets, sizeof(srtp_entry));
    if (!pool->entries) {
        free(pool);
        return 0;
    }
    pool->num_entries = max_targets;
    pool->desc = *source;
    pool->desc.max_targets = max_targets;
    pool->desc.frames_in_flight = frames_in_flight;
    pool->desc.color_format = _srtp_def_format(source->color_format, SG_PIXELFORMAT_RGBA8);
    pool->desc.sample_count = sample_count;
    return pool;
}

void srtp_destroy(srtp_pool* pool) {
    if (!pool) {
        return;
    }
    for (int i = 0; i < pool->num_entries; i++) {
        srtp_entry* entry = &pool->entries[i];
        if (_srtp_image_valid(entry->image)) {
            _srtp_destroy_resources(pool, entry);
        }
    }
    free(pool->entries);
    free(pool);
}

void srtp_begin_frame(srtp_pool* pool) {
    if (!pool) {
        return;
    }
    pool->frame++;
    _srtp_purge_expired(pool);
    _srtp_update_counts(pool);
}

void srtp_end_frame(srtp_pool* pool) {
    if (!pool) {
        return;
    }
    _srtp_update_counts(pool);
}

srtp_target srtp_acquire(srtp_pool* pool, const srtp_target_desc* desc) {
    if (!pool || !desc) {
        if (pool) {
            pool->stats.invalid_desc++;
        }
        return SRTP_INVALID_TARGET;
    }
    srtp_target_desc resolved = *desc;
    resolved.color_format = _srtp_def_format(desc->color_format, pool->desc.color_format);
    resolved.sample_count = _srtp_def(desc->sample_count, pool->desc.sample_count);
    if ((resolved.width <= 0) || (resolved.height <= 0)
        || (resolved.sample_count < 1)
        || (resolved.color_format == SG_PIXELFORMAT_NONE)
        || ((int)resolved.color_format >= _SG_PIXELFORMAT_NUM)
        || !sg_query_pixelformat(resolved.color_format).render) {
        pool->stats.invalid_desc++;
        return SRTP_INVALID_TARGET;
    }
    for (int i = 0; i < pool->num_entries; i++) {
        srtp_entry* entry = &pool->entries[i];
        if (_srtp_key_match(entry, &resolved)) {
            _srtp_next_id(entry, i);
            entry->state = SRTP_ENTRY_LIVE;
            entry->epoch++;
            entry->invalidated = false;
            entry->retire_frame = 0;
            pool->stats.acquires++;
            pool->stats.reuses++;
            _srtp_update_counts(pool);
            return (srtp_target){ .id = entry->id };
        }
    }
    for (int i = 0; i < pool->num_entries; i++) {
        srtp_entry* entry = &pool->entries[i];
        if ((entry->state != SRTP_ENTRY_FREE) || _srtp_image_valid(entry->image)) {
            continue;
        }
        if (!_srtp_create_resources(pool, entry, &resolved)) {
            pool->stats.create_failures++;
            _srtp_reset_entry(entry);
            _srtp_update_counts(pool);
            return SRTP_INVALID_TARGET;
        }
        _srtp_next_id(entry, i);
        entry->state = SRTP_ENTRY_LIVE;
        entry->epoch = 1;
        entry->retire_frame = 0;
        entry->invalidated = false;
        pool->stats.acquires++;
        _srtp_update_counts(pool);
        return (srtp_target){ .id = entry->id };
    }
    pool->stats.exhausted++;
    return SRTP_INVALID_TARGET;
}

static void _srtp_retire(srtp_pool* pool, srtp_entry* entry, bool invalidated) {
    entry->state = SRTP_ENTRY_RETIRED;
    entry->invalidated = invalidated;
    entry->retire_frame = pool->frame + (uint64_t)pool->desc.frames_in_flight;
    pool->stats.deferred_retires++;
    _srtp_update_counts(pool);
}

void srtp_release(srtp_pool* pool, srtp_target target) {
    if (!pool) {
        return;
    }
    srtp_entry* entry = _srtp_lookup(pool, target);
    if (!entry || (entry->state != SRTP_ENTRY_LIVE)) {
        pool->stats.stale_rejects++;
        return;
    }
    _srtp_retire(pool, entry, false);
    pool->stats.releases++;
}

void srtp_invalidate(srtp_pool* pool, srtp_target target) {
    if (!pool) {
        return;
    }
    srtp_entry* entry = _srtp_lookup(pool, target);
    if (!entry || (entry->state != SRTP_ENTRY_LIVE)) {
        pool->stats.stale_rejects++;
        return;
    }
    _srtp_retire(pool, entry, true);
    pool->stats.invalidations++;
}

void srtp_invalidate_all(srtp_pool* pool) {
    if (!pool) {
        return;
    }
    for (int i = 0; i < pool->num_entries; i++) {
        srtp_entry* entry = &pool->entries[i];
        if (entry->state == SRTP_ENTRY_LIVE) {
            _srtp_retire(pool, entry, true);
            pool->stats.invalidations++;
        }
    }
}

bool srtp_query_target(srtp_pool* pool, srtp_target target, srtp_target_info* out_info) {
    if (!pool) {
        return false;
    }
    srtp_entry* entry = _srtp_lookup(pool, target);
    if (!entry || (entry->state != SRTP_ENTRY_LIVE)) {
        pool->stats.stale_rejects++;
        return false;
    }
    if (out_info) {
        *out_info = (srtp_target_info){
            .image = entry->image,
            .color_attachment = entry->attachment,
            .texture = entry->texture,
            .width = entry->width,
            .height = entry->height,
            .color_format = entry->color_format,
            .sample_count = entry->sample_count,
            .epoch = entry->epoch,
        };
    }
    return true;
}

srtp_pool_stats srtp_query_stats(const srtp_pool* pool) {
    if (!pool) {
        return (srtp_pool_stats){0};
    }
    return pool->stats;
}

sg_pass srtp_clear_pass(const srtp_target_info* info, sg_color clear_color) {
    sg_pass pass = {0};
    pass.action.colors[0] = (sg_color_attachment_action){
        .load_action = SG_LOADACTION_CLEAR,
        .clear_value = clear_color,
        .store_action = SG_STOREACTION_STORE,
    };
    if (info) {
        pass.attachments.colors[0] = info->color_attachment;
    }
    return pass;
}
