//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "timeline_annotations.h"

// Keep every Gator header and implementation detail behind the private build
// definition. Builds without Streamline support compile this translation unit
// into a dependency-free no-op adapter.
#ifdef COMPUTE_FLOW_HAS_STREAMLINE
#include "streamline_annotate.h"

#include <mutex>
#include <string>
#include <unordered_set>
#endif

namespace compute_flow::timeline_annotations {
namespace {

#ifdef COMPUTE_FLOW_HAS_STREAMLINE
/** The single annotation group owned by Compute Flow on each thread. */
constexpr uint32_t k_group = 1;

/** Stable channel reserved for top-level lifecycle intervals. */
constexpr uint32_t k_lifecycle_channel = 1;

/** First channel in the depth-indexed stage channel range. */
constexpr uint32_t k_stage_channel_base = 100;

/** First channel in the depth-indexed model channel range. */
constexpr uint32_t k_model_channel_base = 200;

/** Serializes the one-time initialization required by the Gator C library. */
std::once_flag setup_once;

// Gator groups and channels are thread-local protocol objects. Track the
// metadata emitted by each application thread so names are sent exactly once
// per thread without imposing a process-wide lock on every annotation.
thread_local bool group_named = false;
thread_local std::unordered_set<uint32_t> named_channels;

/**
 * @brief Convert a logical track and nesting depth into a stable channel.
 *
 * Lifecycle intervals intentionally share one channel because top-level
 * profiler scopes do not overlap within a profiler instance. Stage and model
 * intervals use depth-indexed ranges so nested intervals never overwrite a
 * parent interval on the same thread.
 */
uint32_t channel_for(Track track, uint32_t depth) {
    switch (track) {
        case Track::Lifecycle:
            return k_lifecycle_channel;
        case Track::Stage:
            return k_stage_channel_base + depth;
        case Track::Model:
            return k_model_channel_base + depth;
    }
    return 0;
}

/**
 * @brief Produce the persistent display name associated with a channel.
 *
 * Lifecycle uses a fixed name. Stage and model names include a one-based
 * depth for readability. The caller-owned storage keeps generated text alive
 * for the duration of the annotation API call.
 */
const char* channel_name(Track track, uint32_t depth, std::string& storage) {
    switch (track) {
        case Track::Lifecycle:
            return "Lifecycle";
        case Track::Stage:
            storage = "Stage ";
            break;
        case Track::Model:
            storage = "Model ";
            break;
    }
    storage += std::to_string(depth + 1);
    return storage.c_str();
}
#endif

}  // namespace

bool is_available() noexcept {
#ifdef COMPUTE_FLOW_HAS_STREAMLINE
    // The backend being compiled in is sufficient for availability. A capture
    // agent may connect later, so probing the transport here would be both
    // racy and unnecessarily expensive.
    return true;
#else
    return false;
#endif
}

uint32_t begin(Track track, uint32_t depth, std::string_view label) noexcept {
#ifdef COMPUTE_FLOW_HAS_STREAMLINE
    try {
        // Gator owns process-level transport state and requires setup before
        // any annotation call. call_once also makes concurrent first use by
        // independent profiler instances safe.
        std::call_once(setup_once, []() { ANNOTATE_SETUP; });

        // Group and channel declarations belong to the calling thread in the
        // annotation protocol, even when their numeric identifiers match
        // declarations made by another thread.
        if (!group_named) {
            ANNOTATE_NAME_GROUP(k_group, "Compute Flow");
            group_named = true;
        }

        const uint32_t channel = channel_for(track, depth);
        if (named_channels.insert(channel).second) {
            std::string storage;
            ANNOTATE_NAME_CHANNEL(channel, k_group, channel_name(track, depth, storage));
        }

        // string_view does not guarantee null termination. Materialize the
        // label before crossing the C API boundary; Gator consumes it during
        // the call and does not retain this storage.
        const std::string label_string(label);
        // Let Streamline cycle through its predefined palette. Colours improve
        // visual separation but do not encode annotation semantics.
        ANNOTATE_CHANNEL_COLOR(channel, ANNOTATE_COLOR_CYCLE, label_string.c_str());
        return channel;
    } catch (...) {
        // Timeline output is diagnostic only. In particular, allocation
        // failures while preparing metadata must never alter inference or
        // lifecycle behavior.
        return 0;
    }
#else
    // Preserve the same callable interface in builds without Gator while
    // allowing the compiler to diagnose no unused parameters.
    (void)track;
    (void)depth;
    (void)label;
    return 0;
#endif
}

void end(uint32_t channel) noexcept {
#ifdef COMPUTE_FLOW_HAS_STREAMLINE
    // Zero is the adapter's inactive token, returned when begin() is disabled
    // or cannot prepare an annotation.
    if (channel != 0) {
        ANNOTATE_CHANNEL_END(channel);
    }
#else
    // No backend state exists in this build.
    (void)channel;
#endif
}

}  // namespace compute_flow::timeline_annotations
