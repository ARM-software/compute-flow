//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string_view>

namespace compute_flow::timeline_annotations {

/**
 * @brief Logical timeline tracks emitted by the Compute Flow profiler.
 *
 * This enumeration is deliberately backend-neutral. Profiler users select a
 * semantic track, while the implementation translates it into the channel
 * identifiers required by the compiled annotation backend.
 */
enum class Track {
    /** Top-level brick lifecycle operations, such as setup and run. */
    Lifecycle,

    /** Nested brick stages, with a distinct channel for each nesting depth. */
    Stage,

    /** Model invocations, with a distinct channel for each nesting depth. */
    Model,
};

/**
 * @brief Report whether a timeline annotation backend is compiled in.
 *
 * Availability is a compile-time property. It does not indicate whether a
 * capture agent is currently running or accepting annotations.
 *
 * @return `true` when timeline annotations are supported by this build;
 *         otherwise, `false`.
 */
[[nodiscard]] bool is_available() noexcept;

/**
 * @brief Begin a labelled interval on a logical timeline track.
 *
 * Backend initialization is performed once per process. Group and channel
 * metadata are emitted lazily on each thread because annotation channels are
 * thread-local in the backend protocol.
 *
 * The function is a no-op when no backend is compiled in. Backend connection
 * failures, including the absence of a capture agent, do not change program
 * execution and are not surfaced to callers.
 *
 * @param track Logical track on which to emit the interval.
 * @param depth Zero-based nesting depth within the selected track type.
 * @param label Text displayed for the interval. The text is consumed during
 *              this call and does not need to outlive it.
 *
 * @return An opaque, non-zero channel token when an interval was submitted;
 *         otherwise, zero. Pass the returned value to end() to balance the
 *         interval.
 */
[[nodiscard]] uint32_t begin(Track track, uint32_t depth, std::string_view label) noexcept;

/**
 * @brief End an interval previously started by begin().
 *
 * Passing the zero token is explicitly supported and performs no operation.
 * This lets callers unconditionally end scopes when timeline support is
 * disabled or interval creation failed.
 *
 * @param channel Opaque channel token returned by begin().
 */
void end(uint32_t channel) noexcept;

}  // namespace compute_flow::timeline_annotations
