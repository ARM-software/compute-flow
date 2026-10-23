// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "compute_flow/memory_planner.h"

namespace compute_flow::bricks::stt::nemotron_asr {

/** @brief Frontend geometry in samples/Hz and coefficients supplied by the model manifest. */
struct LogMelConfig {
    int64_t sample_rate;
    int64_t window_samples;
    int64_t hop_samples;
    int64_t fft_size;
    int64_t mel_bins;
    float preemphasis;
    float log_guard;
};

/**
 * @brief Centered, zero-padded log-mel frontend for an immutable float waveform.
 *
 * Computes only the requested frame interval; scratch memory is independent of
 * utterance length. Pre-emphasis reads the preceding original waveform sample,
 * so partitioning the frame interval cannot reset the filter at chunk borders.
 */
class LogMel {
public:
    /**
     * @brief Validate geometry and precompute the symmetric Hann window and Slaney filters.
     * @param[in] config Frontend geometry and coefficients from the model manifest.
     */
    explicit LogMel(const LogMelConfig& config);
    /**
     * @brief Write a frame interval in mel-major order without retaining the audio.
     * @param[in] audio Complete finite waveform, normalized to PCM float scale; nullptr
     * is permitted only when num_frames is zero. Must not alias output.
     * @param[in] num_samples Number of samples in the complete waveform.
     * @param[in] first_frame Absolute frame offset, not relative to a chunk.
     * @param[in] num_frames Frames to compute, within floor(num_samples / hop_samples).
     * @param[out] output Destination's first requested column. Caller provides space for
     * (mel_bins - 1) * row_stride + num_frames floats for a nonempty interval.
     * @param[in] row_stride Distance in floats between mel rows, at least num_frames.
     * Only the requested columns are written. Empty intervals allow nullptr output.
     * @throws std::runtime_error If the interval, stride, or required pointers are invalid.
     * @note Scratch is reused, so calls on one instance must not overlap.
     */
    void compute(const float* audio, int64_t num_samples, int64_t first_frame, int64_t num_frames, float* output,
                 int64_t row_stride);

private:
    LogMelConfig _config;
    MemoryPlanner _memory;
    std::vector<std::complex<float>> _spectrum;
};

}  // namespace compute_flow::bricks::stt::nemotron_asr
