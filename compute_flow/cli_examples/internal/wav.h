//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace compute_flow::bricks::internal {

/**
 * @brief Read mono WAV samples and convert supported PCM data to floating point.
 *
 * @param[in] path Path to the input WAV file.
 * @param[out] audio Loaded mono audio samples.
 * @return Sample rate in hertz.
 */
int64_t read_mono_wav(const std::string& path, std::vector<float>& audio);

/**
 * @brief Write mono floating-point samples to a WAV file.
 *
 * @param[in] path Path to the output WAV file.
 * @param[in] audio Mono audio samples.
 * @param[in] num_samples Number of samples in @p audio.
 * @param[in] sample_rate Sample rate in hertz.
 */
void write_mono_wav(const std::string& path, const float* audio, int64_t num_samples, int64_t sample_rate);

/**
 * @brief Write stereo floating-point samples to a WAV file.
 *
 * @param[in] path Path to the output WAV file.
 * @param[in] left_channel Left-channel audio samples.
 * @param[in] right_channel Right-channel audio samples.
 * @param[in] num_samples Number of samples in each channel.
 * @param[in] sample_rate Sample rate in hertz.
 */
void write_stereo_wav(const std::string& path,
                      const float* left_channel,
                      const float* right_channel,
                      int64_t num_samples,
                      int64_t sample_rate);

} // namespace compute_flow::bricks::internal
