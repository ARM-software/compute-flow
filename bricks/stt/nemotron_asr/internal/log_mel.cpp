// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

#include "log_mel.h"

#include <algorithm>
#include <cmath>

#include "compute_flow/check.h"

namespace compute_flow::bricks::stt::nemotron_asr {
namespace {

constexpr float k_pi             = 3.14159265358979323846f;
constexpr float k_frequency_step = 200.0f / 3.0f;
constexpr float k_min_log_hertz  = 1000.0f;
constexpr float k_min_log_mel    = k_min_log_hertz / k_frequency_step;
constexpr float k_log_step       = 0.06875177742094912f;

float hertz_to_mel(float frequency) {
    if (frequency < k_min_log_hertz) {
        return frequency / k_frequency_step;
    }

    return k_min_log_mel + std::log(frequency / k_min_log_hertz) / k_log_step;
}

float mel_to_hertz(float mel) {
    if (mel < k_min_log_mel) {
        return mel * k_frequency_step;
    }

    return k_min_log_hertz * std::exp(k_log_step * (mel - k_min_log_mel));
}

void fft(std::vector<std::complex<float>>& values) {
    const int64_t count = static_cast<int64_t>(values.size());

    for (int64_t index = 1, reversed = 0; index < count; ++index) {
        int64_t bit = count >> 1;

        while ((reversed & bit) != 0) {
            reversed ^= bit;
            bit >>= 1;
        }

        reversed ^= bit;

        if (index < reversed) {
            std::swap(values[index], values[reversed]);
        }
    }

    for (int64_t length = 2; length <= count; length <<= 1) {
        const float angle = -2.0f * k_pi / static_cast<float>(length);
        const std::complex<float> root(std::cos(angle), std::sin(angle));

        for (int64_t start = 0; start < count; start += length) {
            std::complex<float> factor(1.0f, 0.0f);

            for (int64_t offset = 0; offset < length / 2; ++offset) {
                const auto even                     = values[start + offset];
                const auto odd                      = values[start + offset + length / 2] * factor;
                values[start + offset]              = even + odd;
                values[start + offset + length / 2] = even - odd;
                factor *= root;
            }
        }
    }
}

}  // namespace

LogMel::LogMel(const LogMelConfig& config) : _config(config) {
    COMPUTE_FLOW_CHECK(config.sample_rate > 0 && config.hop_samples > 0);
    COMPUTE_FLOW_CHECK(config.fft_size > 0 && (config.fft_size & (config.fft_size - 1)) == 0);
    COMPUTE_FLOW_CHECK(config.window_samples > 1 && config.window_samples <= config.fft_size);
    COMPUTE_FLOW_CHECK(config.mel_bins > 0 && config.log_guard > 0.0f);

    const int64_t num_frequencies = config.fft_size / 2 + 1;

    _memory.allocate("window", {DataType::F32, {config.window_samples}});
    _memory.allocate("filters", {DataType::F32, {config.mel_bins, num_frequencies}});
    _spectrum.resize(static_cast<size_t>(config.fft_size));

    auto* window        = _memory.get_data_as<float>("window");
    auto* filters       = _memory.get_data_as<float>("filters");
    const float max_mel = hertz_to_mel(static_cast<float>(config.sample_rate) / 2.0f);
    std::vector<float> points(static_cast<size_t>(config.mel_bins + 2));

    for (int64_t index = 0; index < config.window_samples; ++index) {
        window[index] = 0.5f - 0.5f * std::cos(2.0f * k_pi * static_cast<float>(index) /
                                               static_cast<float>(config.window_samples - 1));
    }

    for (int64_t index = 0; index < config.mel_bins + 2; ++index) {
        const float fraction = static_cast<float>(index) / static_cast<float>(config.mel_bins + 1);
        points[index]        = mel_to_hertz(fraction * max_mel);
    }

    for (int64_t mel = 0; mel < config.mel_bins; ++mel) {
        const float left          = points[mel];
        const float center        = points[mel + 1];
        const float right         = points[mel + 2];
        const float normalization = 2.0f / (right - left);

        for (int64_t bin = 0; bin < num_frequencies; ++bin) {
            const float frequency = static_cast<float>(bin * config.sample_rate) / static_cast<float>(config.fft_size);
            const float lower     = (frequency - left) / (center - left);
            const float upper     = (right - frequency) / (right - center);
            filters[mel * num_frequencies + bin] = std::max(0.0f, std::min(lower, upper)) * normalization;
        }
    }
}

void LogMel::compute(const float* audio, int64_t num_samples, int64_t first_frame, int64_t num_frames, float* output,
                     int64_t row_stride) {
    COMPUTE_FLOW_CHECK(num_samples >= 0 && first_frame >= 0 && num_frames >= 0);
    COMPUTE_FLOW_CHECK(num_frames <= row_stride);
    COMPUTE_FLOW_CHECK(first_frame <= num_samples / _config.hop_samples);
    COMPUTE_FLOW_CHECK(num_frames <= num_samples / _config.hop_samples - first_frame);
    COMPUTE_FLOW_CHECK(num_frames == 0 || (audio != nullptr && output != nullptr));

    const auto* window            = _memory.get_data_as<float>("window");
    const auto* filters           = _memory.get_data_as<float>("filters");
    const int64_t num_frequencies = _config.fft_size / 2 + 1;

    for (int64_t frame = 0; frame < num_frames; ++frame) {
        std::fill(_spectrum.begin(), _spectrum.end(), std::complex<float>{});

        const int64_t center = (first_frame + frame) * _config.hop_samples;

        for (int64_t index = 0; index < _config.window_samples; ++index) {
            const int64_t sample_index = center + index - _config.window_samples / 2;
            float sample               = 0.0f;

            if (sample_index >= 0 && sample_index < num_samples) {
                sample = audio[sample_index];

                if (sample_index > 0) {
                    sample -= _config.preemphasis * audio[sample_index - 1];
                }
            }

            const int64_t fft_index = index + (_config.fft_size - _config.window_samples) / 2;
            _spectrum[fft_index]    = sample * window[index];
        }

        fft(_spectrum);

        for (int64_t mel = 0; mel < _config.mel_bins; ++mel) {
            float energy = 0.0f;

            for (int64_t bin = 0; bin < num_frequencies; ++bin) {
                energy += std::norm(_spectrum[bin]) * filters[mel * num_frequencies + bin];
            }

            output[mel * row_stride + frame] = std::log(energy + _config.log_guard);
        }
    }
}

}  // namespace compute_flow::bricks::stt::nemotron_asr
