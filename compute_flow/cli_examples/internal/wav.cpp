//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/cli_examples/internal/wav.h"

#include "compute_flow/compute_flow.h"

#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace compute_flow::bricks::internal {
namespace {

constexpr uint16_t k_wave_format_pcm = 0x0001;
constexpr uint16_t k_wave_format_ieee_float = 0x0003;
constexpr uint16_t k_wave_format_extensible = 0xFFFE;
constexpr uint16_t k_float_bits_per_sample = 32;
constexpr uint32_t k_format_chunk_size = 16;
constexpr uint32_t k_wav_header_size = 44;

template <typename T>
void read_value(std::ifstream& input_stream, T& value) {
    COMPUTE_FLOW_CHECK(input_stream.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

void write_value(std::ofstream& output_stream, const void* value, std::streamsize num_bytes) {
    COMPUTE_FLOW_CHECK(output_stream.write(reinterpret_cast<const char*>(value), num_bytes));
}

void write_wav_header(std::ofstream& output_stream,
                      uint16_t num_channels,
                      uint32_t sample_rate,
                      uint32_t data_chunk_size) {
    constexpr uint16_t k_audio_format = k_wave_format_ieee_float;
    constexpr uint16_t k_bits_per_sample = k_float_bits_per_sample;

    const uint16_t block_align = num_channels * (k_bits_per_sample / 8);
    const uint32_t byte_rate = sample_rate * block_align;
    const uint32_t file_size = k_wav_header_size + data_chunk_size - 8;

    write_value(output_stream, "RIFF", 4);
    write_value(output_stream, &file_size, sizeof(file_size));
    write_value(output_stream, "WAVE", 4);
    write_value(output_stream, "fmt ", 4);
    write_value(output_stream, &k_format_chunk_size, sizeof(k_format_chunk_size));
    write_value(output_stream, &k_audio_format, sizeof(k_audio_format));
    write_value(output_stream, &num_channels, sizeof(num_channels));
    write_value(output_stream, &sample_rate, sizeof(sample_rate));
    write_value(output_stream, &byte_rate, sizeof(byte_rate));
    write_value(output_stream, &block_align, sizeof(block_align));
    write_value(output_stream, &k_bits_per_sample, sizeof(k_bits_per_sample));
    write_value(output_stream, "data", 4);
    write_value(output_stream, &data_chunk_size, sizeof(data_chunk_size));
}

uint32_t checked_data_chunk_size(int64_t num_samples, uint16_t num_channels) {
    COMPUTE_FLOW_CHECK(num_samples >= 0);

    const uint64_t num_bytes =
        static_cast<uint64_t>(num_samples) * num_channels * static_cast<uint64_t>(sizeof(float));
    COMPUTE_FLOW_CHECK(num_bytes <= std::numeric_limits<uint32_t>::max());
    return static_cast<uint32_t>(num_bytes);
}

} // namespace

int64_t read_mono_wav(const std::string& path, std::vector<float>& audio) {
    std::ifstream input_stream(path, std::ios::binary);
    COMPUTE_FLOW_CHECK(input_stream);

    char riff[4];
    char wave[4];
    char format_chunk_id[4];
    uint32_t riff_size = 0;
    uint32_t format_chunk_size = 0;
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint32_t byte_rate = 0;
    uint16_t block_align = 0;
    uint16_t bits_per_sample = 0;

    const std::streampos riff_base = input_stream.tellg();
    COMPUTE_FLOW_CHECK(input_stream.read(riff, 4));
    read_value(input_stream, riff_size);
    COMPUTE_FLOW_CHECK(input_stream.read(wave, 4));
    COMPUTE_FLOW_CHECK(std::string(riff, 4) == "RIFF" && std::string(wave, 4) == "WAVE");

    COMPUTE_FLOW_CHECK(input_stream.read(format_chunk_id, 4));
    COMPUTE_FLOW_CHECK(std::string(format_chunk_id, 4) == "fmt ");
    read_value(input_stream, format_chunk_size);
    COMPUTE_FLOW_CHECK(format_chunk_size >= k_format_chunk_size);

    read_value(input_stream, audio_format);
    read_value(input_stream, num_channels);
    read_value(input_stream, sample_rate);
    read_value(input_stream, byte_rate);
    read_value(input_stream, block_align);
    read_value(input_stream, bits_per_sample);
    (void)byte_rate;

    const bool is_float32 =
        (audio_format == k_wave_format_ieee_float || audio_format == k_wave_format_extensible) &&
        bits_per_sample == 32;
    const bool is_pcm16 = audio_format == k_wave_format_pcm && bits_per_sample == 16;
    const bool is_pcm32 = audio_format == k_wave_format_pcm && bits_per_sample == 32;
    COMPUTE_FLOW_CHECK(num_channels == 1);
    COMPUTE_FLOW_CHECK(is_float32 || is_pcm16 || is_pcm32);
    COMPUTE_FLOW_CHECK(block_align != 0);

    if (format_chunk_size > k_format_chunk_size) {
        input_stream.seekg(static_cast<std::streamoff>(format_chunk_size - k_format_chunk_size), std::ios::cur);
        COMPUTE_FLOW_CHECK(input_stream);
    }

    const std::streampos riff_end = riff_base + static_cast<std::streamoff>(8ULL + riff_size);
    char chunk_id[4];
    uint32_t data_chunk_size = 0;
    for (;;) {
        uint32_t chunk_size = 0;
        COMPUTE_FLOW_CHECK(input_stream.read(chunk_id, 4));
        read_value(input_stream, chunk_size);

        if (std::string(chunk_id, 4) == "data") {
            data_chunk_size = chunk_size;
            COMPUTE_FLOW_CHECK(input_stream.tellg() + static_cast<std::streamoff>(data_chunk_size) <= riff_end);
            break;
        }

        input_stream.seekg(static_cast<std::streamoff>(chunk_size + (chunk_size & 1U)), std::ios::cur);
        COMPUTE_FLOW_CHECK(input_stream);
    }

    COMPUTE_FLOW_CHECK(data_chunk_size % block_align == 0);
    const uint32_t num_samples = data_chunk_size / block_align;
    audio.resize(num_samples);

    if (is_float32) {
        COMPUTE_FLOW_CHECK(input_stream.read(
            reinterpret_cast<char*>(audio.data()), static_cast<std::streamsize>(data_chunk_size)));
    } else if (is_pcm16) {
        std::vector<int16_t> pcm(num_samples);
        COMPUTE_FLOW_CHECK(input_stream.read(
            reinterpret_cast<char*>(pcm.data()), static_cast<std::streamsize>(data_chunk_size)));
        constexpr float scale = 1.0F / 32768.0F;
        for (uint32_t index = 0; index < num_samples; ++index) {
            audio[index] = static_cast<float>(pcm[index]) * scale;
        }
    } else {
        std::vector<int32_t> pcm(num_samples);
        COMPUTE_FLOW_CHECK(input_stream.read(
            reinterpret_cast<char*>(pcm.data()), static_cast<std::streamsize>(data_chunk_size)));
        constexpr float scale = 1.0F / 2147483648.0F;
        for (uint32_t index = 0; index < num_samples; ++index) {
            audio[index] = static_cast<float>(pcm[index]) * scale;
        }
    }

    return sample_rate;
}

void write_mono_wav(const std::string& path, const float* audio, int64_t num_samples, int64_t sample_rate) {
    constexpr uint16_t num_channels = 1;
    COMPUTE_FLOW_CHECK(sample_rate > 0);
    COMPUTE_FLOW_CHECK(static_cast<uint64_t>(sample_rate) <= std::numeric_limits<uint32_t>::max());

    const uint32_t data_chunk_size = checked_data_chunk_size(num_samples, num_channels);
    std::ofstream output_stream(path, std::ios::binary);
    COMPUTE_FLOW_CHECK(output_stream);

    write_wav_header(output_stream, num_channels, static_cast<uint32_t>(sample_rate), data_chunk_size);
    write_value(output_stream, audio, data_chunk_size);
}

void write_stereo_wav(const std::string& path,
                      const float* left_channel,
                      const float* right_channel,
                      int64_t num_samples,
                      int64_t sample_rate) {
    constexpr uint16_t num_channels = 2;
    COMPUTE_FLOW_CHECK(sample_rate > 0);
    COMPUTE_FLOW_CHECK(static_cast<uint64_t>(sample_rate) <= std::numeric_limits<uint32_t>::max());

    const uint32_t data_chunk_size = checked_data_chunk_size(num_samples, num_channels);
    std::ofstream output_stream(path, std::ios::binary);
    COMPUTE_FLOW_CHECK(output_stream);

    write_wav_header(output_stream, num_channels, static_cast<uint32_t>(sample_rate), data_chunk_size);
    for (int64_t index = 0; index < num_samples; ++index) {
        write_value(output_stream, &left_channel[index], sizeof(float));
        write_value(output_stream, &right_channel[index], sizeof(float));
    }
}

} // namespace compute_flow::bricks::internal
