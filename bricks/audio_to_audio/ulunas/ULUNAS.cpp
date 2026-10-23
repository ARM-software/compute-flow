//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ULUNAS.h"

#include <algorithm>
#include <cmath>

namespace compute_flow::bricks::audio_to_audio {

namespace {

constexpr int32_t k_brick_major_version = 1;
constexpr float k_pi = 3.14159265358979323846f;

int64_t reflected_index(int64_t idx, int64_t size) {
    if (size <= 1) {
        return 0;
    }

    while (idx < 0 || idx >= size) {
        if (idx < 0) {
            idx = -idx;
        } else {
            idx = (2 * size) - idx - 2;
        }
    }
    return idx;
}

}  // namespace

ValidationResult ULUNAS::validate(const ComputeFlowManifest& manifest) const {
    const BrickInfo& info = manifest.get_brick_info();
    COMPUTE_FLOW_VALIDATE(info.type == "audio_to_audio");
    COMPUTE_FLOW_VALIDATE(info.name == "ULUNAS");
    COMPUTE_FLOW_VALIDATE(info.version.major == k_brick_major_version);

    COMPUTE_FLOW_VALIDATE(manifest.num_modules() == 1);
    COMPUTE_FLOW_VALIDATE(manifest.has_module("enhancer"));
    COMPUTE_FLOW_VALIDATE(manifest["enhancer"].framework_type == RunnerType::LiteRT);
    COMPUTE_FLOW_VALIDATE(manifest["enhancer"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["enhancer"].acc_config.has_accelerator(AcceleratorType::Cpu));
    COMPUTE_FLOW_VALIDATE(manifest["enhancer"].model_path != "");
    COMPUTE_FLOW_VALIDATE(manifest["enhancer"].has_input("frame"));
    COMPUTE_FLOW_VALIDATE(manifest["enhancer"].has_output("enhanced_frame"));
    COMPUTE_FLOW_VALIDATE(manifest["enhancer"].has_state("conv_cache"));
    COMPUTE_FLOW_VALIDATE(manifest["enhancer"].has_state("tfa_cache"));
    COMPUTE_FLOW_VALIDATE(manifest["enhancer"].has_state("inter_cache"));

    COMPUTE_FLOW_VALIDATE(manifest.has_constant("sample_rate"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("hop_length"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("win_length"));

    return {};
}

void ULUNAS::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    _runner = make_runner(manifest["enhancer"].framework_type);
    _runner->setup(manifest["enhancer"], profiler);
}

void ULUNAS::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    _sample_rate = manifest.get_constant<int64_t>("sample_rate");
    _hop_length = manifest.get_constant<int64_t>("hop_length");
    _win_length = manifest.get_constant<int64_t>("win_length");
    _output_sample_rate = _sample_rate;

    const TensorFormat frame_format = _runner->get_input_format("frame");
    COMPUTE_FLOW_ASSERT(frame_format.dt == DataType::F32);
    if (frame_format.shape.size() == 2) {
        COMPUTE_FLOW_ASSERT(frame_format.shape[0] == 1);
        _model_chunk_size = 1;
        _n_fft = frame_format.shape[1];
    } else {
        COMPUTE_FLOW_ASSERT(frame_format.shape.size() == 3);
        COMPUTE_FLOW_ASSERT(frame_format.shape[0] == 1);
        _model_chunk_size = frame_format.shape[1];
        _n_fft = frame_format.shape[2];
    }
    if (_model_chunk_size <= 0) {
        _model_chunk_size = 1;
    }
    if (_n_fft <= 0) {
        _n_fft = _win_length;
    }

    const TensorFormat enhanced_format = _runner->get_output_format("enhanced_frame");
    COMPUTE_FLOW_ASSERT(enhanced_format.dt == DataType::F32);

    COMPUTE_FLOW_ASSERT(_sample_rate == 16000);
    COMPUTE_FLOW_ASSERT(_hop_length == 256);
    COMPUTE_FLOW_ASSERT(_win_length == _n_fft);

    TensorFormat model_frame_format = frame_format;
    if (_model_chunk_size == 1) {
        model_frame_format.shape = {1, _n_fft};
    } else {
        model_frame_format.shape = {1, _model_chunk_size, _n_fft};
    }
    COMPUTE_FLOW_ASSERT(enhanced_format.has_dynamic_shape() || enhanced_format.shape == model_frame_format.shape);

    _internal_memory.allocate("frame", model_frame_format);
    _window.resize(static_cast<size_t>(_win_length));

    for (int64_t i = 0; i < _win_length; ++i) {
        _window[static_cast<size_t>(i)] =
            0.5f - 0.5f * std::cos((2.0f * k_pi * static_cast<float>(i)) /
                                   static_cast<float>(_win_length));
    }
}

void ULUNAS::warmup_impl(ProfilerView profiler) {
    (void)profiler;
    auto& frame_tensor = _internal_memory.get_tensor("frame");
    float* frame_data = _internal_memory.get_data_as<float>("frame");
    std::fill(frame_data, frame_data + (_model_chunk_size * _n_fft), 0.0f);

    _runner->set_input("frame", frame_tensor);
    _runner->run();
    reset_states_impl(profiler);
}

void ULUNAS::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)manifest;
    (void)profiler;
}

void ULUNAS::reset_states_impl(ProfilerView profiler) {
    (void)profiler;
    if (_runner) {
        _runner->reset_states();
    }
    _is_output_ready = false;
}

bool ULUNAS::is_generation_complete() const {
    return _is_output_ready;
}

TensorDict ULUNAS::run_impl(const TensorDict& input, ProfilerView profiler) {
    (void)profiler;

    COMPUTE_FLOW_ASSERT(input["raw_wav"].format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(input["raw_wav"].data != nullptr);
    COMPUTE_FLOW_ASSERT(input["sample_rate"].format.dt == DataType::I64);
    COMPUTE_FLOW_ASSERT(input["sample_rate"].data != nullptr);
    COMPUTE_FLOW_ASSERT(reinterpret_cast<const int64_t*>(input["sample_rate"].data)[0] == _sample_rate);

    const auto& raw_wav_shape = input["raw_wav"].format.shape;
    COMPUTE_FLOW_ASSERT(raw_wav_shape.size() == 1 || raw_wav_shape.size() == 2);
    if (raw_wav_shape.size() == 2) {
        COMPUTE_FLOW_ASSERT(raw_wav_shape[0] == 1);
    }

    const int64_t num_samples = raw_wav_shape.back();
    COMPUTE_FLOW_ASSERT(num_samples > 0);

    const int64_t num_frames = num_stft_frames(num_samples);
    resize_audio_buffers(num_samples, num_frames);

    const float* raw_audio = reinterpret_cast<const float*>(input["raw_wav"].data);
    auto& frame_tensor = _internal_memory.get_tensor("frame");
    float* frame_data = _internal_memory.get_data_as<float>("frame");

    for (int64_t frame_idx = 0; frame_idx < num_frames; frame_idx += _model_chunk_size) {
        std::fill(frame_data, frame_data + (_model_chunk_size * _n_fft), 0.0f);
        for (int64_t chunk_idx = 0; chunk_idx < _model_chunk_size; ++chunk_idx) {
            const int64_t current_frame_idx = frame_idx + chunk_idx;
            if (current_frame_idx >= num_frames) {
                break;
            }

            prepare_model_frame(frame_data + (chunk_idx * _n_fft),
                                raw_audio,
                                num_samples,
                                current_frame_idx);
        }

        _runner->set_input("frame", frame_tensor);
        _runner->run();

        const TensorData enhanced = _runner->get_output("enhanced_frame");
        COMPUTE_FLOW_ASSERT(enhanced.format.dt == DataType::F32);
        COMPUTE_FLOW_ASSERT(enhanced.data != nullptr);
        const float* enhanced_frame = reinterpret_cast<const float*>(enhanced.data);
        for (int64_t chunk_idx = 0; chunk_idx < _model_chunk_size; ++chunk_idx) {
            const int64_t current_frame_idx = frame_idx + chunk_idx;
            if (current_frame_idx >= num_frames) {
                break;
            }

            append_enhanced_frame(enhanced_frame + (chunk_idx * _n_fft), current_frame_idx);
        }
    }

    finalize_output_audio(num_samples);
    _is_output_ready = true;

    TensorDict output;
    output.declare_tensor("enhanced_wav").format = TensorFormat{DataType::F32, {num_samples}};
    output["enhanced_wav"].data = _output_audio.data();
    output["enhanced_wav"].user_hosted = true;
    output.declare_tensor("sample_rate").format = TensorFormat{DataType::I64, {1}};
    output["sample_rate"].data = &_output_sample_rate;
    output["sample_rate"].user_hosted = true;

    return output;
}

int64_t ULUNAS::num_stft_frames(int64_t num_samples) const {
    return (num_samples / _hop_length) + 1;
}

void ULUNAS::resize_audio_buffers(int64_t num_samples, int64_t num_frames) {
    const int64_t padded_samples = num_samples + _n_fft;
    const int64_t overlap_samples = ((num_frames - 1) * _hop_length) + _n_fft;
    COMPUTE_FLOW_ASSERT(overlap_samples <= padded_samples + _hop_length);

    _overlap_audio.assign(static_cast<size_t>(overlap_samples), 0.0f);
    _window_square_sum.assign(static_cast<size_t>(overlap_samples), 0.0f);
    _output_audio.assign(static_cast<size_t>(num_samples), 0.0f);
}

void ULUNAS::prepare_model_frame(float* frame,
                                 const float* audio,
                                 int64_t num_samples,
                                 int64_t frame_idx) {
    const int64_t center_pad = _n_fft / 2;
    const int64_t frame_start = (frame_idx * _hop_length) - center_pad;

    for (int64_t i = 0; i < _n_fft; ++i) {
        const int64_t audio_idx = reflected_index(frame_start + i, num_samples);
        frame[static_cast<size_t>(i)] = audio[static_cast<size_t>(audio_idx)];
    }
}

void ULUNAS::append_enhanced_frame(const float* enhanced_frame, int64_t frame_idx) {
    const int64_t frame_start = frame_idx * _hop_length;
    for (int64_t i = 0; i < _n_fft; ++i) {
        const int64_t dst_idx = frame_start + i;
        if (dst_idx >= static_cast<int64_t>(_overlap_audio.size())) {
            break;
        }

        const float window = _window[static_cast<size_t>(i)];
        _overlap_audio[static_cast<size_t>(dst_idx)] +=
            enhanced_frame[static_cast<size_t>(i)];
        _window_square_sum[static_cast<size_t>(dst_idx)] += window * window;
    }
}

void ULUNAS::finalize_output_audio(int64_t num_samples) {
    const int64_t center_pad = _n_fft / 2;
    constexpr float k_min_window_sum = 1.0e-8f;

    for (int64_t i = 0; i < num_samples; ++i) {
        const int64_t src_idx = i + center_pad;
        COMPUTE_FLOW_ASSERT(src_idx < static_cast<int64_t>(_overlap_audio.size()));

        const float denom = _window_square_sum[static_cast<size_t>(src_idx)];
        if (denom > k_min_window_sum) {
            _output_audio[static_cast<size_t>(i)] =
                _overlap_audio[static_cast<size_t>(src_idx)] / denom;
        } else {
            _output_audio[static_cast<size_t>(i)] = 0.0f;
        }
    }
}

}  // namespace compute_flow::bricks::audio_to_audio
