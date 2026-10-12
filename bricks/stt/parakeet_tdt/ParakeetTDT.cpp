//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ParakeetTDT.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace compute_flow::bricks::stt {

namespace {

constexpr int32_t k_brick_major_version = 1;
constexpr float k_norm_epsilon = 1.0e-5f;

void append_metaspace_piece(const std::string& piece, std::string& target) {
    for (size_t i = 0; i < piece.size();) {
        const unsigned char c0 = static_cast<unsigned char>(piece[i]);
        if (c0 == 0xE2 && i + 2 < piece.size()) {
            const unsigned char c1 = static_cast<unsigned char>(piece[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(piece[i + 2]);
            if (c1 == 0x96 && c2 == 0x81) {
                target.push_back(' ');
                i += 3;
                continue;
            }
        }
        target.push_back(piece[i]);
        ++i;
    }
}

uint32_t read_u32_le(const std::vector<char>& data, size_t offset) {
    if (offset + 4 > data.size()) {
        throw std::runtime_error("Unexpected end of binary asset");
    }
    return static_cast<uint32_t>(static_cast<unsigned char>(data[offset])) |
           (static_cast<uint32_t>(static_cast<unsigned char>(data[offset + 1])) << 8) |
           (static_cast<uint32_t>(static_cast<unsigned char>(data[offset + 2])) << 16) |
           (static_cast<uint32_t>(static_cast<unsigned char>(data[offset + 3])) << 24);
}

}  // namespace

ValidationResult ParakeetTDT::validate(const ComputeFlowManifest& manifest) const {
    const BrickInfo& info = manifest.get_brick_info();
    COMPUTE_FLOW_VALIDATE(info.type == "stt");
    COMPUTE_FLOW_VALIDATE(info.name == "ParakeetTDT");
    COMPUTE_FLOW_VALIDATE(info.version.major == k_brick_major_version);

    COMPUTE_FLOW_VALIDATE(manifest.num_modules() == 4);
    COMPUTE_FLOW_VALIDATE(manifest.has_module("featurizer"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("encoder"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("decoder_step"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("joint_step"));

    COMPUTE_FLOW_VALIDATE(!manifest["featurizer"].model_path.empty());
    COMPUTE_FLOW_VALIDATE(!manifest["encoder"].model_path.empty());
    COMPUTE_FLOW_VALIDATE(!manifest["decoder_step"].model_path.empty());
    COMPUTE_FLOW_VALIDATE(!manifest["joint_step"].model_path.empty());

    COMPUTE_FLOW_VALIDATE(manifest["featurizer"].framework_type == RunnerType::LiteRT);
    COMPUTE_FLOW_VALIDATE(manifest["encoder"].framework_type == RunnerType::LiteRT);
    COMPUTE_FLOW_VALIDATE(manifest["decoder_step"].framework_type == RunnerType::LiteRT);
    COMPUTE_FLOW_VALIDATE(manifest["joint_step"].framework_type == RunnerType::LiteRT);

    COMPUTE_FLOW_VALIDATE(manifest["featurizer"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["encoder"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["decoder_step"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["joint_step"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["featurizer"].acc_config.has_accelerator(AcceleratorType::Cpu));
    COMPUTE_FLOW_VALIDATE(manifest["encoder"].acc_config.has_accelerator(AcceleratorType::Cpu));
    COMPUTE_FLOW_VALIDATE(manifest["decoder_step"].acc_config.has_accelerator(AcceleratorType::Cpu));
    COMPUTE_FLOW_VALIDATE(manifest["joint_step"].acc_config.has_accelerator(AcceleratorType::Cpu));

    COMPUTE_FLOW_VALIDATE(manifest.has_asset("token_vocab"));

    COMPUTE_FLOW_VALIDATE(manifest.has_constant("sample_rate"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("feature_size"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("num_valid_feature_frames"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("hop_length"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("encoder_hidden_size"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("num_encoder_frames"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("max_symbols_per_step"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("blank_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("vocab_size"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("streaming_chunk_samples"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("streaming_pre_encode_cache_frames"));
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int32_t>("streaming_chunk_samples") > 0);
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int32_t>("streaming_pre_encode_cache_frames") > 0);

    COMPUTE_FLOW_VALIDATE(manifest["featurizer"].has_input("raw_wav"));
    COMPUTE_FLOW_VALIDATE(manifest["featurizer"].has_input("sample_mask"));
    COMPUTE_FLOW_VALIDATE(manifest["featurizer"].has_input("frame_mask"));
    COMPUTE_FLOW_VALIDATE(manifest["featurizer"].has_input("feature_mean"));
    COMPUTE_FLOW_VALIDATE(manifest["featurizer"].has_input("feature_inv_std"));
    COMPUTE_FLOW_VALIDATE(manifest["featurizer"].has_input("pre_encode_feature_cache"));
    COMPUTE_FLOW_VALIDATE(manifest["featurizer"].has_input("previous_sample"));
    COMPUTE_FLOW_VALIDATE(manifest["featurizer"].has_output("input_features"));
    COMPUTE_FLOW_VALIDATE(manifest["featurizer"].has_output("log_mel"));
    COMPUTE_FLOW_VALIDATE(manifest["encoder"].has_input("input_features"));
    COMPUTE_FLOW_VALIDATE(manifest["encoder"].has_input("input_length"));
    COMPUTE_FLOW_VALIDATE(manifest["encoder"].has_output("encoder_pooler_output"));
    COMPUTE_FLOW_VALIDATE(manifest["encoder"].has_output("encoder_output_length"));
    COMPUTE_FLOW_VALIDATE(manifest["decoder_step"].has_input("input_ids"));
    COMPUTE_FLOW_VALIDATE(manifest["decoder_step"].has_output("decoder_hidden"));
    COMPUTE_FLOW_VALIDATE(manifest["decoder_step"].has_state("hidden_state"));
    COMPUTE_FLOW_VALIDATE(manifest["decoder_step"].has_state("cell_state"));
    COMPUTE_FLOW_VALIDATE(!manifest["decoder_step"].has_input("hidden_state"));
    COMPUTE_FLOW_VALIDATE(!manifest["decoder_step"].has_input("cell_state"));
    COMPUTE_FLOW_VALIDATE(!manifest["decoder_step"].has_output("next_hidden_state"));
    COMPUTE_FLOW_VALIDATE(!manifest["decoder_step"].has_output("next_cell_state"));

    return {};
}

void ParakeetTDT::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)manifest;
    (void)profiler;
}

void ParakeetTDT::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    _runner["featurizer"]   = make_runner(manifest["featurizer"].framework_type);
    _runner["encoder"]      = make_runner(manifest["encoder"].framework_type);
    _runner["decoder_step"] = make_runner(manifest["decoder_step"].framework_type);
    _runner["joint_step"]   = make_runner(manifest["joint_step"].framework_type);

    _runner["featurizer"]->setup(manifest["featurizer"], profiler);
    _runner["encoder"]->setup(manifest["encoder"], profiler);
    _runner["decoder_step"]->setup(manifest["decoder_step"], profiler);
    _runner["joint_step"]->setup(manifest["joint_step"], profiler);
}

void ParakeetTDT::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    _sample_rate                       = manifest.get_constant<int32_t>("sample_rate");
    _feature_size                      = manifest.get_constant<int32_t>("feature_size");
    _num_valid_feature_frames          = manifest.get_constant<int32_t>("num_valid_feature_frames");
    _hop_length                        = manifest.get_constant<int32_t>("hop_length");
    _encoder_hidden_size               = manifest.get_constant<int32_t>("encoder_hidden_size");
    _num_encoder_frames                = manifest.get_constant<int32_t>("num_encoder_frames");
    _max_symbols_per_step              = manifest.get_constant<int32_t>("max_symbols_per_step");
    _blank_token_id                    = manifest.get_constant<int64_t>("blank_token_id");
    _vocab_size                        = manifest.get_constant<int64_t>("vocab_size");
    _streaming_chunk_samples           = manifest.get_constant<int32_t>("streaming_chunk_samples");
    _streaming_pre_encode_cache_frames = manifest.get_constant<int32_t>("streaming_pre_encode_cache_frames");

    load_token_vocab(*manifest.get_asset("token_vocab"));

    _feature_mean.assign(static_cast<size_t>(_feature_size), 0.0f);
    _feature_inv_std.assign(static_cast<size_t>(_feature_size), 1.0f);
    _pre_encode_feature_cache.assign(
        static_cast<size_t>(_streaming_pre_encode_cache_frames * _feature_size),
        0.0f);

    TensorFormat raw_wav_chunk_format;
    raw_wav_chunk_format.dt = DataType::F32;
    raw_wav_chunk_format.shape = {1, _streaming_chunk_samples};
    _internal_memory.allocate("raw_wav_chunk", raw_wav_chunk_format);

    TensorFormat sample_mask_format;
    sample_mask_format.dt = DataType::F32;
    sample_mask_format.shape = {1, _streaming_chunk_samples};
    _internal_memory.allocate("sample_mask", sample_mask_format);

    TensorFormat frame_mask_format;
    frame_mask_format.dt = DataType::F32;
    frame_mask_format.shape = {1, _num_valid_feature_frames, 1};
    _internal_memory.allocate("frame_mask", frame_mask_format);

    TensorFormat feature_mean_format;
    feature_mean_format.dt = DataType::F32;
    feature_mean_format.shape = {1, 1, _feature_size};
    _internal_memory.allocate("feature_mean", feature_mean_format);

    TensorFormat feature_inv_std_format;
    feature_inv_std_format.dt = DataType::F32;
    feature_inv_std_format.shape = {1, 1, _feature_size};
    _internal_memory.allocate("feature_inv_std", feature_inv_std_format);

    TensorFormat previous_sample_format;
    previous_sample_format.dt = DataType::F32;
    previous_sample_format.shape = {1};
    _internal_memory.allocate("previous_sample", previous_sample_format);

    TensorFormat input_length_format;
    input_length_format.dt = DataType::I64;
    input_length_format.shape = {1};
    _internal_memory.allocate("input_length", input_length_format);

    TensorFormat input_ids_format;
    input_ids_format.dt = DataType::I64;
    input_ids_format.shape = {1, 1};
    _internal_memory.allocate("input_ids", input_ids_format);

}

void ParakeetTDT::warmup_impl(ProfilerView profiler) {
    TensorDict input;
    input.declare_tensor("raw_wav");
    input.declare_tensor("sample_rate");

    const int64_t sample_rate = _sample_rate;
    std::vector<float> mono_raw_audio(static_cast<size_t>(_sample_rate));

    input["raw_wav"].data = mono_raw_audio.data();
    input["raw_wav"].format.dt = DataType::F32;
    input["raw_wav"].format.shape = {1, static_cast<int64_t>(mono_raw_audio.size())};
    input["sample_rate"].data = const_cast<int64_t*>(&sample_rate);
    input["sample_rate"].format.dt = DataType::I64;
    input["sample_rate"].format.shape = {1};

    run_impl(input, profiler);
}

void ParakeetTDT::reset_states_impl(ProfilerView profiler) {
    (void)profiler;
    _runner["featurizer"]->reset_states();
    _runner["encoder"]->reset_states();
    _runner["decoder_step"]->reset_states();
    _runner["joint_step"]->reset_states();
    _is_output_ready = false;
    _stream_started = false;
    _stream_num_audio_samples = 0;
    _stream_sample_offset = 0;
    _current_num_valid_feature_frames = 0;
    _has_active_encoder_output = false;
    _decode_frame_idx = 0;
    _decode_final_encoder_frame = 0;
    _decode_symbols_on_frame = 0;
    _current_encoder_output = {};
    _current_input_features = {};
    _global_feature_normalization_ready = false;
    _has_emitted_text = false;
    _output_chunk_text.clear();
    _pending_text.clear();
    std::fill(_pre_encode_feature_cache.begin(), _pre_encode_feature_cache.end(), 0.0f);
}

bool ParakeetTDT::is_generation_complete() const {
    return _is_output_ready;
}

TensorDict ParakeetTDT::run_impl(const TensorDict& input, ProfilerView profiler) {
    (void)profiler;

    COMPUTE_FLOW_ASSERT(input["raw_wav"].format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(input["sample_rate"].format.dt == DataType::I64);
    COMPUTE_FLOW_ASSERT(input["raw_wav"].format.shape.size() == 2);
    COMPUTE_FLOW_ASSERT(input["raw_wav"].format.shape[0] == 1);
    COMPUTE_FLOW_ASSERT(reinterpret_cast<const int64_t*>(input["sample_rate"].data)[0] == _sample_rate);

    if (!_stream_started) {
        _stream_num_audio_samples = input["raw_wav"].format.shape[1];
        _stream_sample_offset = 0;
        _has_emitted_text = false;
        _pending_text.clear();
        _current_num_valid_feature_frames = 0;
        _has_active_encoder_output = false;
        _decode_frame_idx = 0;
        _decode_final_encoder_frame = 0;
        _decode_symbols_on_frame = 0;
        _current_encoder_output = {};
        _global_feature_normalization_ready = false;
        std::fill(_pre_encode_feature_cache.begin(), _pre_encode_feature_cache.end(), 0.0f);

        _runner["encoder"]->reset_states();
        _runner["decoder_step"]->reset_states();

        compute_global_feature_normalization(input["raw_wav"]);
        run_decoder_step(_blank_token_id);
        _stream_started = true;
        _is_output_ready = false;
    }

    _output_chunk_text.clear();

    while (!_is_output_ready && _output_chunk_text.empty()) {
        if (!_has_active_encoder_output) {
            const int64_t remaining_samples = _stream_num_audio_samples - _stream_sample_offset;
            if (remaining_samples < _hop_length) {
                _is_output_ready = true;
                break;
            }

            const int64_t num_new_samples =
                std::min<int64_t>(_streaming_chunk_samples, remaining_samples);
            compute_features(input["raw_wav"], _stream_sample_offset, num_new_samples);
            run_encoder();
            start_encoder_decode();
            _stream_sample_offset += num_new_samples;
        }

        if (_has_active_encoder_output && decode_next_text_piece()) {
            break;
        }
    }

    TensorDict output;
    output.declare_tensor("text");
    output["text"].data         = _output_chunk_text.data();
    output["text"].format.dt    = DataType::String;
    output["text"].format.shape = {1, static_cast<int64_t>(_output_chunk_text.size())};
    return output;
}

void ParakeetTDT::load_token_vocab(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open token vocab file: " + path);
    }
    std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    size_t offset = 0;
    const uint32_t count = read_u32_le(data, offset);
    offset += 4;

    _tokens_vocab.clear();
    _tokens_vocab.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t length = read_u32_le(data, offset);
        offset += 4;
        if (offset + length > data.size()) {
            throw std::runtime_error("Token vocab file is truncated");
        }
        _tokens_vocab.emplace_back(data.data() + offset, length);
        offset += length;
    }
    if (offset != data.size()) {
        throw std::runtime_error("Token vocab file has trailing bytes");
    }
}

void ParakeetTDT::compute_global_feature_normalization(const TensorData& audio) {
    std::fill(_feature_mean.begin(), _feature_mean.end(), 0.0f);
    std::fill(_feature_inv_std.begin(), _feature_inv_std.end(), 1.0f);
    std::vector<double> mean(static_cast<size_t>(_feature_size), 0.0);
    std::vector<double> m2(static_cast<size_t>(_feature_size), 0.0);

    int64_t num_frames = 0;
    for (int64_t sample_offset = 0; sample_offset < _stream_num_audio_samples;
         sample_offset += _streaming_chunk_samples) {
        const int64_t remaining_samples = _stream_num_audio_samples - sample_offset;
        if (remaining_samples < _hop_length) {
            break;
        }
        const int64_t num_new_samples =
            std::min<int64_t>(_streaming_chunk_samples, remaining_samples);
        run_featurizer(audio, sample_offset, num_new_samples);

        const TensorData log_mel_tensor = _runner["featurizer"]->get_output("log_mel");
        const float* log_mel = reinterpret_cast<const float*>(log_mel_tensor.data);
        const int64_t valid_feature_frames = std::min<int64_t>(
            _num_valid_feature_frames,
            std::max<int64_t>(0, num_new_samples / _hop_length));
        for (int64_t frame = 0; frame < valid_feature_frames; ++frame) {
            ++num_frames;
            const double count = static_cast<double>(num_frames);
            for (int32_t mel = 0; mel < _feature_size; ++mel) {
                const size_t index = static_cast<size_t>(mel);
                const double value =
                    log_mel[static_cast<size_t>(frame * _feature_size + mel)];
                const double delta = value - mean[index];
                mean[index] += delta / count;
                const double delta2 = value - mean[index];
                m2[index] += delta * delta2;
            }
        }
    }

    if (num_frames == 0) {
        std::fill(_feature_inv_std.begin(), _feature_inv_std.end(), 1.0f);
        _global_feature_normalization_ready = true;
        return;
    }

    const float variance_denominator =
        static_cast<float>(std::max<int64_t>(1, num_frames - 1));
    for (int32_t mel = 0; mel < _feature_size; ++mel) {
        const size_t index = static_cast<size_t>(mel);
        _feature_mean[index] = static_cast<float>(mean[index]);
        const float variance = static_cast<float>(m2[index]) / variance_denominator;
        _feature_inv_std[index] = 1.0f / (std::sqrt(variance) + k_norm_epsilon);
    }
    _global_feature_normalization_ready = true;
}

int64_t ParakeetTDT::prepare_featurizer_inputs(const TensorData& audio,
                                               int64_t start_sample,
                                               int64_t num_chunk_samples) {
    const int64_t valid_feature_frames = std::min<int64_t>(
        _num_valid_feature_frames,
        std::max<int64_t>(0, num_chunk_samples / _hop_length));

    TensorData raw_wav_chunk_tensor = _internal_memory.get_tensor("raw_wav_chunk");
    TensorData sample_mask_tensor = _internal_memory.get_tensor("sample_mask");
    TensorData frame_mask_tensor = _internal_memory.get_tensor("frame_mask");
    TensorData feature_mean_tensor = _internal_memory.get_tensor("feature_mean");
    TensorData feature_inv_std_tensor = _internal_memory.get_tensor("feature_inv_std");
    TensorData previous_sample_tensor = _internal_memory.get_tensor("previous_sample");

    float* raw_wav_chunk = reinterpret_cast<float*>(raw_wav_chunk_tensor.data);
    float* sample_mask = reinterpret_cast<float*>(sample_mask_tensor.data);
    float* frame_mask = reinterpret_cast<float*>(frame_mask_tensor.data);
    float* feature_mean = reinterpret_cast<float*>(feature_mean_tensor.data);
    float* feature_inv_std = reinterpret_cast<float*>(feature_inv_std_tensor.data);
    float* previous_sample = reinterpret_cast<float*>(previous_sample_tensor.data);
    const float* audio_data = reinterpret_cast<const float*>(audio.data);

    std::fill(raw_wav_chunk, raw_wav_chunk + _streaming_chunk_samples, 0.0f);
    std::fill(sample_mask, sample_mask + _streaming_chunk_samples, 0.0f);
    std::fill(frame_mask, frame_mask + _num_valid_feature_frames, 0.0f);

    if (num_chunk_samples > 0) {
        std::memcpy(raw_wav_chunk,
                    audio_data + start_sample,
                    static_cast<size_t>(num_chunk_samples) * sizeof(float));
        std::fill(sample_mask, sample_mask + num_chunk_samples, 1.0f);
    }
    std::fill(frame_mask, frame_mask + valid_feature_frames, 1.0f);
    std::memcpy(feature_mean, _feature_mean.data(), static_cast<size_t>(_feature_size) * sizeof(float));
    std::memcpy(feature_inv_std,
                _feature_inv_std.data(),
                static_cast<size_t>(_feature_size) * sizeof(float));
    previous_sample[0] = start_sample > 0 ? audio_data[start_sample - 1] : 0.0f;

    return valid_feature_frames;
}

void ParakeetTDT::run_featurizer(const TensorData& audio,
                                 int64_t start_sample,
                                 int64_t num_chunk_samples) {
    prepare_featurizer_inputs(audio, start_sample, num_chunk_samples);

    TensorData pre_encode_feature_cache;
    pre_encode_feature_cache.format.dt = DataType::F32;
    pre_encode_feature_cache.format.shape = {1, _streaming_pre_encode_cache_frames, _feature_size};
    pre_encode_feature_cache.data = _pre_encode_feature_cache.data();
    pre_encode_feature_cache.user_hosted = true;

    _runner["featurizer"]->set_input("raw_wav", _internal_memory.get_tensor("raw_wav_chunk"));
    _runner["featurizer"]->set_input("sample_mask", _internal_memory.get_tensor("sample_mask"));
    _runner["featurizer"]->set_input("frame_mask", _internal_memory.get_tensor("frame_mask"));
    _runner["featurizer"]->set_input("feature_mean", _internal_memory.get_tensor("feature_mean"));
    _runner["featurizer"]->set_input("feature_inv_std", _internal_memory.get_tensor("feature_inv_std"));
    _runner["featurizer"]->set_input("pre_encode_feature_cache", pre_encode_feature_cache);
    _runner["featurizer"]->set_input("previous_sample", _internal_memory.get_tensor("previous_sample"));
    _runner["featurizer"]->run();
}

void ParakeetTDT::compute_features(const TensorData& audio,
                                   int64_t start_sample,
                                   int64_t num_chunk_samples) {
    COMPUTE_FLOW_ASSERT(_global_feature_normalization_ready);
    _current_num_valid_feature_frames = prepare_featurizer_inputs(audio, start_sample, num_chunk_samples);
    reinterpret_cast<int64_t*>(_internal_memory.get_tensor("input_length").data)[0] =
        _streaming_pre_encode_cache_frames + _current_num_valid_feature_frames;

    TensorData pre_encode_feature_cache;
    pre_encode_feature_cache.format.dt = DataType::F32;
    pre_encode_feature_cache.format.shape = {1, _streaming_pre_encode_cache_frames, _feature_size};
    pre_encode_feature_cache.data = _pre_encode_feature_cache.data();
    pre_encode_feature_cache.user_hosted = true;

    _runner["featurizer"]->set_input("raw_wav", _internal_memory.get_tensor("raw_wav_chunk"));
    _runner["featurizer"]->set_input("sample_mask", _internal_memory.get_tensor("sample_mask"));
    _runner["featurizer"]->set_input("frame_mask", _internal_memory.get_tensor("frame_mask"));
    _runner["featurizer"]->set_input("feature_mean", _internal_memory.get_tensor("feature_mean"));
    _runner["featurizer"]->set_input("feature_inv_std", _internal_memory.get_tensor("feature_inv_std"));
    _runner["featurizer"]->set_input("pre_encode_feature_cache", pre_encode_feature_cache);
    _runner["featurizer"]->set_input("previous_sample", _internal_memory.get_tensor("previous_sample"));
    _runner["featurizer"]->run();
    _current_input_features = _runner["featurizer"]->get_output("input_features");

    if (_current_num_valid_feature_frames == 0) {
        return;
    }

    const int64_t pre_encode_cache_frames = _streaming_pre_encode_cache_frames;
    const int64_t frames_to_copy =
        std::min<int64_t>(pre_encode_cache_frames, _current_num_valid_feature_frames);
    const int64_t source_frame =
        pre_encode_cache_frames + _current_num_valid_feature_frames - frames_to_copy;
    if (frames_to_copy < _streaming_pre_encode_cache_frames) {
        std::fill(_pre_encode_feature_cache.begin(), _pre_encode_feature_cache.end(), 0.0f);
    }
    const float* features = reinterpret_cast<const float*>(_current_input_features.data);
    std::memcpy(
        _pre_encode_feature_cache.data() +
            static_cast<size_t>((pre_encode_cache_frames - frames_to_copy) * _feature_size),
        features + source_frame * _feature_size,
        static_cast<size_t>(frames_to_copy * _feature_size) * sizeof(float));
}

void ParakeetTDT::run_encoder() {
    _runner["encoder"]->set_input("input_features", _current_input_features);
    _runner["encoder"]->set_input("input_length", _internal_memory.get_tensor("input_length"));
    _runner["encoder"]->run();
}

void ParakeetTDT::start_encoder_decode() {
    _current_encoder_output = _runner["encoder"]->get_output("encoder_pooler_output");
    TensorData encoder_length = _runner["encoder"]->get_output("encoder_output_length");
    COMPUTE_FLOW_ASSERT(encoder_length.format.dt == DataType::I64);
    _decode_final_encoder_frame = std::min<int64_t>(
        std::min<int64_t>(_num_encoder_frames, _current_encoder_output.format.shape[1]),
        reinterpret_cast<const int64_t*>(encoder_length.data)[0]);
    _decode_frame_idx = 0;
    _decode_symbols_on_frame = 0;
    _has_active_encoder_output = _decode_final_encoder_frame > 0;
}

bool ParakeetTDT::decode_next_text_piece() {
    TensorData decoder_hidden = _runner["decoder_step"]->get_output("decoder_hidden");

    while (_decode_frame_idx < _decode_final_encoder_frame) {
        TensorData encoder_frame;
        encoder_frame.format.dt = DataType::F32;
        encoder_frame.format.shape = {1, 1, _encoder_hidden_size};
        encoder_frame.data = reinterpret_cast<float*>(_current_encoder_output.data) +
                             _decode_frame_idx * _encoder_hidden_size;
        encoder_frame.user_hosted = true;

        _runner["joint_step"]->set_input("encoder_frame", encoder_frame);
        _runner["joint_step"]->set_input("decoder_hidden", decoder_hidden);
        _runner["joint_step"]->run();

        TensorData logits_tensor = _runner["joint_step"]->get_output("logits");
        const float* logits = reinterpret_cast<const float*>(logits_tensor.data);
        const int64_t token_id = argmax(logits, _vocab_size);
        const int64_t duration_id = argmax(logits + _vocab_size, 5);
        int64_t duration = duration_id;

        if (token_id == _blank_token_id) {
            if (duration == 0) {
                duration = 1;
            }
            _decode_frame_idx += duration;
            _decode_symbols_on_frame = 0;
            continue;
        }

        const bool emitted_text = append_token_text(token_id);
        ++_decode_symbols_on_frame;

        if (duration > 0) {
            _decode_frame_idx += duration;
            _decode_symbols_on_frame = 0;
        } else if (_decode_symbols_on_frame >= _max_symbols_per_step) {
            ++_decode_frame_idx;
            _decode_symbols_on_frame = 0;
        }

        decoder_hidden = run_decoder_step(token_id);
        if (emitted_text) {
            if (_decode_frame_idx >= _decode_final_encoder_frame) {
                _has_active_encoder_output = false;
                _current_encoder_output = {};
                if (_stream_num_audio_samples - _stream_sample_offset < _hop_length) {
                    _is_output_ready = true;
                }
            }
            return true;
        }
    }
    _has_active_encoder_output = false;
    _current_encoder_output = {};
    return false;
}

TensorData ParakeetTDT::make_scalar_input_id(int64_t token_id) {
    TensorData input_ids = _internal_memory.get_tensor("input_ids");
    reinterpret_cast<int64_t*>(input_ids.data)[0] = token_id;
    return input_ids;
}

TensorData ParakeetTDT::run_decoder_step(int64_t token_id) {
    TensorData input_ids = make_scalar_input_id(token_id);
    _runner["decoder_step"]->set_input("input_ids", input_ids);
    _runner["decoder_step"]->run();

    return _runner["decoder_step"]->get_output("decoder_hidden");
}

bool ParakeetTDT::append_token_text(int64_t token_id) {
    if (token_id < 0 ||
        token_id == _blank_token_id ||
        token_id >= static_cast<int64_t>(_tokens_vocab.size())) {
        return false;
    }

    const std::string& piece = _tokens_vocab[static_cast<size_t>(token_id)];
    if (!piece.empty() && piece.front() == '<' && piece.back() == '>') {
        return false;
    }

    std::string text;
    append_metaspace_piece(piece, text);

    const size_t output_size = _output_chunk_text.size();
    for (char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (_has_emitted_text || !_output_chunk_text.empty() || !_pending_text.empty()) {
                _pending_text.push_back(ch);
            }
            continue;
        }

        if (!_pending_text.empty()) {
            _output_chunk_text += _pending_text;
            _pending_text.clear();
        }
        _output_chunk_text.push_back(ch);
        _has_emitted_text = true;
    }
    return _output_chunk_text.size() > output_size;
}

int64_t ParakeetTDT::argmax(const float* values, int64_t count) const {
    int64_t max_index = 0;
    float max_value = std::numeric_limits<float>::lowest();
    for (int64_t i = 0; i < count; ++i) {
        if (values[i] > max_value) {
            max_value = values[i];
            max_index = i;
        }
    }
    return max_index;
}

}  // namespace compute_flow::bricks::stt
