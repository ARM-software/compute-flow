//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "Perch.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace compute_flow::bricks::audio_detection {

namespace {

constexpr int32_t k_brick_major_version = 1;

std::vector<std::string> load_class_names_from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open JSON file: " + path);
    }

    json j;
    f >> j;

    int64_t min_id = std::numeric_limits<int64_t>::max();
    int64_t max_id = -1;
    for (const auto& [token, id] : j.items()) {
        const int64_t idx = id.get<int64_t>();
        if (idx < 0) {
            throw std::runtime_error("Negative class id found in JSON");
        }
        min_id = std::min(min_id, idx);
        max_id = std::max(max_id, idx);
    }

    const int64_t offset = min_id == 1 ? 1 : 0;
    std::vector<std::string> class_names(static_cast<size_t>(max_id + 1 - offset));
    for (const auto& [token, id] : j.items()) {
        const int64_t idx = id.get<int64_t>() - offset;
        class_names[static_cast<size_t>(idx)] = token;
    }

    return class_names;
}

}  // namespace

ValidationResult Perch::validate(const ComputeFlowManifest& manifest) const {
    const BrickInfo& info = manifest.get_brick_info();
    COMPUTE_FLOW_VALIDATE(info.type == "audio_detection");
    COMPUTE_FLOW_VALIDATE(info.name == "Perch");
    COMPUTE_FLOW_VALIDATE(info.version.major == k_brick_major_version);

    COMPUTE_FLOW_VALIDATE(manifest.has_module("classifier"));
    COMPUTE_FLOW_VALIDATE(manifest["classifier"].model_path != "");
    COMPUTE_FLOW_VALIDATE(manifest["classifier"].framework_type == RunnerType::LiteRT);
    COMPUTE_FLOW_VALIDATE(manifest["classifier"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["classifier"].acc_config.has_accelerator(AcceleratorType::Cpu));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("class_names"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("sample_rate"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("version"));

    const std::string perch_version = manifest.get_constant<std::string>("version");
    COMPUTE_FLOW_VALIDATE(perch_version == "v2");

    return {};
}

void Perch::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    _runner = make_runner(manifest["classifier"].framework_type);
    _runner->setup(manifest["classifier"], profiler);
}

void Perch::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    _sample_rate = manifest.get_constant<int64_t>("sample_rate");
    COMPUTE_FLOW_ASSERT(_sample_rate > 0);

    const std::string class_names_file = *manifest.get_asset("class_names");
    _class_names = load_class_names_from_json(class_names_file);
    COMPUTE_FLOW_ASSERT(!_class_names.empty());

    const TensorFormat audio_input_format = _runner->get_input_format("audio");
    COMPUTE_FLOW_ASSERT(audio_input_format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(audio_input_format.shape.size() == 2);
    COMPUTE_FLOW_ASSERT(audio_input_format.shape[1] > 0);

    _num_samples_per_frame = audio_input_format.shape[1];

    const TensorFormat label_format = _runner->get_output_format("label");
    COMPUTE_FLOW_ASSERT(label_format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(label_format.shape.size() == 2);
    COMPUTE_FLOW_ASSERT(label_format.shape[1] > 0);

    _num_classes = label_format.shape[1];

    COMPUTE_FLOW_ASSERT(static_cast<int64_t>(_class_names.size()) >= _num_classes);

    _max_class_name_bytes = 1;
    for (const auto& class_name : _class_names) {
        _max_class_name_bytes = std::max(_max_class_name_bytes, static_cast<int64_t>(class_name.size()));
    }

    // We generate one frame at the time in streaming mode.
    // One frame corresponds to a full window size consumed by the input model.
    // For this reason, the number of frames is set to 1
    _internal_memory.allocate("id", TensorFormat{DataType::I64, {1, _num_classes}});
    _internal_memory.allocate("score", TensorFormat{DataType::F32, {1, _num_classes}});
    _internal_memory.allocate("name", TensorFormat{DataType::U8, {1, _num_classes, _max_class_name_bytes}});
    _internal_memory.allocate("frame", TensorFormat{DataType::F32, {1, 2}});
}

void Perch::warmup_impl(ProfilerView profiler) {
    TensorDict input;
    input.declare_tensor("raw_wav");
    input.declare_tensor("sample_rate");

    const int64_t window_size_samples = _num_samples_per_frame;
    std::vector<float> mono_raw_audio(static_cast<size_t>(window_size_samples), 0.0f);
    const int64_t sample_rate = _sample_rate;

    input["raw_wav"].data = mono_raw_audio.data();
    input["raw_wav"].format.dt = DataType::F32;
    input["raw_wav"].format.shape = {window_size_samples};
    input["sample_rate"].data = (void*)&sample_rate;
    input["sample_rate"].format.dt = DataType::I64;
    input["sample_rate"].format.shape = {1};

    run_impl(input, profiler);
    reset_states_impl(profiler);
}

void Perch::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)manifest;
    (void)profiler;
}

void Perch::reset_states_impl(ProfilerView profiler) {
    (void)profiler;
    _runner->reset_states();
    _idx_window = 0;
    _num_windows = 0;
    _is_output_ready = false;
}

bool Perch::is_generation_complete() const {
    return _is_output_ready;
}

TensorDict Perch::run_impl(const TensorDict& input, ProfilerView profiler) {
    (void)profiler;

    if (_is_output_ready) {
        return TensorDict{};
    }

    COMPUTE_FLOW_ASSERT(input["raw_wav"].format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(input["raw_wav"].data != nullptr);
    COMPUTE_FLOW_ASSERT(input["sample_rate"].format.dt == DataType::I64);
    COMPUTE_FLOW_ASSERT(input["sample_rate"].data != nullptr);
    COMPUTE_FLOW_ASSERT(reinterpret_cast<const int64_t*>(input["sample_rate"].data)[0] == _sample_rate);

    const auto& raw_wav_shape = input["raw_wav"].format.shape;
    COMPUTE_FLOW_ASSERT(raw_wav_shape.size() == 1);
    COMPUTE_FLOW_ASSERT(raw_wav_shape[0] > 0);

    const int64_t window_size_samples = _num_samples_per_frame;
    if (_idx_window == 0) {
        _num_windows = num_windows(raw_wav_shape[0], window_size_samples);
    }
    COMPUTE_FLOW_ASSERT(_idx_window < _num_windows);

    TensorData audio;
    audio.data   = reinterpret_cast<float*>(input["raw_wav"].data) + (_idx_window * window_size_samples);
    audio.format = TensorFormat{DataType::F32, {1, window_size_samples}};

    _runner->set_input("audio", audio);
    _runner->run();

    sort_scores(_runner->get_output("label"));

    float* frame = _internal_memory.get_data_as<float>("frame");
    frame[0] = static_cast<float>(_idx_window * window_size_samples) /
               static_cast<float>(_sample_rate);
    frame[1] = static_cast<float>((_idx_window + 1) * window_size_samples) /
               static_cast<float>(_sample_rate);

    TensorDict output;
    output.declare_tensor("id")    = _internal_memory.get_tensor("id");
    output.declare_tensor("score") = _internal_memory.get_tensor("score");
    output.declare_tensor("name")  = _internal_memory.get_tensor("name");
    output.declare_tensor("frame") = _internal_memory.get_tensor("frame");

    ++_idx_window;
    _is_output_ready = _idx_window >= _num_windows;
    return output;
}

int64_t Perch::num_windows(int64_t num_samples, int64_t window_samples) {
    COMPUTE_FLOW_ASSERT(window_samples > 0);
    COMPUTE_FLOW_ASSERT(num_samples > 0);
    COMPUTE_FLOW_ASSERT(num_samples % window_samples == 0);
    return num_samples / window_samples;
}

void Perch::sort_scores(const TensorData& scores) {
    COMPUTE_FLOW_ASSERT(scores.format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(scores.data != nullptr);
    COMPUTE_FLOW_ASSERT(!scores.format.shape.empty());
    COMPUTE_FLOW_ASSERT(scores.format.shape.back() == _num_classes);

    const float* score_data = reinterpret_cast<const float*>(scores.data);
    int64_t* out_class_ids = _internal_memory.get_data_as<int64_t>("id");
    float* out_scores = _internal_memory.get_data_as<float>("score");
    uint8_t* out_class_names = _internal_memory.get_data_as<uint8_t>("name");

    std::iota(out_class_ids, out_class_ids + _num_classes, 0);
    std::sort(out_class_ids, out_class_ids + _num_classes, [&](int64_t lhs, int64_t rhs) {
        const float lhs_score = score_data[lhs];
        const float rhs_score = score_data[rhs];
        if (lhs_score != rhs_score) {
            return lhs_score > rhs_score;
        }
        return lhs < rhs;
    });

    std::memset(out_class_names, 0, static_cast<size_t>(_num_classes * _max_class_name_bytes));

    for (int64_t rank = 0; rank < _num_classes; ++rank) {
        const int64_t class_id = out_class_ids[rank];
        const std::string& name = _class_names[static_cast<size_t>(class_id)];

        out_scores[rank] = score_data[class_id];
        std::memcpy(out_class_names + (rank * _max_class_name_bytes), name.data(), name.size());
    }
}

}  // namespace compute_flow::bricks::audio_detection
