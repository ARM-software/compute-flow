//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "SileroVAD.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace compute_flow::bricks::vad {

constexpr int32_t k_brick_major_version = 1;

ValidationResult SileroVAD::validate(const ComputeFlowManifest& manifest) const {
    const BrickInfo& info = manifest.get_brick_info();
    COMPUTE_FLOW_VALIDATE(info.type == "vad");
    COMPUTE_FLOW_VALIDATE(info.name == "SileroVAD");
    COMPUTE_FLOW_VALIDATE(info.version.major == k_brick_major_version);

    COMPUTE_FLOW_VALIDATE(manifest.num_modules() == 1);
    COMPUTE_FLOW_VALIDATE(manifest.has_module("silero"));

    COMPUTE_FLOW_VALIDATE(manifest["silero"].framework_type == RunnerType::ONNXRT);
    COMPUTE_FLOW_VALIDATE(manifest["silero"].model_path != "");

    // At the moment, only the CPU is supported
    COMPUTE_FLOW_VALIDATE(manifest["silero"].acc_config.priorities.size() == 1);

    COMPUTE_FLOW_VALIDATE(manifest.has_constant("sample_rate"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("frame_size"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("context_size"));
    return {};
}

void SileroVAD::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)manifest;
    (void)profiler;
    // TODO: Implement configuration updates in a separate MR.
}

void SileroVAD::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    RunnerType runner_type = manifest["silero"].framework_type;

    _runner["silero"] = make_runner(runner_type);
    _runner["silero"]->setup(manifest["silero"], profiler);
}

void SileroVAD::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    const int64_t sample_rate = manifest.get_constant<int64_t>("sample_rate");
    _frame_sz                 = manifest.get_constant<int64_t>("frame_size");
    _context_sz               = manifest.get_constant<int64_t>("context_size");

    COMPUTE_FLOW_ASSERT(_context_sz > 0);

    TensorFormat audio_input_format = _runner["silero"]->get_input_format("audio");

    COMPUTE_FLOW_ASSERT(audio_input_format.shape.size() == 2);
    audio_input_format.shape[0] = 1;
    audio_input_format.shape[1] = static_cast<int64_t>(_context_sz + _frame_sz);

    TensorFormat sr_input_format = _runner["silero"]->get_input_format("sr");

    if(sr_input_format.shape.empty()) {
        sr_input_format.shape = {1};
    } else {
        sr_input_format.shape[0] = 1;
    }

    _internal_memory.allocate("audio", audio_input_format);
    _internal_memory.allocate("sr", sr_input_format);

    int64_t* sr_ptr = _internal_memory.get_data_as<int64_t>("sr");
    *sr_ptr = sample_rate;
}

void SileroVAD::warmup_impl(ProfilerView profiler) {
    TensorDict input;
    TensorDict output;

    // Set the input
    input.declare_tensor("raw_wav");
    input.declare_tensor("sample_rate");

    std::vector<float> mono_raw_audio(_frame_sz * 4);
    const int64_t sample_rate = 16000;

    input["raw_wav"].data         = mono_raw_audio.data();
    input["raw_wav"].format.dt    = DataType::F32;
    input["raw_wav"].format.shape = {static_cast<int64_t>(mono_raw_audio.size())};
    input["sample_rate"].data         = const_cast<int64_t*>(&sample_rate);
    input["sample_rate"].format.dt    = DataType::I64;
    input["sample_rate"].format.shape = {1};

    // Run the model inference. We can ignore the output
    run_impl(input, profiler);
}

void SileroVAD::reset_states_impl(ProfilerView profiler) {
    auto t_audio_in = _internal_memory.get_tensor("audio");

    // Fill the context samples with 0
    memset(t_audio_in.data, 0, _context_sz * sizeof(float));

    _runner["silero"]->reset_states();
    _is_output_ready = false;
}

bool SileroVAD::is_generation_complete() const {
    return _is_output_ready;
}

TensorDict SileroVAD::run_impl(const TensorDict& input, ProfilerView profiler) {
    _is_output_ready = true;

    COMPUTE_FLOW_ASSERT(input["raw_wav"].format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(input["raw_wav"].data != nullptr);
    COMPUTE_FLOW_ASSERT(input["raw_wav"].format.shape.size() == 1);
    COMPUTE_FLOW_ASSERT(input["raw_wav"].format.shape[0] > 0);
    COMPUTE_FLOW_ASSERT(input["sample_rate"].format.dt == DataType::I64);
    COMPUTE_FLOW_ASSERT(input["sample_rate"].data != nullptr);
    COMPUTE_FLOW_ASSERT(input["sample_rate"].format.shape.size() == 1);
    COMPUTE_FLOW_ASSERT(input["sample_rate"].format.shape[0] == 1);

    const int64_t total_in_samples = input["raw_wav"].format.shape[0];

    const int64_t num_frames = get_num_output_frames(total_in_samples);

    _internal_memory.allocate("score", TensorFormat{DataType::F32, {num_frames}});
    _internal_memory.allocate("frame", TensorFormat{DataType::F32, {num_frames, 2}});

    const auto t_audio_in = _internal_memory.get_tensor("audio");
    const auto t_sr_in = _internal_memory.get_tensor("sr");
    const auto t_score_out = _internal_memory.get_tensor("score");
    const auto t_frame_out = _internal_memory.get_tensor("frame");

    float* const scores = reinterpret_cast<float*>(t_score_out.data);
    float* const frames = reinterpret_cast<float*>(t_frame_out.data);

    const int64_t total_bytes             = total_in_samples * sizeof(float);
    const int64_t context_bytes           = _context_sz * sizeof(float);
    const int64_t frame_bytes             = _frame_sz * sizeof(float);
    const int64_t frame_and_context_bytes = context_bytes + frame_bytes;

    // Check whether the sample rate is what we expect.
    const int64_t sample_rate = reinterpret_cast<const int64_t*>(t_sr_in.data)[0];
    COMPUTE_FLOW_ASSERT(
        sample_rate == reinterpret_cast<const int64_t*>(input["sample_rate"].data)[0]);

    // TODO: Add resempler if the input sample rate is not what we expect

    //    Memory layout of t_audio_in.data (contiguous float buffer):
    //
    //    |<------ context_bytes ------>|<----------- frame_bytes ------------>|
    //    +-----------------------------+--------------------------------------+
    //    |           CONTEXT           |               FRAME                  |
    //    +-----------------------------+--------------------------------------+
    //    ^                             ^        |<------ context_bytes ------>|
    //    |                             |        |                             |
    //    curr_context_ptr          frame_ptr    next_context_ptr

    // - curr_context_ptr = start of CONTEXT      (offset = 0)
    // - frame_ptr        = start of FRAME        (offset = context_bytes)
    // - next_context_ptr = start of NEXT CONTEXT (offset = frame_bytes)

    // The starting position for the next context (next_context_ptr) is frame_and_context_bytes - context_bytes = frame_bytes
    void* curr_context_ptr = t_audio_in.data;
    void* next_context_ptr = (void*)((uint8_t*)t_audio_in.data + frame_bytes);
    void* frame_ptr        = (void*)((uint8_t*)t_audio_in.data + context_bytes);

    // Streaming support: use reset_states() to invalid the context and start a new stream

    for (int64_t frame_off_bytes = 0; frame_off_bytes < total_bytes; frame_off_bytes+=frame_bytes) {

        // Number of samples to copy from the input audio
        const int64_t copy_bytes = frame_off_bytes + frame_bytes >= total_bytes? (total_bytes - frame_off_bytes) : frame_bytes;

        // Frame index
        const int64_t frame_idx = frame_off_bytes / frame_bytes;

        // Reset to 0 the frame samples because if we need to copy less than frame size samples, the left over should be 0
        memset(frame_ptr, 0, frame_bytes);

        // We either copy a full frame or partial frame
        memcpy(frame_ptr, (uint8_t*)input["raw_wav"].data + frame_off_bytes, copy_bytes);

        _runner["silero"]->set_input("audio", t_audio_in);
        _runner["silero"]->set_input("sr", t_sr_in);
        _runner["silero"]->run();
        const auto prob_output = _runner["silero"]->get_output("speech_prob");

        scores[frame_idx] = reinterpret_cast<const float*>(prob_output.data)[0];

        const int64_t frame_start_sample = frame_idx * _frame_sz;
        const int64_t frame_end_sample = std::min(frame_start_sample + _frame_sz, total_in_samples);
        frames[frame_idx * 2] = static_cast<float>(frame_start_sample) / sample_rate;
        frames[frame_idx * 2 + 1] = static_cast<float>(frame_end_sample) / sample_rate;

        // Update the current context with the context_sz samples from the audio input
        if(copy_bytes >= context_bytes) {
            memcpy(curr_context_ptr, next_context_ptr, context_bytes);
        } else {
            // Not enough audio for a full context (last partial frame):
            // Left-pad with zeros, right-align the available bytes.
            memset(curr_context_ptr, 0, context_bytes);

            // Copy the available bytes (copy_bytes) into the end of the context buffer.
            memcpy((uint8_t*)curr_context_ptr + (context_bytes - copy_bytes), next_context_ptr, copy_bytes);
        }
    }

    TensorDict output;

    output.declare_tensor("score") = t_score_out;
    output.declare_tensor("frame") = t_frame_out;

    return output;
}

int64_t SileroVAD::get_num_output_frames(int64_t total_input_samples) const {
    return (total_input_samples + _frame_sz - 1) / _frame_sz;
}

} // namespace compute_flow::bricks::vad
