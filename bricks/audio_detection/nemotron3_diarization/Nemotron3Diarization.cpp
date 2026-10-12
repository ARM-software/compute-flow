//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "Nemotron3Diarization.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

namespace compute_flow::bricks::audio_detection {

ValidationResult Nemotron3Diarization::validate(const ComputeFlowManifest& manifest) const {
    const auto& info = manifest.get_brick_info();

    COMPUTE_FLOW_VALIDATE(info.type == "audio_detection");
    COMPUTE_FLOW_VALIDATE(info.name == "Nemotron3Diarization");
    COMPUTE_FLOW_VALIDATE(info.version.major == 1);

    // Check modules presence
    COMPUTE_FLOW_VALIDATE(manifest.has_module("frontend"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("encoder"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("speaker_cache"));

    // Check valid file for each module
    COMPUTE_FLOW_VALIDATE(!manifest["frontend"].model_path.empty());
    COMPUTE_FLOW_VALIDATE(!manifest["encoder"].model_path.empty());
    COMPUTE_FLOW_VALIDATE(!manifest["speaker_cache"].model_path.empty());

    // Check valid runner for each module
    COMPUTE_FLOW_VALIDATE(manifest["frontend"].framework_type == RunnerType::ExecuTorch);
    COMPUTE_FLOW_VALIDATE(manifest["encoder"].framework_type == RunnerType::ExecuTorch ||
                          manifest["encoder"].framework_type == RunnerType::LiteRT);
    COMPUTE_FLOW_VALIDATE(manifest["speaker_cache"].framework_type == RunnerType::LiteRT);

    // Check assets
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("class_names"));
    COMPUTE_FLOW_VALIDATE(!manifest.get_asset("class_names")->empty());

    // Check configs presence
    COMPUTE_FLOW_VALIDATE(manifest.has_config("threshold"));

    // Check constants presence
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("sample_rate"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("hop_length"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("fft_length"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("subsampling_factor"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("cache_length"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("max_chunk_length"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("enable_streaming_mode"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("offline_chunk_length"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("offline_right_context"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("offline_fifo_length"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("offline_cache_update_period"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("streaming_chunk_length"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("streaming_right_context"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("streaming_fifo_length"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("streaming_cache_update_period"));

    // Validate constants values
    const auto sample_rate      = manifest.get_constant<int64_t>("sample_rate");
    const auto hop_length       = manifest.get_constant<int64_t>("hop_length");
    const auto fft_length       = manifest.get_constant<int64_t>("fft_length");
    const auto max_chunk_length = manifest.get_constant<int64_t>("max_chunk_length");

    COMPUTE_FLOW_VALIDATE(sample_rate > 0 && hop_length > 0 && fft_length > hop_length);
    COMPUTE_FLOW_VALIDATE(fft_length % 2 == 0);
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("subsampling_factor") > 0);
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("cache_length") > 0);
    COMPUTE_FLOW_VALIDATE(max_chunk_length > 0);

    for (const std::string prefix : {"offline_", "streaming_"}) {
        const auto chunk_length  = manifest.get_constant<int64_t>(prefix + "chunk_length");
        const auto update_period = manifest.get_constant<int64_t>(prefix + "cache_update_period");

        COMPUTE_FLOW_VALIDATE(chunk_length > 0 && chunk_length <= max_chunk_length);
        COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>(prefix + "right_context") >= 0);
        COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>(prefix + "fifo_length") > 0);
        COMPUTE_FLOW_VALIDATE(update_period > 0 && update_period <= max_chunk_length);
    }

    return {};
}

void Nemotron3Diarization::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    _frontend      = make_runner(manifest["frontend"].framework_type);
    _encoder       = make_runner(manifest["encoder"].framework_type);
    _speaker_cache = make_runner(manifest["speaker_cache"].framework_type);

    _frontend->setup(manifest["frontend"], profiler);
    _encoder->setup(manifest["encoder"], profiler);
    _speaker_cache->setup(manifest["speaker_cache"], profiler);
}

void Nemotron3Diarization::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {

    // Get the constants
    _en_streaming_mode       = manifest.get_constant<bool>("enable_streaming_mode");
    _sample_rate             = manifest.get_constant<int64_t>("sample_rate");
    _hop                     = manifest.get_constant<int64_t>("hop_length");
    _fft_length              = manifest.get_constant<int64_t>("fft_length");
    _factor                  = manifest.get_constant<int64_t>("subsampling_factor");
    _cache_length            = manifest.get_constant<int64_t>("cache_length");

    const std::string prefix = _en_streaming_mode ? "streaming_" : "offline_";
    _chunk_length            = manifest.get_constant<int64_t>(prefix + "chunk_length");
    const auto right_context = manifest.get_constant<int64_t>(prefix + "right_context");
    const auto fifo_length   = manifest.get_constant<int64_t>(prefix + "fifo_length");
    const auto update_period = manifest.get_constant<int64_t>(prefix + "cache_update_period");

    // The frontend model computes the STFT and mel features. These sizes only select its waveform window.
    _window_length                  = _chunk_length + right_context;
    _context_length                 = _cache_length + fifo_length;
    _window_samples                 = _window_length * _factor * _hop + _fft_length - _hop + 1;
    const auto embeds_format        = _frontend->get_output_format("embeds");
    const auto probabilities_format = _encoder->get_output_format("probabilities");
    const auto cache_chunk_format   = _speaker_cache->get_input_format("chunk");
    const auto cache_probs_format   = _speaker_cache->get_input_format("recent_probs");

    _hidden_size      = embeds_format.shape[2];
    _num_speakers     = probabilities_format.shape[2];
    _max_chunk_length = cache_chunk_format.shape[1];
    // The LiteRT encoder declares dynamic frame dimensions. The cache input fixes the
    // maximum frame count used for encoder scratch storage and cache padding.
    _encoder_length = cache_probs_format.shape[1];
    COMPUTE_FLOW_CHECK(_encoder_length >= _context_length + _window_length);

    // Only boundary windows need an audio copy; interior windows use the caller's storage.
    for (const auto* name : {"audio", "valid_start", "valid_end", "valid_frames"}) {
        auto format = _frontend->get_input_format(name);

        if (std::string_view(name) == "audio") {
            format.shape[1] = _window_samples;
        }

        _memory.allocate(name, format);
    }

    auto mask_format = _encoder->get_input_format("mask");
    mask_format.shape[1] = _encoder_length;
    _memory.allocate("mask", mask_format);
    _memory.allocate("sequence_length", _encoder->get_input_format("sequence_length"));

    // Encoder inputs and cache padding are consumed sequentially, so share their scratch storage.
    auto encoder_embeds_format = _encoder->get_input_format("embeds");
    encoder_embeds_format.shape[1] = _encoder_length;
    const auto scratch_bytes =
        std::max({encoder_embeds_format.get_nbytes(), cache_chunk_format.get_nbytes(), cache_probs_format.get_nbytes()});

    _memory.allocate("model_scratch", {DataType::F32, {scratch_bytes / static_cast<int64_t>(sizeof(float))}});
    _memory.allocate("parameters", {DataType::I32, {3}});

    auto* parameters = _memory.get_data_as<int32_t>("parameters");
    parameters[0]    = static_cast<int32_t>(_chunk_length);
    parameters[1]    = static_cast<int32_t>(fifo_length);
    parameters[2]    = static_cast<int32_t>(update_period);

    // Get the class names
    std::ifstream class_names_file(*manifest.get_asset("class_names"));

    COMPUTE_FLOW_CHECK(class_names_file.good());

    const auto class_names = nlohmann::json::parse(class_names_file);

    COMPUTE_FLOW_CHECK(class_names.is_object() && static_cast<int64_t>(class_names.size()) == _num_speakers);

    _speaker_names.assign(static_cast<uint64_t>(_num_speakers), "");
    _max_name_bytes = 1;

    for (const auto& [name, value] : class_names.items()) {
        COMPUTE_FLOW_CHECK(!name.empty() && value.is_number_integer());

        const auto speaker = value.get<int64_t>();

        COMPUTE_FLOW_CHECK(speaker >= 0 && speaker < _num_speakers);
        COMPUTE_FLOW_CHECK(_speaker_names[static_cast<uint64_t>(speaker)].empty());

        _speaker_names[static_cast<uint64_t>(speaker)] = name;
        _max_name_bytes = std::max(_max_name_bytes, static_cast<int64_t>(name.size()) + 1);
    }

    const auto max_segments = _num_speakers * (_chunk_length * _factor / 2 + 1);

    _memory.allocate("id", {DataType::I64, {max_segments, 1}});
    _memory.allocate("score", {DataType::F32, {max_segments, 1}});
    _memory.allocate("name", {DataType::U8, {max_segments, 1, _max_name_bytes}});
    _memory.allocate("frame", {DataType::F32, {max_segments, 2}});
    _segments.reserve(static_cast<uint64_t>(max_segments));
    _open_segments.resize(static_cast<uint64_t>(_num_speakers));
    _score_sums.resize(static_cast<uint64_t>(_num_speakers));

    update_config_impl(manifest, profiler);
}

void Nemotron3Diarization::warmup_impl(ProfilerView profiler) {
    // One chunk plus one frame keeps the first call non-final, warming the cache model too.
    const auto num_warmup_samples = (_chunk_length * _factor + 1) * _hop;
    std::vector<float> silence(static_cast<uint64_t>(num_warmup_samples), 0.0F);
    TensorDict input;

    input.declare_tensor("raw_wav")     = {{DataType::F32, {num_warmup_samples}}, silence.data(), true};
    input.declare_tensor("sample_rate") = {{DataType::I64, {1}}, &_sample_rate, true};

    // BrickBaseV1 resets all state after warmup.
    run_impl(input, profiler);
}

void Nemotron3Diarization::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    const auto threshold = manifest.get_config<float>("threshold");

    COMPUTE_FLOW_CHECK(std::isfinite(threshold) && threshold >= 0.0F && threshold <= 1.0F);

    _threshold = threshold;
}

void Nemotron3Diarization::reset_states_impl(ProfilerView profiler) {
    (void)profiler;

    _frontend->reset_states();
    _encoder->reset_states();
    _speaker_cache->reset_states();
    std::fill(_open_segments.begin(), _open_segments.end(), -1);
    std::fill(_score_sums.begin(), _score_sums.end(), 0.0);
    _segments.clear();

    _num_recording_samples = -1;
    _frame_offset          = 0;
    _num_context           = 0;
    _complete              = false;
}

bool Nemotron3Diarization::is_generation_complete() const { return _complete; }

void Nemotron3Diarization::run_window(const float* audio, int64_t num_samples, bool update_cache) {
    const auto num_padding_frames = _en_streaming_mode ? 0 : 1;
    const auto num_groups =
        std::min(_window_length, (num_samples / _hop + num_padding_frames - _frame_offset + _factor - 1) / _factor);
    const auto num_frontend_samples = num_groups * _factor * _hop + _fft_length - _hop + 1;
    const auto first_sample         = _frame_offset * _hop - _fft_length / 2 - 1;
    const auto available_start      = std::max<int64_t>(first_sample, 0);
    const auto available_end        = std::min(first_sample + num_frontend_samples, num_samples);
    const auto valid_start          = available_start - first_sample;
    const auto valid_end            = std::max(valid_start, available_end - first_sample);
    const auto valid_frames         = std::min(num_samples / _hop - _frame_offset, _window_length * _factor);
    auto audio_input                = _memory.get_tensor("audio");

    audio_input.format.shape[1] = num_frontend_samples;

    if (valid_start == 0 && valid_end == num_frontend_samples) {
        // The CPU runner finishes reading this view before run_window() returns.
        audio_input.data = const_cast<float*>(audio + available_start);
    } else {
        // Leading padding needs storage before the recording; trailing padding needs storage after it.
        auto* window = static_cast<float*>(audio_input.data);

        std::fill_n(window, valid_start, 0.0F);
        std::copy_n(audio + available_start, valid_end - valid_start, window + valid_start);
        std::fill_n(window + valid_end, num_frontend_samples - valid_end, 0.0F);
    }

    // Set the inputs of the frontend model
    _frontend->set_input("audio", audio_input);

    *_memory.get_data_as<int32_t>("valid_start") = static_cast<int32_t>(valid_start);
    *_memory.get_data_as<int32_t>("valid_end")   = static_cast<int32_t>(valid_end);
    *_memory.get_data_as<int32_t>("valid_frames") = static_cast<int32_t>(valid_frames);
    for (const auto* name : {"valid_start", "valid_end", "valid_frames"}) {
        _frontend->set_input(name, _memory.get_tensor(name));
    }

    _frontend->run();

    const auto embeds             = _frontend->get_output("embeds");
    const auto mask               = _frontend->get_output("mask");
    const auto num_encoder_frames = _num_context + num_groups;
    auto encoder_embeds           = embeds;
    auto encoder_mask             = mask;
    auto* scratch                 = _memory.get_data_as<float>("model_scratch");

    // Without cache context, connect the frontend outputs directly to the encoder.
    if (_num_context > 0) {
        auto* combined_mask = _memory.get_data_as<int32_t>("mask");
        const auto context  = _speaker_cache->get_output("context");

        std::copy_n(static_cast<const float*>(context.data), _num_context * _hidden_size, scratch);
        std::fill_n(combined_mask, _num_context, 1);
        std::copy_n(static_cast<const float*>(embeds.data), num_groups * _hidden_size,
                    scratch + _num_context * _hidden_size);
        std::copy_n(static_cast<const int32_t*>(mask.data), num_groups, combined_mask + _num_context);

        encoder_embeds = {{DataType::F32, {1, num_encoder_frames, _hidden_size}}, scratch, true};
        encoder_mask   = {{DataType::I32, {1, num_encoder_frames}}, combined_mask, true};
    }

    *_memory.get_data_as<int32_t>("sequence_length") = static_cast<int32_t>(_num_context + num_groups);

    _encoder->set_input("embeds", encoder_embeds);
    _encoder->set_input("mask", encoder_mask);
    _encoder->set_input("sequence_length", _memory.get_tensor("sequence_length"));
    _encoder->run();

    if (update_cache) {
        auto recent_probs           = _encoder->get_output("pooled");
        const auto num_chunk_groups = std::min(_chunk_length, num_groups);
        auto chunk                  = embeds;

        chunk.format.shape[1] = _max_chunk_length;

        // Borrow full chunks; use the encoder's finished scratch space when padding is needed.
        if (num_chunk_groups < _max_chunk_length) {
            std::copy_n(static_cast<const float*>(embeds.data), num_chunk_groups * _hidden_size, scratch);
            std::fill_n(scratch + num_chunk_groups * _hidden_size,
                        (_max_chunk_length - num_chunk_groups) * _hidden_size, 0.0F);

            chunk.data        = scratch;
            chunk.user_hosted = true;
        }

        _speaker_cache->set_input("parameters", _memory.get_tensor("parameters"));
        _speaker_cache->set_input("chunk", chunk);

        // LiteRT copies set_input() synchronously, so the same padding storage can be reused.
        if (num_encoder_frames < _encoder_length) {
            std::copy_n(static_cast<const float*>(recent_probs.data), num_encoder_frames * _num_speakers, scratch);
            std::fill_n(scratch + num_encoder_frames * _num_speakers,
                        (_encoder_length - num_encoder_frames) * _num_speakers, 0.0F);

            recent_probs.format.shape[1] = _encoder_length;
            recent_probs.data            = scratch;
            recent_probs.user_hosted     = true;
        }

        _speaker_cache->set_input("recent_probs", recent_probs);
        _speaker_cache->run();
    }
}

void Nemotron3Diarization::process_window(const float* audio, int64_t num_samples, int64_t num_output_frames,
                                        bool is_last_window) {
    if (num_output_frames > 0) {
        run_window(audio, num_samples, !is_last_window);

        const auto output         = _encoder->get_output("probabilities");
        const auto* probabilities = static_cast<const float*>(output.data) + _num_context * _factor * _num_speakers;

        for (int64_t frame = 0; frame < num_output_frames; ++frame) {
            for (int64_t speaker = 0; speaker < _num_speakers; ++speaker) {
                auto& start            = _open_segments[static_cast<uint64_t>(speaker)];
                auto& score_sum        = _score_sums[static_cast<uint64_t>(speaker)];
                const auto probability = probabilities[frame * _num_speakers + speaker];
                const auto active      = probability > _threshold;

                if (active && start < 0) {
                    start = _frame_offset + frame;
                } else if (!active && start >= 0) {
                    if (start < _frame_offset + frame) {
                        _segments.push_back(
                            {start, _frame_offset + frame, speaker, score_sum / (_frame_offset + frame - start)});
                    }

                    start     = -1;
                    score_sum = 0.0;
                }

                if (active) {
                    score_sum += probability;
                }
            }
        }

        _frame_offset += num_output_frames;

        if (!is_last_window) {
            _num_context = *static_cast<const int32_t*>(_speaker_cache->get_output("context_length").data);

            COMPUTE_FLOW_CHECK(_num_context >= 0 && _num_context <= _context_length);
        }
    }

    if (is_last_window || _en_streaming_mode) {
        for (int64_t speaker = 0; speaker < _num_speakers; ++speaker) {
            auto& start = _open_segments[static_cast<uint64_t>(speaker)];

            if (start >= 0) {
                if (start < _frame_offset) {
                    _segments.push_back({start, _frame_offset, speaker,
                                         _score_sums[static_cast<uint64_t>(speaker)] / (_frame_offset - start)});
                }

                start                                       = is_last_window ? -1 : _frame_offset;
                _score_sums[static_cast<uint64_t>(speaker)] = 0.0;
            }
        }
    }

    _complete = is_last_window;
}

TensorDict Nemotron3Diarization::run_impl(const TensorDict& input, ProfilerView profiler) {
    (void)profiler;
    _segments.clear();

    if (_complete) {
        return output_segments();
    }

    const auto& waveform = input["raw_wav"];
    const auto& rate     = input["sample_rate"];

    COMPUTE_FLOW_CHECK(waveform.format.dt == DataType::F32 && waveform.format.shape.size() == 1);
    COMPUTE_FLOW_CHECK(waveform.format.shape[0] >= 0);
    COMPUTE_FLOW_CHECK(waveform.data != nullptr || waveform.format.shape[0] == 0);

    COMPUTE_FLOW_CHECK(rate.data != nullptr && *static_cast<const int64_t*>(rate.data) == _sample_rate);

    if (_num_recording_samples < 0) {
        // Capture the recording length on the first call after reset.
        _num_recording_samples = waveform.format.shape[0];
    } else {
        // Later calls advance through the same complete recording, rather than new audio chunks.
        COMPUTE_FLOW_CHECK(waveform.format.shape[0] == _num_recording_samples);
    }

    const auto num_frames        = _num_recording_samples / _hop;
    const auto num_output_frames = std::min(_chunk_length * _factor, num_frames - _frame_offset);
    const auto is_last_window    = _frame_offset + num_output_frames == num_frames;

    process_window(static_cast<const float*>(waveform.data), _num_recording_samples, num_output_frames, is_last_window);

    return output_segments();
}

TensorDict Nemotron3Diarization::output_segments() {
    std::sort(_segments.begin(), _segments.end(), [](const Segment& left, const Segment& right) {
        return std::tie(left.start, left.speaker) < std::tie(right.start, right.speaker);
    });

    const auto num_segments = static_cast<int64_t>(_segments.size());

    auto* ids                    = _memory.get_data_as<int64_t>("id");
    auto* scores                 = _memory.get_data_as<float>("score");
    auto* names                  = _memory.get_data_as<uint8_t>("name");
    auto* frames                 = _memory.get_data_as<float>("frame");
    const auto seconds_per_frame = static_cast<double>(_hop) / static_cast<double>(_sample_rate);

    COMPUTE_FLOW_CHECK(num_segments <= _memory.get_tensor("id").format.shape[0]);

    std::fill_n(names, num_segments * _max_name_bytes, 0);

    for (int64_t index = 0; index < num_segments; ++index) {
        const auto& segment = _segments[static_cast<uint64_t>(index)];
        ids[index]          = segment.speaker;
        scores[index]       = static_cast<float>(segment.score);
        const auto& name    = _speaker_names[static_cast<uint64_t>(segment.speaker)];

        std::memcpy(names + index * _max_name_bytes, name.data(), name.size());

        frames[index * 2]     = static_cast<float>(segment.start * seconds_per_frame);
        frames[index * 2 + 1] = static_cast<float>(segment.end * seconds_per_frame);
    }

    TensorDict output;

    output.declare_tensor("id")    = {{DataType::I64, {num_segments, 1}}, ids, false};
    output.declare_tensor("score") = {{DataType::F32, {num_segments, 1}}, scores, false};
    output.declare_tensor("name")  = {{DataType::U8, {num_segments, 1, _max_name_bytes}}, names, false};
    output.declare_tensor("frame") = {{DataType::F32, {num_segments, 2}}, frames, false};

    return output;
}

}  // namespace compute_flow::bricks::audio_detection
