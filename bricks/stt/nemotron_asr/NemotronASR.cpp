// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

#include "NemotronASR.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <utility>

#include <nlohmann/json.hpp>

namespace compute_flow::bricks::stt {
namespace {

void require_matching_state(IRunner& runner, const std::string& name, DataType dtype, int64_t rank) {
    const auto input  = runner.get_input_format(name);
    const auto output = runner.get_output_format(name);

    COMPUTE_FLOW_CHECK(input.dt == dtype && static_cast<int64_t>(input.shape.size()) == rank);
    COMPUTE_FLOW_CHECK(std::all_of(input.shape.begin(), input.shape.end(), [](int64_t dim) { return dim > 0; }));
    COMPUTE_FLOW_CHECK(output.dt == dtype && output.shape == input.shape);
}

int64_t prompt_id_for_language(const std::string& language, const std::string& map_path) {
    std::ifstream file(map_path);
    COMPUTE_FLOW_CHECK(file.is_open());

    const auto mapping = nlohmann::json::parse(file, nullptr, false);
    COMPUTE_FLOW_CHECK(!mapping.is_discarded());
    COMPUTE_FLOW_CHECK(mapping.is_object() && mapping.contains(language) && mapping.at(language).is_number_integer());

    return mapping.at(language).get<int64_t>();
}

}  // namespace

ValidationResult NemotronASR::validate(const ComputeFlowManifest& manifest) const {
    const auto& info = manifest.get_brick_info();

    COMPUTE_FLOW_VALIDATE(info.type == "stt");
    COMPUTE_FLOW_VALIDATE(info.name == "NemotronASR");
    COMPUTE_FLOW_VALIDATE(info.version.major == 1);
    COMPUTE_FLOW_VALIDATE(manifest.num_modules() == 4);
    COMPUTE_FLOW_VALIDATE(manifest.has_module("encoder"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("predictor"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("joiner"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("detokenizer"));

    for (const auto& model : manifest.models()) {
        COMPUTE_FLOW_VALIDATE(!model.model_path.empty());
        COMPUTE_FLOW_VALIDATE(model.acc_config.priorities.size() == 1);
        COMPUTE_FLOW_VALIDATE(model.acc_config.is_accelerator_first_priority(AcceleratorType::Cpu));
    }

    COMPUTE_FLOW_VALIDATE(manifest["encoder"].framework_type == RunnerType::ExecuTorch);
    COMPUTE_FLOW_VALIDATE(manifest["predictor"].framework_type == RunnerType::ExecuTorch);
    COMPUTE_FLOW_VALIDATE(manifest["joiner"].framework_type == RunnerType::ExecuTorch);
    COMPUTE_FLOW_VALIDATE(manifest["detokenizer"].framework_type == RunnerType::SentencePieceDecode);

    for (const auto* key :
         {"sample_rate", "window_samples", "hop_samples", "fft_size", "preemphasis", "log_guard",
          "overlap_mel_frames", "blank_token_id"}) {
        COMPUTE_FLOW_VALIDATE(manifest.has_constant(key));
    }

    COMPUTE_FLOW_VALIDATE(manifest.has_asset("languages_map_file"));
    COMPUTE_FLOW_VALIDATE(manifest.has_config("language"));
    COMPUTE_FLOW_VALIDATE(!manifest.has_config("prompt_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_config("max_symbols_per_frame"));

    const auto& encoder     = manifest["encoder"];
    const auto& predictor   = manifest["predictor"];
    const auto& joiner      = manifest["joiner"];
    const auto& detokenizer = manifest["detokenizer"];

    COMPUTE_FLOW_VALIDATE(encoder.inputs.size() == 3);
    COMPUTE_FLOW_VALIDATE(encoder.has_input("prompt_id"));
    COMPUTE_FLOW_VALIDATE(encoder.outputs.size() == 2 && encoder.states.size() == 3);
    COMPUTE_FLOW_VALIDATE(encoder.has_input("mel") && encoder.has_input("mel_length"));
    COMPUTE_FLOW_VALIDATE(encoder.has_output("encoded") && encoder.has_output("encoded_length"));
    COMPUTE_FLOW_VALIDATE(encoder.has_state("attention") && encoder.has_state("convolution"));
    COMPUTE_FLOW_VALIDATE(encoder.has_state("cache_length"));
    COMPUTE_FLOW_VALIDATE(predictor.inputs.size() == 1 && predictor.outputs.size() == 1 &&
                          predictor.states.size() == 2);
    COMPUTE_FLOW_VALIDATE(predictor.has_input("token") && predictor.has_output("prediction"));
    COMPUTE_FLOW_VALIDATE(predictor.has_state("hidden") && predictor.has_state("cell"));
    COMPUTE_FLOW_VALIDATE(joiner.inputs.size() == 2 && joiner.outputs.size() == 1 && joiner.states.empty());
    COMPUTE_FLOW_VALIDATE(joiner.has_input("encoded") && joiner.has_input("prediction"));
    COMPUTE_FLOW_VALIDATE(joiner.has_output("logits"));
    COMPUTE_FLOW_VALIDATE(detokenizer.inputs.size() == 1 && detokenizer.outputs.size() == 1 &&
                          detokenizer.states.empty());
    COMPUTE_FLOW_VALIDATE(detokenizer.has_input("tokens") && detokenizer.has_output("text"));

    COMPUTE_FLOW_VALIDATE(encoder.input("mel") == 0 && encoder.input("mel_length") == 1);
    COMPUTE_FLOW_VALIDATE(encoder.output("encoded") == 0 && encoder.output("encoded_length") == 1);
    COMPUTE_FLOW_VALIDATE(encoder.state("attention") == std::make_pair(int64_t{2}, int64_t{2}));
    COMPUTE_FLOW_VALIDATE(encoder.state("convolution") == std::make_pair(int64_t{3}, int64_t{3}));
    COMPUTE_FLOW_VALIDATE(encoder.state("cache_length") == std::make_pair(int64_t{4}, int64_t{4}));

    COMPUTE_FLOW_VALIDATE(encoder.input("prompt_id") == 5);

    COMPUTE_FLOW_VALIDATE(predictor.input("token") == 0 && predictor.output("prediction") == 0);
    COMPUTE_FLOW_VALIDATE(predictor.state("hidden") == std::make_pair(int64_t{1}, int64_t{1}));
    COMPUTE_FLOW_VALIDATE(predictor.state("cell") == std::make_pair(int64_t{2}, int64_t{2}));
    COMPUTE_FLOW_VALIDATE(joiner.input("encoded") == 0 && joiner.input("prediction") == 1);
    COMPUTE_FLOW_VALIDATE(joiner.output("logits") == 0);
    COMPUTE_FLOW_VALIDATE(detokenizer.input("tokens") == 0 && detokenizer.output("text") == 0);
    return {};
}

void NemotronASR::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    _encoder     = make_runner(manifest["encoder"].framework_type);
    _predictor   = make_runner(manifest["predictor"].framework_type);
    _joiner      = make_runner(manifest["joiner"].framework_type);
    _detokenizer = make_runner(manifest["detokenizer"].framework_type);

    _encoder->setup(manifest["encoder"], profiler);
    _predictor->setup(manifest["predictor"], profiler);
    _joiner->setup(manifest["joiner"], profiler);
    _detokenizer->setup(manifest["detokenizer"], profiler);
}

void NemotronASR::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    _sample_rate    = manifest.get_constant<int64_t>("sample_rate");
    _hop_samples    = manifest.get_constant<int64_t>("hop_samples");
    _overlap_frames = manifest.get_constant<int64_t>("overlap_mel_frames");
    _blank_token_id = manifest.get_constant<int64_t>("blank_token_id");
    _started        = false;
    _complete       = false;

    // validate() runs before model loading. ModelData retains only tensor mappings,
    // so the actual shapes and dtypes can only be checked after runner setup.
    const auto mel = _encoder->get_input_format("mel");

    COMPUTE_FLOW_CHECK(mel.dt == DataType::F32 && mel.shape.size() == 3);
    COMPUTE_FLOW_CHECK(mel.shape[0] == 1 && mel.shape[1] > 0 && mel.shape[2] > 0);
    COMPUTE_FLOW_CHECK(_overlap_frames > 0 && _overlap_frames < mel.shape[2]);

    _mel_bins     = mel.shape[1];
    _input_frames = mel.shape[2];
    _fresh_frames = _input_frames - _overlap_frames;

    const auto mel_length     = _encoder->get_input_format("mel_length");
    const auto encoded_length = _encoder->get_output_format("encoded_length");

    COMPUTE_FLOW_CHECK(mel_length.dt == DataType::I64 && mel_length.shape == std::vector<int64_t>{1});
    COMPUTE_FLOW_CHECK(encoded_length.dt == DataType::I64 && encoded_length.shape == std::vector<int64_t>{1});
    require_matching_state(*_encoder, "attention", DataType::F32, 4);
    require_matching_state(*_encoder, "convolution", DataType::F32, 4);
    require_matching_state(*_encoder, "cache_length", DataType::I64, 1);

    const auto prompt = _encoder->get_input_format("prompt_id");

    COMPUTE_FLOW_CHECK(prompt.dt == DataType::I64 && prompt.shape == std::vector<int64_t>{1});

    const auto encoded    = _encoder->get_output_format("encoded");
    const auto prediction = _predictor->get_output_format("prediction");

    COMPUTE_FLOW_CHECK(encoded.dt == DataType::F32 && encoded.shape.size() == 3);
    COMPUTE_FLOW_CHECK(encoded.shape[0] == 1 && encoded.shape[1] > 0 && encoded.shape[2] > 0);
    COMPUTE_FLOW_CHECK(prediction.dt == DataType::F32 && prediction.shape.size() == 3);
    COMPUTE_FLOW_CHECK(prediction.shape[0] == 1 && prediction.shape[1] == 1 && prediction.shape[2] > 0);

    _encoder_dimension = encoded.shape[2];

    const auto token = _predictor->get_input_format("token");

    COMPUTE_FLOW_CHECK((token.dt == DataType::I64 && token.shape == std::vector<int64_t>{1, 1}));
    require_matching_state(*_predictor, "hidden", DataType::F32, 3);
    require_matching_state(*_predictor, "cell", DataType::F32, 3);

    const auto cell              = _predictor->get_input_format("cell");
    const auto hidden            = _predictor->get_input_format("hidden");
    const auto joiner_encoded    = _joiner->get_input_format("encoded");
    const auto joiner_prediction = _joiner->get_input_format("prediction");

    COMPUTE_FLOW_CHECK(cell.shape == hidden.shape);
    COMPUTE_FLOW_CHECK((joiner_encoded.dt == DataType::F32 &&
                       joiner_encoded.shape == std::vector<int64_t>{1, 1, _encoder_dimension}));
    COMPUTE_FLOW_CHECK(joiner_prediction.dt == DataType::F32 && joiner_prediction.shape == prediction.shape);

    const auto logits = _joiner->get_output_format("logits");

    COMPUTE_FLOW_CHECK(logits.dt == DataType::F32 && logits.shape.size() == 1 && logits.shape[0] > 0);
    COMPUTE_FLOW_CHECK(_blank_token_id >= 0 && _blank_token_id < logits.shape[0]);

    _frontend = std::make_unique<nemotron_asr::LogMel>(
        nemotron_asr::LogMelConfig{_sample_rate, manifest.get_constant<int64_t>("window_samples"), _hop_samples,
                                  manifest.get_constant<int64_t>("fft_size"), _mel_bins,
                                  manifest.get_constant<float>("preemphasis"), manifest.get_constant<float>("log_guard")});

    _memory = MemoryPlanner{};
    _memory.allocate("mel", {DataType::F32, {1, _mel_bins, _input_frames}});
    _memory.allocate("overlap", {DataType::F32, {_mel_bins, _overlap_frames}});
    _memory.allocate("mel_length", {DataType::I64, {1}});
    _memory.allocate("token", {DataType::I64, {1, 1}});

    _memory.allocate("prompt_id", prompt);

    update_config_impl(manifest, profiler);
}

void NemotronASR::warmup_impl(ProfilerView profiler) {
    std::vector<float> silence(static_cast<size_t>(_fresh_frames * _hop_samples), 0.0f);
    TensorDict input;

    input.declare_tensor("raw_wav") = {{DataType::F32, {static_cast<int64_t>(silence.size())}}, silence.data(), true};
    input.declare_tensor("sample_rate") = {{DataType::I64, {1}}, &_sample_rate, true};
    run_impl(input, profiler);
}

void NemotronASR::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    const int64_t prompt_id = prompt_id_for_language(
        manifest.get_config<std::string>("language"), *manifest.get_asset("languages_map_file"));
    const int64_t max_symbols = manifest.get_config<int64_t>("max_symbols_per_frame");

    COMPUTE_FLOW_CHECK(prompt_id >= 0);
    COMPUTE_FLOW_CHECK(max_symbols > 0);

    _prompt_id             = prompt_id;
    _max_symbols_per_frame = max_symbols;
}

void NemotronASR::reset_states_impl(ProfilerView profiler) {
    (void)profiler;

    _encoder->reset_states();
    _predictor->reset_states();
    _joiner->reset_states();
    _detokenizer->reset_states();
    std::memset(_memory.get_tensor("overlap").data, 0, _memory.nbytes("overlap"));
    _tokens.clear();
    _emitted_text.clear();
    _output_text.clear();
    _frame_offset = 0;
    _num_samples  = 0;
    _started      = false;
    _complete     = false;
}

bool NemotronASR::is_generation_complete() const { return _complete; }

void NemotronASR::run_predictor(int64_t token) {
    *_memory.get_data_as<int64_t>("token") = token;

    _predictor->set_input("token", _memory.get_tensor("token"));
    _predictor->run();
}

void NemotronASR::decode_frames() {
    const auto encoded       = _encoder->get_output("encoded");
    const auto length        = _encoder->get_output("encoded_length");
    const int64_t num_frames = *static_cast<const int64_t*>(length.data);

    COMPUTE_FLOW_CHECK(num_frames >= 0 && num_frames <= encoded.format.shape[1]);

    for (int64_t frame = 0; frame < num_frames; ++frame) {
        // This view borrows encoder output. The encoder is not run again until
        // every joiner call consuming this chunk has completed.
        TensorData frame_view{
            {DataType::F32, {1, 1, _encoder_dimension}},
            static_cast<float*>(encoded.data) + frame * _encoder_dimension,
            true};

        for (int64_t symbol = 0; symbol < _max_symbols_per_frame; ++symbol) {
            _joiner->set_input("encoded", frame_view);
            _joiner->set_input("prediction", _predictor->get_output("prediction"));
            _joiner->run();

            const auto logits        = _joiner->get_output("logits");
            const auto* scores       = static_cast<const float*>(logits.data);
            const int64_t num_logits = logits.format.get_num_elems();

            COMPUTE_FLOW_CHECK(num_logits > 0 && scores != nullptr);

            const int64_t token = std::max_element(scores, scores + num_logits) - scores;

            if (token == _blank_token_id) {
                break;
            }

            COMPUTE_FLOW_CHECK(symbol + 1 < _max_symbols_per_frame && "Decoder symbol cap reached; reset before reuse");

            _tokens.push_back(token);
            run_predictor(token);
        }
    }
}

void NemotronASR::update_text() {
    TensorData tokens{{DataType::I64, {static_cast<int64_t>(_tokens.size())}}, _tokens.data(), true};

    _detokenizer->set_input("tokens", tokens);
    _detokenizer->run();

    const auto text = _detokenizer->get_output("text");
    const std::string decoded(static_cast<const char*>(text.data), static_cast<size_t>(text.format.get_nbytes()));

    COMPUTE_FLOW_CHECK(decoded.compare(0, _emitted_text.size(), _emitted_text) == 0);

    // Keep the trailing word pending: byte-fallback tokens can revise an
    // incomplete UTF-8 suffix, and stt.v1 consumers append every returned byte.
    const size_t last_space  = decoded.find_last_of(" \n\t");
    const size_t stable_size = _complete ? decoded.size() : last_space == std::string::npos ? 0 : last_space + 1;

    COMPUTE_FLOW_CHECK(stable_size >= _emitted_text.size());

    _output_text = decoded.substr(_emitted_text.size(), stable_size - _emitted_text.size());
    _emitted_text.assign(decoded, 0, stable_size);
}

TensorDict NemotronASR::run_impl(const TensorDict& input, ProfilerView profiler) {
    (void)profiler;

    COMPUTE_FLOW_CHECK(!_complete && "Reset the brick before starting another utterance");

    const auto& audio = input.at("raw_wav");
    const auto& rate  = input.at("sample_rate");

    COMPUTE_FLOW_CHECK(rate.format.dt == DataType::I64 && rate.format.shape == std::vector<int64_t>{1});
    COMPUTE_FLOW_CHECK(rate.data != nullptr && *static_cast<const int64_t*>(rate.data) == _sample_rate);
    COMPUTE_FLOW_CHECK(audio.format.dt == DataType::F32);
    COMPUTE_FLOW_CHECK(audio.format.shape.size() == 1 ||
                       (audio.format.shape.size() == 2 && audio.format.shape[0] == 1));
    COMPUTE_FLOW_CHECK(audio.format.shape.back() >= 0);

    const int64_t num_samples = audio.format.shape.back();
    const auto* waveform      = static_cast<const float*>(audio.data);

    COMPUTE_FLOW_CHECK(num_samples == 0 || waveform != nullptr);

    if (!_started) {
        COMPUTE_FLOW_CHECK(num_samples == 0 || std::all_of(waveform, waveform + num_samples,
                                                           [](float sample) { return std::isfinite(sample); }));

        _num_samples = num_samples;
        _started     = true;

        if (num_samples >= _hop_samples) {
            run_predictor(_blank_token_id);
        }
    }

    COMPUTE_FLOW_CHECK(num_samples == _num_samples);

    const int64_t remaining = num_samples / _hop_samples - _frame_offset;
    const int64_t fresh     = std::min(remaining, _fresh_frames);

    if (fresh > 0) {
        auto* mel     = _memory.get_data_as<float>("mel");
        auto* overlap = _memory.get_data_as<float>("overlap");

        // Reuse this buffer for every chunk. Clearing it prevents a short final
        // chunk from retaining previous MEL values in its unused padding.
        std::fill_n(mel, _mel_bins * _input_frames, 0.0f);
        _frontend->compute(waveform, num_samples, _frame_offset, fresh, mel + _overlap_frames, _input_frames);

        // Overlap columns are separated by the rest of each MEL row. TensorData
        // has no strides, so a contiguous pointer view cannot represent them.
        for (int64_t bin = 0; bin < _mel_bins; ++bin) {
            float* row   = mel + bin * _input_frames;
            float* cache = overlap + bin * _overlap_frames;

            std::copy_n(cache, _overlap_frames, row);
            // The valid prefix ends at overlap + fresh. Its last overlap
            // columns therefore start at fresh, including short final chunks.
            std::copy_n(row + fresh, _overlap_frames, cache);
        }

        *_memory.get_data_as<int64_t>("mel_length") = fresh + _overlap_frames;

        _encoder->set_input("mel", _memory.get_tensor("mel"));
        _encoder->set_input("mel_length", _memory.get_tensor("mel_length"));

        *_memory.get_data_as<int64_t>("prompt_id") = _prompt_id;

        _encoder->set_input("prompt_id", _memory.get_tensor("prompt_id"));

        _encoder->run();
        decode_frames();
        _frame_offset += fresh;
    }

    _complete = fresh == remaining;

    update_text();

    TensorDict output;

    output.declare_tensor("text") = {
        {DataType::String, {static_cast<int64_t>(_output_text.size())}}, _output_text.data(), false};
    return output;
}

}  // namespace compute_flow::bricks::stt
