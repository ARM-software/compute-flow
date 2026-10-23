//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "NeuDecide.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <unordered_set>

namespace compute_flow::bricks::audio_tool_calling {
namespace {

using Json = nlohmann::ordered_json;

const std::vector<std::string> k_models = {"audio_encoder", "tool_encoder", "decoder_start", "decoder_step",
                                           "tokenizer"};

std::string tools_text(const Json& tools) {
    const std::string compact = tools.dump(-1, ' ', true);
    std::string spaced;
    bool quoted  = false;
    bool escaped = false;

    for (const char character : compact) {
        spaced.push_back(character);

        if (escaped) {
            escaped = false;
        } else if (quoted && character == '\\') {
            escaped = true;
        } else if (character == '"') {
            quoted = !quoted;
        } else if (!quoted && (character == ',' || character == ':')) {
            spaced.push_back(' ');
        }
    }

    return spaced;
}

TensorData tensor(void* data, DataType type, const std::vector<int64_t>& shape) { return {{type, shape}, data, true}; }

}  // namespace

ValidationResult NeuDecide::validate(const ComputeFlowManifest& manifest) const {
    const auto& info = manifest.get_brick_info();

    COMPUTE_FLOW_VALIDATE(info.type == "audio_tool_calling");
    COMPUTE_FLOW_VALIDATE(info.name == "NeuDecide");
    COMPUTE_FLOW_VALIDATE(info.version.major == 1);
    COMPUTE_FLOW_VALIDATE(manifest.num_modules() == k_models.size());

    for (const auto& name : k_models) {
        COMPUTE_FLOW_VALIDATE(manifest.has_module(name));
        COMPUTE_FLOW_VALIDATE(!manifest[name].model_path.empty());
        COMPUTE_FLOW_VALIDATE(manifest[name].acc_config.priorities.size() == 1);
        COMPUTE_FLOW_VALIDATE(manifest[name].acc_config.has_accelerator(AcceleratorType::Cpu));
        COMPUTE_FLOW_VALIDATE(manifest[name].framework_type ==
                              (name == "tokenizer" ? RunnerType::SentencePieceEncode : RunnerType::ExecuTorch));
    }

    COMPUTE_FLOW_VALIDATE(manifest["audio_encoder"].has_input("waveform"));
    COMPUTE_FLOW_VALIDATE(manifest["audio_encoder"].has_input("waveform_len"));
    COMPUTE_FLOW_VALIDATE(manifest["audio_encoder"].has_output("audio_embeds"));
    COMPUTE_FLOW_VALIDATE(manifest["audio_encoder"].has_output("audio_len"));
    COMPUTE_FLOW_VALIDATE(manifest["tool_encoder"].has_input("audio_embeds"));
    COMPUTE_FLOW_VALIDATE(manifest["tool_encoder"].has_input("audio_len"));
    COMPUTE_FLOW_VALIDATE(manifest["tool_encoder"].has_input("tools_ids"));

    for (const auto& name : {"cross_key", "cross_value", "encoder_mask"}) {
        COMPUTE_FLOW_VALIDATE(manifest["tool_encoder"].has_output(name));
        COMPUTE_FLOW_VALIDATE(manifest["decoder_start"].has_input(name));
        COMPUTE_FLOW_VALIDATE(manifest["decoder_step"].has_input(name));
    }

    for (const auto& name : {"decoder_start", "decoder_step"}) {
        COMPUTE_FLOW_VALIDATE(manifest[name].has_input("input_ids"));
        COMPUTE_FLOW_VALIDATE(manifest[name].has_output("logits"));
    }

    for (const auto& name : {"past_key", "past_value"}) {
        COMPUTE_FLOW_VALIDATE(manifest["decoder_start"].has_output(name));
        COMPUTE_FLOW_VALIDATE(manifest["decoder_step"].has_state(name));
    }

    COMPUTE_FLOW_VALIDATE(manifest["tokenizer"].has_input("text"));
    COMPUTE_FLOW_VALIDATE(manifest["tokenizer"].has_output("token_ids"));
    for (const auto& name : {"token_bytes", "tools"}) {
        COMPUTE_FLOW_VALIDATE(manifest.has_asset(name));
        COMPUTE_FLOW_VALIDATE(!manifest.get_asset(name)->empty());
    }

    for (const auto& name : {"sample_rate", "max_audio_samples", "max_tools_tokens", "max_answer_tokens", "vocab_size",
                             "eos_token_id", "tool_call_token_id", "tools_token_id"}) {
        COMPUTE_FLOW_VALIDATE(manifest.has_constant(name));
    }

    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("sample_rate") == 16000);
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("max_audio_samples") > 0);
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("max_audio_samples") <= 480000);
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("max_tools_tokens") > 1);
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("max_tools_tokens") <= 1536);
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("max_answer_tokens") > 1);
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("max_answer_tokens") <= 128);
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("vocab_size") == 8192);
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("eos_token_id") == 1);
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("tool_call_token_id") == 4);
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("tools_token_id") == 5);
    COMPUTE_FLOW_VALIDATE(manifest.has_config("max_output_tokens"));

    return {};
}

void NeuDecide::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    _runners.clear();

    for (const auto& name : k_models) {
        auto runner = make_runner(manifest[name].framework_type);

        runner->setup(manifest[name], profiler);

        _runners.emplace(name, std::move(runner));
    }
}

void NeuDecide::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    _sample_rate       = manifest.get_constant<int64_t>("sample_rate");
    _max_audio_samples = manifest.get_constant<int64_t>("max_audio_samples");
    _max_answer_tokens = manifest.get_constant<int64_t>("max_answer_tokens");
    _vocab_size        = manifest.get_constant<int64_t>("vocab_size");
    _eos_token         = manifest.get_constant<int64_t>("eos_token_id");
    _tool_call_token   = manifest.get_constant<int64_t>("tool_call_token_id");

    // The asset maps token IDs to hex-encoded bytes. Decode and cache them for tool-call grammar checks.
    std::ifstream stream(*manifest.get_asset("token_bytes"));

    COMPUTE_FLOW_CHECK(stream.good());

    const auto vocabulary = Json::parse(stream, nullptr, false);

    COMPUTE_FLOW_CHECK(vocabulary.is_array());
    COMPUTE_FLOW_CHECK(vocabulary.size() == static_cast<uint64_t>(_vocab_size));
    _token_bytes.clear();

    for (const auto& entry : vocabulary) {
        COMPUTE_FLOW_CHECK(entry.is_string());

        const auto& hex = entry.get_ref<const std::string&>();

        COMPUTE_FLOW_CHECK(hex.size() % 2 == 0);

        std::string bytes;

        for (uint64_t index = 0; index < hex.size(); index += 2) {
            const auto high = std::string_view("0123456789abcdef").find(hex[index]);
            const auto low  = std::string_view("0123456789abcdef").find(hex[index + 1]);

            COMPUTE_FLOW_CHECK(high != std::string_view::npos && low != std::string_view::npos);

            bytes.push_back(static_cast<char>(high * 16 + low));
        }

        _token_bytes.push_back(std::move(bytes));
    }

    _memory.allocate("input_ids", {DataType::I64, {1, 1}});
    _candidates.resize(_vocab_size);

    load_tools(manifest);
    update_config_impl(manifest, profiler);
}

void NeuDecide::warmup_impl(ProfilerView profiler) {
    float sample        = 0;
    int64_t sample_rate = _sample_rate;
    TensorDict input;
    input.declare_tensor("raw_wav")     = tensor(&sample, DataType::F32, {1});
    input.declare_tensor("sample_rate") = tensor(&sample_rate, DataType::I64, {1});

    // A small request limit must not truncate the dummy response during setup.
    const int64_t max_output_tokens = _max_output_tokens;
    _max_output_tokens              = _max_answer_tokens;

    run_impl(input, profiler);

    _max_output_tokens = max_output_tokens;
}

void NeuDecide::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    const int64_t max_output_tokens = manifest.get_config<int64_t>("max_output_tokens");

    COMPUTE_FLOW_CHECK(max_output_tokens > 1 && max_output_tokens <= _max_answer_tokens);

    _max_output_tokens = max_output_tokens;
}

void NeuDecide::reset_states_impl(ProfilerView profiler) {
    (void)profiler;

    for (auto& [name, runner] : _runners) {
        (void)name;

        runner->reset_states();
    }

    _text.clear();
    _complete = false;
}

bool NeuDecide::is_generation_complete() const { return _complete; }

TensorDict NeuDecide::run_impl(const TensorDict& input, ProfilerView profiler) {
    (void)profiler;

    if (_complete) {
        return output();
    }

    encode_request(input);

    const auto& ids = _memory.get_tensor("input_ids");

    _runners["decoder_start"]->set_input("input_ids", ids);
    _runners["decoder_start"]->run();
    _runners["decoder_step"]->set_input("past_key", _runners["decoder_start"]->get_output("past_key"));
    _runners["decoder_step"]->set_input("past_value", _runners["decoder_start"]->get_output("past_value"));

    // The upstream grammar forces this token, so it has no confidence contribution.
    *static_cast<int64_t*>(ids.data) = _tool_call_token;
    neu_decide::PrefixResult match;

    for (int64_t num_tokens = 1; num_tokens < _max_output_tokens; ++num_tokens) {
        _runners["decoder_step"]->set_input("input_ids", ids);
        _runners["decoder_step"]->run();

        const auto logits = _runners["decoder_step"]->get_output("logits");

        COMPUTE_FLOW_CHECK(logits.format.dt == DataType::F32 && logits.format.get_num_elems() == _vocab_size);
        COMPUTE_FLOW_CHECK(logits.data != nullptr);

        const auto* scores = static_cast<const float*>(logits.data);

        for (int64_t index = 0; index < _vocab_size; ++index) {
            COMPUTE_FLOW_CHECK(std::isfinite(scores[index]));
        }

        std::iota(_candidates.begin(), _candidates.end(), 0);
        const auto lower_score = [scores](int64_t left, int64_t right) {
            return scores[left] < scores[right] || (scores[left] == scores[right] && left > right);
        };

        // Test the highest-scoring tokens without sorting the entire vocabulary at every step.
        std::make_heap(_candidates.begin(), _candidates.end(), lower_score);

        bool selected  = false;
        auto remaining = _candidates.end();

        while (remaining != _candidates.begin()) {
            const auto token = _candidates.front();

            std::pop_heap(_candidates.begin(), remaining, lower_score);

            --remaining;

            if (_token_bytes[token].empty()) {
                continue;
            }

            const auto begin = _text.size();

            _text.append(_token_bytes[token]);

            auto candidate_match = neu_decide::match_tool_calls(_text, _tools);

            if (!candidate_match.valid) {
                // Roll back a rejected token while retaining storage for the next candidate.
                _text.resize(begin);

                continue;
            }

            match                            = std::move(candidate_match);
            *static_cast<int64_t*>(ids.data) = token;
            selected                         = true;
            break;
        }

        COMPUTE_FLOW_CHECK(selected);

        if (match.complete) {
            break;
        }
    }

    COMPUTE_FLOW_CHECK(match.valid);

    if (match.complete) {
        const auto calls = Json::parse(_text, nullptr, false);

        COMPUTE_FLOW_CHECK(calls.is_array());

        for (const auto& call : calls) {
            COMPUTE_FLOW_CHECK(call.is_object() && call.size() == 2 && call.contains("name") &&
                               call["name"].is_string() && call.contains("arguments") && call["arguments"].is_object());
        }
    } else {
        // Never expose a partial tool call when decoding reaches the token limit.
        _text = "[]";
    }

    _complete = true;
    return output();
}

void NeuDecide::load_tools(const ComputeFlowManifest& manifest) {
    std::ifstream stream(*manifest.get_asset("tools"));

    COMPUTE_FLOW_CHECK(stream.good());

    const auto tools = Json::parse(stream, nullptr, false);

    COMPUTE_FLOW_CHECK(tools.is_array());

    _tools.clear();
    std::unordered_set<std::string> names;

    for (const auto& tool : tools) {
        COMPUTE_FLOW_CHECK(tool.is_object());
        COMPUTE_FLOW_CHECK(tool.contains("name") && tool["name"].is_string());
        COMPUTE_FLOW_CHECK(tool.contains("description") && tool["description"].is_string());
        COMPUTE_FLOW_CHECK(tool.contains("parameters") && tool["parameters"].is_object());
        COMPUTE_FLOW_CHECK(tool["parameters"].value("type", std::string{}) == "object");

        neu_decide::Tool definition;
        definition.name = tool["name"].get<std::string>();

        COMPUTE_FLOW_CHECK(!definition.name.empty() && definition.name != "no_tool" &&
                           definition.name.find('\0') == std::string::npos && names.insert(definition.name).second);

        if (tool["parameters"].contains("properties")) {
            COMPUTE_FLOW_CHECK(tool["parameters"]["properties"].is_object());

            for (const auto& [name, schema] : tool["parameters"]["properties"].items()) {
                (void)schema;
                definition.arguments.push_back(name);
            }
        }

        _tools.push_back(std::move(definition));
    }

    std::string text = tools_text(tools);

    _runners["tokenizer"]->set_input("text",
                                     tensor(text.data(), DataType::String, {static_cast<int64_t>(text.size())}));
    _runners["tokenizer"]->run();

    const auto encoded = _runners["tokenizer"]->get_output("token_ids");
    COMPUTE_FLOW_CHECK(encoded.format.dt == DataType::I32 && encoded.data != nullptr);

    const int64_t num_tools_tokens = encoded.format.get_num_elems() + 1;
    const int64_t max_tools_tokens = manifest.get_constant<int64_t>("max_tools_tokens");

    // Package assets must fit in full; truncation would hide some tools from the model.
    COMPUTE_FLOW_CHECK(num_tools_tokens <= max_tools_tokens);

    _memory.allocate("tools_ids", {DataType::I64, {1, num_tools_tokens}});

    auto* tokens = _memory.get_data_as<int64_t>("tools_ids");
    tokens[0]    = manifest.get_constant<int64_t>("tools_token_id");

    for (int64_t index = 1; index < num_tools_tokens; ++index) {
        tokens[index] = static_cast<const int32_t*>(encoded.data)[index - 1];
    }
}

void NeuDecide::encode_request(const TensorDict& input) {
    COMPUTE_FLOW_CHECK(input.size() == 2);

    const auto& audio = input.at("raw_wav");
    const auto& rate  = input.at("sample_rate");

    COMPUTE_FLOW_CHECK(audio.format.dt == DataType::F32);
    COMPUTE_FLOW_CHECK(audio.format.shape.size() == 1 && audio.format.shape[0] > 0);
    COMPUTE_FLOW_CHECK(audio.data != nullptr);
    COMPUTE_FLOW_CHECK(rate.format.dt == DataType::I64 && rate.format.shape == std::vector<int64_t>{1});
    COMPUTE_FLOW_CHECK(rate.data != nullptr && *static_cast<const int64_t*>(rate.data) == _sample_rate);

    const auto* samples = static_cast<const float*>(audio.data);

    for (int64_t index = 0; index < audio.format.shape[0]; ++index) {
        COMPUTE_FLOW_CHECK(std::isfinite(samples[index]));
    }

    int64_t num_samples = std::min(audio.format.shape[0], _max_audio_samples);

    // Both borrowed inputs remain alive until this synchronous CPU run completes.
    _runners["audio_encoder"]->set_input("waveform", tensor(audio.data, DataType::F32, {1, num_samples}));
    _runners["audio_encoder"]->set_input("waveform_len", tensor(&num_samples, DataType::I64, {1}));
    _runners["audio_encoder"]->run();

    _runners["tool_encoder"]->set_input("audio_embeds", _runners["audio_encoder"]->get_output("audio_embeds"));
    _runners["tool_encoder"]->set_input("audio_len", _runners["audio_encoder"]->get_output("audio_len"));
    _runners["tool_encoder"]->set_input("tools_ids", _memory.get_tensor("tools_ids"));
    _runners["tool_encoder"]->run();

    // The tool encoder keeps these outputs alive and unchanged throughout decoding.
    for (const auto& name : {"cross_key", "cross_value", "encoder_mask"}) {
        const auto encoded = _runners["tool_encoder"]->get_output(name);

        _runners["decoder_start"]->set_input(name, encoded);
        _runners["decoder_step"]->set_input(name, encoded);
    }

    *static_cast<int64_t*>(_memory.get_tensor("input_ids").data) = _eos_token;
}

TensorDict NeuDecide::output() {
    TensorDict result;

    result.declare_tensor("calls") = {{DataType::String, {static_cast<int64_t>(_text.size())}}, _text.data(), false};

    return result;
}

}  // namespace compute_flow::bricks::audio_tool_calling
