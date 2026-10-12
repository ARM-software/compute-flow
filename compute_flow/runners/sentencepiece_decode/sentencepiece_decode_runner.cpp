// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

#include "runners/sentencepiece_decode/sentencepiece_decode_runner.h"

#include "compute_flow/check.h"

namespace compute_flow {

void SentencePieceDecodeRunner::setup(const ModelData& model, ProfilerView profiler) {
    COMPUTE_FLOW_CHECK(model.inputs.size() == 1);
    COMPUTE_FLOW_CHECK(model.outputs.size() == 1);
    COMPUTE_FLOW_CHECK(model.states.empty());
    COMPUTE_FLOW_CHECK(_tokenizer.Load(model.model_path).ok());

    _model_name = model.model_name;
    _profiler   = profiler;
    _input_key.clear();
    _output_key.clear();

    map_input(model.inputs.begin()->first, static_cast<size_t>(model.inputs.begin()->second));
    map_output(model.outputs.begin()->first, static_cast<size_t>(model.outputs.begin()->second));
    reset_states();
}

const std::string& SentencePieceDecodeRunner::model_name() const { return _model_name; }

void SentencePieceDecodeRunner::map_input(std::string_view key, size_t index) {
    COMPUTE_FLOW_CHECK(index == 0 && _input_key.empty() && !key.empty());

    _input_key = key;
}

void SentencePieceDecodeRunner::map_input(std::string_view key, std::string_view tensor_name) {
    (void)tensor_name;

    map_input(key, size_t{0});
}

void SentencePieceDecodeRunner::map_output(std::string_view key, size_t index) {
    COMPUTE_FLOW_CHECK(index == 0 && _output_key.empty() && !key.empty());

    _output_key = key;
}

void SentencePieceDecodeRunner::map_output(std::string_view key, std::string_view tensor_name) {
    (void)tensor_name;

    map_output(key, size_t{0});
}

void SentencePieceDecodeRunner::set_input(const std::string& key, const TensorData& tensor) {
    COMPUTE_FLOW_CHECK(!_input_key.empty() && key == _input_key);
    COMPUTE_FLOW_CHECK(tensor.format.dt == DataType::I64);
    COMPUTE_FLOW_CHECK(tensor.format.shape.size() == 1 && tensor.format.shape[0] >= 0);

    const int64_t num_tokens = tensor.format.shape[0];
    const auto* token_ids    = static_cast<const int64_t*>(tensor.data);

    COMPUTE_FLOW_CHECK(num_tokens == 0 || token_ids != nullptr);

    _tokens.clear();
    _tokens.reserve(static_cast<size_t>(num_tokens));

    for (int64_t index = 0; index < num_tokens; ++index) {
        COMPUTE_FLOW_CHECK(token_ids[index] >= 0 && token_ids[index] < _tokenizer.GetPieceSize());

        _tokens.push_back(static_cast<int32_t>(token_ids[index]));
    }
}

TensorData SentencePieceDecodeRunner::get_output(const std::string& key) {
    return {get_output_format(key), _text.data(), false};
}

TensorFormat SentencePieceDecodeRunner::get_input_format(const std::string& key) {
    COMPUTE_FLOW_CHECK(!_input_key.empty() && key == _input_key);

    return {DataType::I64, {-1}};
}

TensorFormat SentencePieceDecodeRunner::get_output_format(const std::string& key) {
    COMPUTE_FLOW_CHECK(!_output_key.empty() && key == _output_key);

    return {DataType::String, {static_cast<int64_t>(_text.size())}};
}

void SentencePieceDecodeRunner::reset_states() {
    _tokens.clear();
    _text.clear();
}

void SentencePieceDecodeRunner::run() {
    auto scope = _profiler.measure_model("sentencepiece_dec", _model_name);

    COMPUTE_FLOW_CHECK(_tokenizer.Decode(_tokens, &_text).ok());
}

}  // namespace compute_flow
