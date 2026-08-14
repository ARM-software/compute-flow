//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifdef COMPUTE_FLOW_HAS_HF_TOKENIZER

#include "runners/huggingface_tokenizer/huggingface_tokenizer_runner.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace compute_flow {

namespace {

std::string load_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("HuggingFaceTokenizerRunner: failed to open file: " + path);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    if (!file) {
        throw std::runtime_error("HuggingFaceTokenizerRunner: failed to read file: " + path);
    }

    return buffer.str();
}

std::shared_ptr<tokenizer::PreTrainedTokenizer> load_tokenizer(const std::string& model_path) {
    namespace fs = std::filesystem;

    if (model_path.empty()) {
        throw std::runtime_error("HuggingFaceTokenizerRunner: model_path is empty");
    }

    fs::path input_path(model_path);
    if (!fs::exists(input_path)) {
        throw std::runtime_error("HuggingFaceTokenizerRunner: model_path does not exist: " + model_path);
    }

    if (fs::is_directory(input_path)) {
        throw std::runtime_error("HuggingFaceTokenizerRunner: model_path cannot be a directory: " + model_path);
    } else {
        const fs::path base_dir = input_path.parent_path();
        printf("Base dir %s\n", base_dir.c_str());
        return tokenizer::AutoTokenizer::from_pretrained(base_dir.string());
    }
}

}  // namespace

void HuggingFaceTokenizerRunner::setup(const ModelData& model, ProfilerView profiler) {
    _model_name = model.model_name;
    _profiler   = profiler;

    _tokenizer = load_tokenizer(model.model_path);
    if (!_tokenizer) {
        throw std::runtime_error("HuggingFaceTokenizerRunner: failed to initialize tokenizer");
    }

    _inputs.resize(1);
    _outputs.resize(1);

    for (const auto& [name_input, idx_input] : model.inputs) {
        map_input(name_input, idx_input);
    }

    for (const auto& [name_output, idx_output] : model.outputs) {
        map_output(name_output, idx_output);
    }
}

void HuggingFaceTokenizerRunner::run() {
    auto model_scope = _profiler.measure_model("hf_tokenizer", _model_name);

    if (!_tokenizer) {
        throw std::runtime_error("HuggingFaceTokenizerRunner: tokenizer not initialized");
    }

    _out = _tokenizer->encode(_in);

}

const std::string& HuggingFaceTokenizerRunner::model_name() const {
    return _model_name;
}

void HuggingFaceTokenizerRunner::map_input(std::string_view key, size_t idx) {
    if (idx != 0) {
        throw std::invalid_argument("HuggingFaceTokenizerRunner: input idx must be 0");
    }
    std::string k(key);
    if (_in_tensor_idx.count(k)) {
        throw std::invalid_argument("Input key already exists: " + k);
    }

    _in_tensor_idx.insert({k, idx});

    auto& tensor = _inputs[idx];
    tensor.data = nullptr;
    tensor.format.dt = DataType::String;
    tensor.format.shape = { -1 };
    tensor.user_hosted = true;
}

void HuggingFaceTokenizerRunner::map_input(std::string_view key, std::string_view /*tensor_name*/) {
    map_input(key, 0);
}

void HuggingFaceTokenizerRunner::map_output(std::string_view key, size_t idx) {
    if (idx != 0) {
        throw std::invalid_argument("HuggingFaceTokenizerRunner: output idx must be 0");
    }
    std::string k(key);
    if (_out_tensor_idx.count(k)) {
        throw std::invalid_argument("Output key already exists: " + k);
    }

    _out_tensor_idx.insert({k, idx});

    auto& tensor = _outputs[idx];
    tensor.data = _out.data();
    tensor.format.dt = DataType::I32;
    tensor.format.shape = { -1 };
    tensor.user_hosted = false;
}

void HuggingFaceTokenizerRunner::map_output(std::string_view key, std::string_view /*tensor_name*/) {
    map_output(key, 0);
}

void HuggingFaceTokenizerRunner::set_input(const std::string& key, const TensorData& tensor) {
    auto it = _in_tensor_idx.find(key);
    if (it == _in_tensor_idx.end()) {
        throw std::out_of_range("Input key not found: " + key);
    }

    if (tensor.data == nullptr) {
        throw std::runtime_error("HuggingFaceTokenizerRunner: input tensor has null data");
    }
    if (tensor.format.shape.empty()) {
        throw std::runtime_error("HuggingFaceTokenizerRunner: input tensor has empty shape");
    }

    // TODO: Currently batching is not supported
    const int64_t idx_val = tensor.format.shape.size() - 1;

    if (tensor.format.shape[idx_val] != tensor.format.get_num_elems()) {
        throw std::runtime_error("HuggingFaceTokenizerRunner: input tensor is not 1D");
    }

    _in.clear();
    _in.assign((const char*)tensor.data, tensor.format.shape[idx_val]);
}

TensorData HuggingFaceTokenizerRunner::get_output(const std::string& key) {
    auto it = _out_tensor_idx.find(key);
    if (it == _out_tensor_idx.end()) {
        throw std::out_of_range("Output key not found: " + key);
    }
    const size_t idx = it->second;

    _outputs[idx].data = _out.data();
    _outputs[idx].format.shape[0] = static_cast<int64_t>(_out.size());
    return _outputs[idx];
}

TensorFormat HuggingFaceTokenizerRunner::get_input_format(const std::string& key) {
    auto it = _in_tensor_idx.find(key);
    if (it == _in_tensor_idx.end()) {
        throw std::out_of_range("Input key not found: " + key);
    }
    const size_t idx = it->second;
    return _inputs[idx].format;
}

TensorFormat HuggingFaceTokenizerRunner::get_output_format(const std::string& key) {
    auto it = _out_tensor_idx.find(key);
    if (it == _out_tensor_idx.end()) {
        throw std::out_of_range("Output key not found: " + key);
    }
    const size_t idx = it->second;
    return _outputs[idx].format;
}

void HuggingFaceTokenizerRunner::reset_states() {
    _in.clear();
    _out.clear();
}

}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_HF_TOKENIZER
