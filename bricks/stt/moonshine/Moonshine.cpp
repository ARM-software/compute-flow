//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "Moonshine.h"

#include <cstring>
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <variant>

namespace compute_flow::bricks::stt {

constexpr int32_t k_brick_major_version = 1;

static void append_txt_piece(const std::string &piece, std::string &target) {
    for (size_t i = 0; i < piece.size();) {
        const unsigned char c = static_cast<unsigned char>(piece[i]);
        if (c == 0xE2 && i + 2 < piece.size()) { // possible '▁'
            const unsigned char c1 = static_cast<unsigned char>(piece[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(piece[i + 2]);
            if (c1 == 0x96 && c2 == 0x81) {
                target.push_back(' ');
                i += 3;
                continue;
            }
        }
        target.push_back(static_cast<char>(piece[i]));
        ++i;
    }
}

static void trim_txt(std::string &text) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    auto begin = std::find_if(text.begin(), text.end(), [&](char ch) {
        return not_space(static_cast<unsigned char>(ch));
    });

    auto end = std::find_if(text.rbegin(), text.rend(), [&](char ch) {
        return not_space(static_cast<unsigned char>(ch));
    }).base();

    if (begin >= end) {
        text.clear();
    } else {
        text.assign(begin, end);
    }
}

static void load_tokenizer_file_from_file(const std::string &path, std::vector<std::string>& tokens_vocab) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open tokenizer file: " + path);
    }

    std::vector<char> buffer((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    size_t idx = 0;
    tokens_vocab.clear();
    while (idx < buffer.size()) {
        const uint8_t length = static_cast<uint8_t>(buffer[idx]);
        ++idx;
        if (idx + length > buffer.size()) {
            throw std::runtime_error("Tokenizer file is truncated");
        }
        tokens_vocab.emplace_back(buffer.data() + idx, length);
        idx += length;
    }
}

ValidationResult Moonshine::validate(const ComputeFlowManifest& manifest) const {
    const BrickInfo& info = manifest.get_brick_info();
    COMPUTE_FLOW_VALIDATE(info.type == "stt");
    COMPUTE_FLOW_VALIDATE(info.name == "Moonshine");
    COMPUTE_FLOW_VALIDATE(info.version.major == k_brick_major_version);

    COMPUTE_FLOW_VALIDATE(manifest.num_modules() == 2);

    COMPUTE_FLOW_VALIDATE(manifest.has_module("encoder"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("decoder"));

    // Validate presence of the model path
    COMPUTE_FLOW_VALIDATE(manifest["encoder"].model_path != "");
    COMPUTE_FLOW_VALIDATE(manifest["decoder"].model_path != "");

    // Validate the target framework.
    COMPUTE_FLOW_VALIDATE(manifest["encoder"].framework_type == RunnerType::ONNXRT);
    COMPUTE_FLOW_VALIDATE(manifest["decoder"].framework_type == RunnerType::ONNXRT);

    // Check whether the models only have 1 accelerator type
    // At the moment, only the CPU is supported
    COMPUTE_FLOW_VALIDATE(manifest["encoder"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["decoder"].acc_config.priorities.size() == 1);

    // Only CPU is supported as accelerator type
    COMPUTE_FLOW_VALIDATE(manifest["encoder"].acc_config.has_accelerator(AcceleratorType::Cpu));
    COMPUTE_FLOW_VALIDATE(manifest["decoder"].acc_config.has_accelerator(AcceleratorType::Cpu));

    // Validate presence of required metadata
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("max_tokens"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("token_rate"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("force_use_kv_cache"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("bos_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("eos_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("vocab_size"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("sample_rate"));

    // Validate presence of required asset
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("tokenizer_file"));
    return {};
}

void Moonshine::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)manifest;
    (void)profiler;
    // TODO: Implement configuration updates in a separate MR.
}

void Moonshine::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    RunnerType enc_runner_type = manifest["encoder"].framework_type;
    RunnerType dec_runner_type = manifest["decoder"].framework_type;

    _runner["encoder"] = make_runner(enc_runner_type);
    _runner["decoder"] = make_runner(dec_runner_type);

    _runner["encoder"]->setup(manifest["encoder"], profiler);
    _runner["decoder"]->setup(manifest["decoder"], profiler);
}

void Moonshine::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    _max_tokens         = manifest.get_constant<int32_t>("max_tokens");
    _token_rate         = manifest.get_constant<int32_t>("token_rate");
    _sample_rate        = manifest.get_constant<int32_t>("sample_rate");
    _force_use_kv_cache = manifest.get_constant<bool>("force_use_kv_cache");
    _bos_token_id       = manifest.get_constant<int64_t>("bos_token_id");
    _eos_token_id       = manifest.get_constant<int64_t>("eos_token_id");
    _vocab_size         = manifest.get_constant<int64_t>("vocab_size");

    const std::string tokenizer_file = *manifest.get_asset("tokenizer_file");
    load_tokenizer_file_from_file(tokenizer_file, _tokens_vocab);

    TensorFormat ids_format = _runner["decoder"]->get_input_format("ids");
    ids_format.shape[0] = 1;
    ids_format.shape[1] = _max_tokens;

    TensorFormat use_cache_format = _runner["decoder"]->get_input_format("use_cache_branch");

    _internal_memory.allocate("ids", ids_format);
    _internal_memory.allocate("use_cache", use_cache_format);
}

void Moonshine::warmup_impl(ProfilerView profiler) {
    TensorDict input;

    // Set the input
    input.declare_tensor("raw_wav");
    input.declare_tensor("sample_rate");

    const int64_t sample_rate = 16000;

    std::vector<float> mono_raw_audio(16000 * 5);

    // Text prompt - shape[number of chars, 1]
    input["raw_wav"].data         = (void *)mono_raw_audio.data();
    input["raw_wav"].format.dt    = DataType::F32;
    input["raw_wav"].format.shape = {1, (int64_t)mono_raw_audio.size()};
    input["sample_rate"].data         = (void *)&sample_rate;
    input["sample_rate"].format.dt    = DataType::I64;
    input["sample_rate"].format.shape = {1};

    // Run the model inference. We can ignore the output
    run_impl(input, profiler);
}

void Moonshine::reset_states_impl(ProfilerView profiler) {
    _runner["decoder"]->reset_states();
    _is_output_ready = false;
}

bool Moonshine::is_generation_complete() const {
    return _is_output_ready;
}

TensorDict Moonshine::run_impl(const TensorDict& input, ProfilerView profiler) {
    _is_output_ready = true;
    // Validate number of input
    // Validate type of input
    // Validate shape

    COMPUTE_FLOW_ASSERT(input["raw_wav"].format.shape[0] == 1);
    COMPUTE_FLOW_ASSERT(((int64_t*)(input["sample_rate"].data))[0] == _sample_rate);

    // TODO: Add resempler if the input sample rate is not what we expect

    _runner["encoder"]->set_input("raw_wav", input["raw_wav"]);
    _runner["encoder"]->run();
    auto t_enc_out = _runner["encoder"]->get_output("encoder_hidden_states");

    // Get the ids and use_cache tensors from the memory planner
    auto t_ids_in       = _internal_memory.get_tensor("ids");
    auto t_use_cache_in = _internal_memory.get_tensor("use_cache");

    // Reset IDs
    memset(t_ids_in.data, 0, t_ids_in.format.get_nbytes());
    memset(t_use_cache_in.data, 0, t_use_cache_in.format.get_nbytes());

    // Initialize IDs with the BOS token id.
    *((int64_t*)t_ids_in.data) = _bos_token_id;

    // Since the encoder_hidden_states does not change in the decoder, we can initialize it
    // only once (TODO: Double check)
    _runner["decoder"]->set_input("encoder_hidden_states", t_enc_out);

    auto in_format = _runner["decoder"]->get_input_format("encoder_hidden_states");

    if(_force_use_kv_cache) {
        // When the KV cache is enabled, the IDS is a scalar containing just the last token id
        int64_t token_id = 0;
        for (; token_id < _max_tokens; ++token_id) {

            // Make sure the ids input is a scale when kv cache is enabled
            t_ids_in.format.shape[0] = 1;
            t_ids_in.format.shape[1] = 1;

            _runner["decoder"]->set_input("ids", t_ids_in);
            _runner["decoder"]->set_input("use_cache_branch", t_use_cache_in);
            _runner["decoder"]->run();
            auto t_logits = _runner["decoder"]->get_output("logits");

            // Check the expected vocab size
            COMPUTE_FLOW_ASSERT(t_logits.format.shape[2] == _vocab_size);
            COMPUTE_FLOW_ASSERT(t_ids_in.format.shape[1] == 1);
            COMPUTE_FLOW_ASSERT(t_logits.format.shape[1] == 1);

            // Move the IDs pointer to the next element
            t_ids_in.data = (void*)((int64_t*)t_ids_in.data + 1);

            custom_op_sampler_max_row(t_logits, t_ids_in);

            // The logits row consists of 32768 values (vocab size)
            // Given the token_id, we have to identify which token has the highest probability
            // The sampler is based on the max

            // Check for end of sequence
            if (*((int64_t*)t_ids_in.data) == _eos_token_id) {
                break;
            }

            // Start using the cache
            memset(t_use_cache_in.data, 1, t_use_cache_in.format.get_nbytes());
        }

        // Convert logits to human-readable text
        // Since we need all tokens, we need to move the ids data pointer to
        // the first element
        t_ids_in.format.shape[0] = 1;
        t_ids_in.format.shape[1] = token_id + 1;
        t_ids_in.data = (void*)((int64_t*)t_ids_in.data - token_id - 1);
    } else {
        // When the KV cache is not used, the IDS is a vector containing all the token ids
        for (int64_t token_id = 0; token_id < _max_tokens; ++token_id) {

            // Check for end of sequence
            if (*((int64_t*)t_ids_in.data + token_id) == _eos_token_id) {
                break;
            }

            // Make sure the ids input is a vector, with the length equal
            // to the number of tokens to be processed
            t_ids_in.format.shape[0] = 1;
            t_ids_in.format.shape[1] = token_id + 1;

            _runner["decoder"]->set_input("ids", t_ids_in);
            _runner["decoder"]->set_input("use_cache_branch", t_use_cache_in);

            _runner["decoder"]->run();

            auto t_logits = _runner["decoder"]->get_output("logits");

            // Check the expected vocab size
            COMPUTE_FLOW_ASSERT(t_logits.format.shape[2] == _vocab_size);
            COMPUTE_FLOW_ASSERT(t_logits.format.shape[1] == (token_id + 1));
            const size_t vocab_size = t_logits.format.shape[2];

            // Get the row of interest from the logits
            t_logits.format.shape[1] = 1;
            t_logits.data = (void*)((float*)t_logits.data + token_id * vocab_size);

            // Get the scalar of interest from the ids
            t_ids_in.format.shape[1] = 1;
            t_ids_in.data = (void*)((int64_t*)t_ids_in.data + token_id + 1);

            // The logits row consists of 32768 values (vocab size)
            // Given the token_id, we have to identify which token has the highest probability
            // The sampler is based on the max
            custom_op_sampler_max_row(t_logits, t_ids_in);

            t_ids_in.format.shape[1] = token_id + 1;
            t_ids_in.data = (void*)((int64_t*)t_ids_in.data - token_id - 1);
        }
    }

    TensorDict output;

    output.declare_tensor("text");

    // Convert logits to human-readable text
    custom_op_ids_to_text(t_ids_in, output["text"]);

    output["text"].data         = (void *)_output_text.data();
    output["text"].format.dt    = DataType::String;
    output["text"].format.shape = {1, (int64_t)_output_text.size()};

    return output;
}

void Moonshine::custom_op_sampler_max_row(const TensorData& logits_row, TensorData& ids) {
    // Check shape
    // Check datatype
    const int64_t vocab_size = logits_row.format.shape[2];

    const float* logits_ptr = (const float*)logits_row.data;
    int64_t* ids_ptr = (int64_t*)ids.data;

    int64_t max_index = 0;
    float max_value = std::numeric_limits<float>::lowest();
    for (int64_t vid = 0; vid < vocab_size; ++vid) {
        const float value = logits_ptr[vid];
        if (value > max_value) {
            max_value = value;
            max_index = vid;
        }
    }

    ids_ptr[0] = max_index;
}

void Moonshine::custom_op_ids_to_text(const TensorData& ids, TensorData& text) {

    const int64_t num_token_ids = ids.format.shape[1];

    _output_text.clear();
    _output_text.reserve(num_token_ids * 4);

    const int64_t* ids_ptr = (const int64_t*)ids.data;

    for (int64_t i = 0; i < num_token_ids; ++i) {
        const int64_t id = ids_ptr[i];
        if (id < 0 || static_cast<int64_t>(id) >= _tokens_vocab.size()) {
            continue;
        }
        if (id == _bos_token_id) { // <s>/BOS
            continue;
        }
        if (id == _eos_token_id) { // </s>/EOS
            break;
        }
        const std::string &piece = _tokens_vocab[id];
        if (!piece.empty() && piece[0] == '<') {
            if (piece == "<unk>") {
                continue;
            }
            if (piece.size() > 2 && piece[1] == '<') {
                continue;
            }
        }

        append_txt_piece(piece, _output_text);
    }

    trim_txt(_output_text);
}

}  // namespace compute_flow::bricks::stt
