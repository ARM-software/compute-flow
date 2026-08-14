//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef COMPUTE_FLOW_HAS_LLAMACPP

#include "compute_flow/lm_runners/lm_runner.h"

#include "compute_flow/accelerator.h"
#include "compute_flow/profiler/profiler_view.h"
#include "compute_flow/tensor_dict.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <llama.h>

namespace compute_flow {

/**
 * Runner wrapper for the LM module (llama.cpp backend).
 */
class LlamaCppLMRunner : public ILMRunner {
public:
    LlamaCppLMRunner() = default;

    // IRunner
    void setup(const ModelData& model, ProfilerView profiler = {}) override;
    const std::string& model_name() const override;
    void map_input(std::string_view key, size_t idx) override;
    void map_input(std::string_view key, std::string_view tensor_name) override;
    void map_output(std::string_view key, size_t idx) override;
    void map_output(std::string_view key, std::string_view tensor_name) override;
    void set_input(const std::string& key, const TensorData& tensor) override;
    TensorData get_output(const std::string& key) override;
    TensorFormat get_input_format(const std::string& key) override;
    TensorFormat get_output_format(const std::string& key) override;
    void reset_states() override;
    void run() override;

    // ILMRunner
    TensorFormat get_preprocessing_format(const TensorData& input_tensor) override;
    TensorFormat get_postprocessing_format(const TensorData& input_tensor) override;
    // The following method is exposed because the user may need to concatenate the token IDs.
    // Additionally, in some cases, we may want to feed the embeddings directly into the language model
    // (as done in certain TTS pipelines).
    void preprocessing(const TensorData& input_tensor, TensorData& output_tensor) override;
    // The following method is exposed because the user may need access to the token IDs instead of the decoded output modality.
    // For example, in certain audio generation pipelines, the output IDs are required rather than the output string,
    // since those IDs serve as input to another model that generates the raw modality.
    void postprocessing(const TensorData& input_tensor, TensorData& output_tensor) override;
    void save_sequence_state(const std::string& name) override;
    void restore_sequence_state(const std::string& name) override;
    bool is_end_of_generation() const override;

private:

    static bool profile_op(ggml_tensor* tensor, bool ask, void* user_data);

    TensorFormat get_tokenize_ids_format(const TensorData& text);
    TensorFormat get_detokenize_format(const TensorData& token_ids);
    void tokenize_ids(const TensorData& text, TensorData& ids);
    void detokenize(const TensorData& token_ids, TensorData& text);

    struct LlamaModelDeleter {
        void operator()(llama_model* m) const noexcept;
    };
    struct LlamaContextDeleter {
        void operator()(llama_context* c) const noexcept;
    };
    struct LlamaSamplerDeleter {
        void operator()(llama_sampler* s) const noexcept;
    };

    ProfilerView _profiler;

    std::string                                         _model_name;
    std::vector<TensorData>                             _inputs;
    std::vector<TensorData>                             _outputs;
    std::unordered_map<std::string, size_t>             _in_tensor_idx;
    std::unordered_map<std::string, size_t>             _out_tensor_idx;
    std::unique_ptr<llama_model, LlamaModelDeleter>     _model;
    std::unique_ptr<llama_context, LlamaContextDeleter> _ctx;
    std::unique_ptr<llama_sampler, LlamaSamplerDeleter> _sampler;
    std::unordered_map<std::string, std::vector<uint8_t>> _saved_states;

    bool _is_first_run;
    bool _is_eog;
    // Tracks whether the caller set new input tensors since the last run.
    bool _has_new_input { false };
    bool _is_sampler_enabled { false };
    bool _append_generated_token { true };
    int32_t _embedding_position_count { 1 };
    int32_t _min_generated_tokens { 0 };
    int32_t _max_tokens_to_gen { -1 };
    int32_t _generated_tokens { 0 };
    int64_t _op_start_time_ns { 0 };

    llama_token            _last_generated_token_id;
    std::vector<float>     _last_hidden_state;
    std::vector<float>     _last_hidden_state_seq; // full-sequence hidden states [num_tokens × n_embd]
    std::vector<float>     _last_pooled_embedding;
    std::vector<float>     _last_logits;
    int32_t                _logits_vocab_size{0};
};

}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_LLAMACPP
