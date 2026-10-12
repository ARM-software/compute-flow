// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sentencepiece_processor.h>

#include <cstdint>
#include <string>
#include <vector>

#include "compute_flow/runners/runner.h"

namespace compute_flow {

/**
 * @brief Decode complete token histories without exposing a tokenizer library to bricks.
 *
 * set_input() copies IDs. Output bytes remain valid until run(), reset_states(),
 * setup(), or destruction. Decoding is stateless, including an empty history.
 * One instance is synchronous and not safe for concurrent calls.
 */
class SentencePieceDecodeRunner final : public IRunner {
public:
    /**
     * @brief Load a SentencePiece .model and clear any previous history and output.
     * @param model Exactly one named input and output, each mapped to index zero;
     * recurrent states are unsupported. Names come from the manifest.
     * @param profiler Non-owning profiler view used to measure decode calls.
     * @throws std::runtime_error If the model cannot load or the mapping is invalid.
     */
    void setup(const ModelData& model, ProfilerView profiler = {}) override;
    /** @brief Return the name supplied at setup, used for profiling. */
    const std::string& model_name() const override;
    /** @brief Bind the sole input once; only index zero is accepted. */
    void map_input(std::string_view key, size_t index) override;
    /** @brief Bind the sole input; tokenizer files have no tensor names, so tensor_name is ignored. */
    void map_input(std::string_view key, std::string_view tensor_name) override;
    /** @brief Bind the sole output once; only index zero is accepted. */
    void map_output(std::string_view key, size_t index) override;
    /** @brief Bind the sole output; tokenizer files have no tensor names, so tensor_name is ignored. */
    void map_output(std::string_view key, std::string_view tensor_name) override;
    /**
     * @brief Copy a complete int64 [N] history; N may be zero with a null data pointer.
     * Every ID must belong to the loaded vocabulary. Invalid tensors raise
     * std::runtime_error; reset or supply a valid history before running again.
     */
    void set_input(const std::string& key, const TensorData& tensor) override;
    /** @brief Borrow the last decoded UTF-8 bytes with the lifetime documented on this class. */
    TensorData get_output(const std::string& key) override;
    /** @brief Return int64 [-1], where -1 denotes the variable token count. */
    TensorFormat get_input_format(const std::string& key) override;
    /** @brief Return string [N], where N is the current output byte count, not character count. */
    TensorFormat get_output_format(const std::string& key) override;
    /** @brief Clear tokens and text without unloading the tokenizer. */
    void reset_states() override;
    /** @brief Replace the output by decoding the current complete history, including empty input. */
    void run() override;

private:
    sentencepiece::SentencePieceProcessor _tokenizer;
    ProfilerView _profiler;
    std::string _model_name;
    std::string _input_key;
    std::string _output_key;
    std::vector<int32_t> _tokens;
    std::string _text;
};

}  // namespace compute_flow
