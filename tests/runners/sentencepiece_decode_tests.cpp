// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "compute_flow/check.h"
#include "compute_flow/runners/runner.h"

namespace {

using namespace compute_flow;

std::string decode_text(IRunner& runner) {
    runner.run();

    const auto output = runner.get_output("text");

    COMPUTE_FLOW_CHECK(output.format.dt == DataType::String);

    return {static_cast<const char*>(output.data), static_cast<size_t>(output.format.get_nbytes())};
}

void reject_input(IRunner& runner, const TensorData& input) {
    bool rejected = false;

    try {
        runner.set_input("tokens", input);
    } catch (const std::exception&) {
        rejected = true;
    }

    COMPUTE_FLOW_CHECK(rejected);
}

}  // namespace

int32_t main(int32_t argc, char** argv) {
    COMPUTE_FLOW_CHECK(argc == 2);

    ModelData model;
    model.model_name = "tokenizer";
    model.model_path = argv[1];
    model.set_input("tokens", 0).set_output("text", 0);
    auto decoder = make_runner(RunnerType::SentencePieceDecode);

    decoder->setup(model);

    ModelData encode_model;
    encode_model.model_path = argv[1];
    encode_model.set_input("text", 0).set_output("tokens", 0);
    auto encoder = make_runner(RunnerType::SentencePieceEncode);

    encoder->setup(encode_model);

    std::string text = "Hello world.";
    TensorData input{{DataType::String, {static_cast<int64_t>(text.size())}}, text.data(), true};

    encoder->set_input("text", input);
    encoder->run();

    const auto encoded      = encoder->get_output("tokens");
    const auto* encoded_ids = static_cast<const int32_t*>(encoded.data);
    std::vector<int64_t> ids(encoded_ids, encoded_ids + encoded.format.get_num_elems());
    TensorData tokens{{DataType::I64, {static_cast<int64_t>(ids.size())}}, ids.data(), true};

    decoder->set_input("tokens", tokens);
    std::fill(ids.begin(), ids.end(), -1);

    COMPUTE_FLOW_CHECK(decode_text(*decoder) == text);
    COMPUTE_FLOW_CHECK(decode_text(*decoder) == text);

    reject_input(*decoder, tokens);
    tokens.format.dt = DataType::I32;

    reject_input(*decoder, tokens);
    reject_input(*decoder, {{DataType::I64, {1}}, nullptr, true});
    reject_input(*decoder, {{DataType::I64, {-1}}, ids.data(), true});
    reject_input(*decoder, {{DataType::I64, {1, 1}}, ids.data(), true});

    decoder->reset_states();

    COMPUTE_FLOW_CHECK(decode_text(*decoder).empty());

    decoder->set_input("tokens", {{DataType::I64, {0}}, nullptr, true});

    COMPUTE_FLOW_CHECK(decode_text(*decoder).empty());

    decoder->setup(model);

    COMPUTE_FLOW_CHECK(decode_text(*decoder).empty());
}
