//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "LMTextGeneric.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

std::string tensor_to_string(const compute_flow::TensorData& t) {
    if (t.format.dt != compute_flow::DataType::String) {
        throw std::invalid_argument("LMTextGeneric: expected DataType::String for text tensor");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument("LMTextGeneric: text tensor has null data");
    }
    if (t.format.shape.empty()) {
        throw std::invalid_argument("LMTextGeneric: text tensor has empty shape");
    }

    int64_t nbytes = 0;
    if (t.format.shape.size() == 1) {
        nbytes = t.format.shape[0];
    } else if (t.format.shape.size() == 2) {
        nbytes = t.format.shape[1];
    } else {
        throw std::invalid_argument("LMTextGeneric: text tensor has unsupported shape rank");
    }

    if (nbytes < 0) {
        throw std::invalid_argument("LMTextGeneric: text tensor has negative size");
    }
    return std::string(static_cast<const char*>(t.data), static_cast<size_t>(nbytes));
}

std::string render_prompt_template(const std::string& templ, const std::string& user_text) {
    if (templ.empty()) {
        return user_text;
    }

    std::string out;
    out.reserve(templ.size() + user_text.size());

    size_t pos = 0;
    constexpr std::string_view k_marker = "<<USER_TEXT>>";
    while (pos < templ.size()) {
        const size_t start = templ.find(k_marker, pos);
        if (start == std::string::npos) {
            out.append(templ.substr(pos));
            break;
        }
        out.append(templ.substr(pos, start - pos));
        out.append(user_text);
        pos = start + k_marker.size();
    }

    return out;
}

}  // namespace

namespace compute_flow::bricks::lm_text {

constexpr int32_t k_brick_major_version = 1;

ValidationResult LMTextGeneric::validate(const ComputeFlowManifest& manifest) const {
    const BrickInfo& info = manifest.get_brick_info();
    COMPUTE_FLOW_VALIDATE(info.type == "lm_text");
    COMPUTE_FLOW_VALIDATE(info.name == "LMTextGeneric");
    COMPUTE_FLOW_VALIDATE(info.version.major == k_brick_major_version);

    COMPUTE_FLOW_VALIDATE(manifest.has_module("lm_core"));
    COMPUTE_FLOW_VALIDATE(manifest["lm_core"].model_path != "");
    COMPUTE_FLOW_VALIDATE(manifest["lm_core"].lm_runner_type == LMRunnerType::Llama_cpp);

    COMPUTE_FLOW_VALIDATE(manifest.has_config("prompt_template"));
    COMPUTE_FLOW_VALIDATE(manifest.has_config("system_prompt"));
    COMPUTE_FLOW_VALIDATE(manifest.has_config("assistant_prefix"));

    // TODO: We should check the presence of <<TEXT_USER>> in prompt_template
    return {};
}

void LMTextGeneric::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    // Resolves nested configs such as system_prompt and assistant_prefix.
    _prompt_template = manifest.get_config<std::string>("prompt_template");
}

void LMTextGeneric::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    LMRunnerType lm_runner_type = manifest["lm_core"].lm_runner_type;

    _runner["lm_core"] = make_lm_runner(lm_runner_type);
    _runner["lm_core"]->setup(manifest["lm_core"], profiler);
}

void LMTextGeneric::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    update_config_impl(manifest, profiler);

    // TODO: We should load it from the manifest.
    const int64_t max_context_length = 2048;
    const int64_t max_generated_token_ids = 16;

    allocate_memory_preprocessing(max_context_length);
    allocate_memory_postprocessing(max_generated_token_ids);
}

void LMTextGeneric::warmup_impl(ProfilerView profiler) {
    TensorDict input;

    // Set the input
    input.declare_tensor("text_prompt");

    std::string mock_input_str = "mock_input";

    TensorData mock_input;
    mock_input.data = (void*)mock_input_str.data();
    mock_input.format.dt = DataType::String;
    mock_input.format.shape = { (int64_t)mock_input_str.size() };

    input["text_prompt"] = mock_input;

    // Run the inference
    run_impl(input, profiler);
}

void LMTextGeneric::reset_states_impl(ProfilerView profiler) {
    if (_runner.count("lm_core") != 0) {
        _runner["lm_core"]->reset_states();
    }
    _is_eog = true;
    _is_first_run = true;
}

bool LMTextGeneric::is_generation_complete() const {
    return !_is_first_run && _is_eog;
}

TensorDict LMTextGeneric::run_impl(const TensorDict& input, ProfilerView profiler) {
    (void)profiler;

    COMPUTE_FLOW_ASSERT(input["text_prompt"].format.dt == DataType::String);

    // When the user calls reset_states(), _is_eog is forced to true.
    // Therefore, whenever _is_eog is true, the following code runs.
    // This can happen in two scenarios:
    // 1. The context was explicitly reset by the user.
    // 2. Generation reached EOG, meaning the previous response finished and
    //    we continue the conversation using the existing context. In this scenario, the user
    //    is responsible to update the input. Otherwise, the same input will be used
    if(_is_eog) {


        const TensorData* prompt_tensor = &input["text_prompt"];
        TensorData templated_input;
        std::string prompt_buffer;

        // Resolve the <<USER_TEXT>> placeholder and finalize the input prompt
        const std::string user_text = tensor_to_string(*prompt_tensor);
        prompt_buffer = render_prompt_template(_prompt_template, user_text);

        templated_input.data = (void*)prompt_buffer.data();
        templated_input.format.dt = DataType::String;
        templated_input.format.shape = { static_cast<int64_t>(prompt_buffer.size()) };
        prompt_tensor = &templated_input;

        const auto token_ids_fmt = _runner["lm_core"]->get_preprocessing_format(*prompt_tensor);

        // Get the memory for the token ids
        auto t_in_token_ids_tmp = _internal_memory.get_tensor("in_token_ids");

        void* token_ids_tmp_ptr      = t_in_token_ids_tmp.data;
        const auto token_ids_tmp_fmt = t_in_token_ids_tmp.format;

        // Check if we have enough memory
        COMPUTE_FLOW_ASSERT(token_ids_tmp_fmt.get_nbytes() >= token_ids_fmt.get_nbytes());

        TensorData t_token_ids;
        t_token_ids.data   = token_ids_tmp_ptr;
        t_token_ids.format = token_ids_fmt;

        _runner["lm_core"]->preprocessing(*prompt_tensor, t_token_ids);

        _runner["lm_core"]->set_input("token_ids", t_token_ids);
    }

    _runner["lm_core"]->run();

    auto t_out_token_ids = _runner["lm_core"]->get_output("token_ids");

    auto t_out_text_tmp = _internal_memory.get_tensor("out_text");

    void* text_tmp_ptr         = t_out_text_tmp.data;
    const auto text_tmp_fmt    = t_out_text_tmp.format;
    const auto text_fmt = _runner["lm_core"]->get_postprocessing_format(t_out_token_ids);

    // Check if we have enough memory
    COMPUTE_FLOW_ASSERT(text_tmp_fmt.get_nbytes() >= text_fmt.get_nbytes());

    TensorDict output;
    output.declare_tensor("text");

    output["text"].data   = text_tmp_ptr;
    output["text"].format = text_fmt;

    _runner["lm_core"]->postprocessing(t_out_token_ids, output["text"]);

    // If we reach the end of generation, at the following call, we will
    // run the encode part
    _is_eog = _runner["lm_core"]->is_end_of_generation();
    _is_first_run = false;

    return output;
}

void LMTextGeneric::allocate_memory_preprocessing(int64_t max_context_length) {
    std::string mock_input_str = "mock_input";

    TensorData mock_input;
    mock_input.data = (void*)mock_input_str.data();
    mock_input.format.dt = DataType::String;
    mock_input.format.shape = { (int64_t)mock_input_str.size() };

    auto token_ids_fmt = _runner["lm_core"]->get_preprocessing_format(mock_input);
    token_ids_fmt.shape = {max_context_length};

    _internal_memory.allocate("in_token_ids", token_ids_fmt);
}

void LMTextGeneric::allocate_memory_postprocessing(int64_t max_token_ids) {
    std::vector<int64_t> mock_output_token_ids(16);

    TensorData mock_output;
    mock_output.data = (void*)mock_output_token_ids.data();
    mock_output.format.dt = DataType::I32; // TODO: We expect the data type to match the input. Make it robust
    mock_output.format.shape = { (int64_t)mock_output_token_ids.size() };
    auto text_fmt = _runner["lm_core"]->get_postprocessing_format(mock_output);

    _internal_memory.allocate("out_text", text_fmt);
}

}  // namespace compute_flow::bricks::lm_text
