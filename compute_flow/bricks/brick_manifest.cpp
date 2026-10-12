//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/accelerator.h"
#include "compute_flow/bricks/brick_manifest.h"
#include "compute_flow/model_data.h" // It should be runner_type.h

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <nlohmann/json.hpp>

namespace compute_flow {

namespace detail {

inline RunnerType runner_from_string(const std::string& s) {
    if (s == "executorch") {
#ifndef COMPUTE_FLOW_HAS_EXECUTORCH
        throw std::runtime_error(
            "Manifest requests runner 'executorch', but ExecuTorch support was disabled at build time. "
            "Rebuild with -DCOMPUTE_FLOW_ENABLE_EXECUTORCH=ON.");
#endif
        return RunnerType::ExecuTorch;
    }
    if (s == "litert") {
#ifndef COMPUTE_FLOW_HAS_LITERT
        throw std::runtime_error(
            "Manifest requests runner 'litert', but LiteRT support was disabled at build time. "
            "Rebuild with -DCOMPUTE_FLOW_ENABLE_LITERT=ON.");
#endif
        return RunnerType::LiteRT;
    }
    if (s == "sentencepiece_enc") {
#ifndef COMPUTE_FLOW_HAS_SENTENCEPIECE
        throw std::runtime_error(
            "Manifest requests runner 'sentencepiece_enc', but SentencePiece support was disabled at build time. "
            "Rebuild with -DCOMPUTE_FLOW_ENABLE_SENTENCEPIECE=ON.");
#endif
        return RunnerType::SentencePieceEncode;
    }
    if (s == "sentencepiece_dec") {
#ifndef COMPUTE_FLOW_HAS_SENTENCEPIECE
        throw std::runtime_error(
            "Manifest requests runner 'sentencepiece_dec', but SentencePiece support was disabled at build time. "
            "Rebuild with -DCOMPUTE_FLOW_ENABLE_SENTENCEPIECE=ON.");
#endif
        return RunnerType::SentencePieceDecode;
    }
    if (s == "hf_tokenizer") {
#ifndef COMPUTE_FLOW_HAS_HF_TOKENIZER
        throw std::runtime_error(
            "Manifest requests runner 'hf_tokenizer', but Hugging Face tokenizer support was disabled at build time. "
            "Rebuild with -DCOMPUTE_FLOW_ENABLE_HF_TOKENIZER=ON.");
#endif
        return RunnerType::HuggingFaceTokenizer;
    }
    if (s == "onnxrt") {
#ifndef COMPUTE_FLOW_HAS_ONNXRT
        throw std::runtime_error(
            "Manifest requests runner 'onnxrt', but ONNX Runtime support was disabled at build time. "
            "Rebuild with -DCOMPUTE_FLOW_ENABLE_ONNXRUNTIME=ON.");
#endif
        return RunnerType::ONNXRT;
    }
    if (s == "mnn") {
#ifndef COMPUTE_FLOW_HAS_MNN
        throw std::runtime_error(
            "Manifest requests runner 'mnn', but MNN support was disabled at build time. "
            "Rebuild Compute Flow with -DCOMPUTE_FLOW_ENABLE_MNN=ON.");
#else
        return RunnerType::MNN;
#endif
    }
    throw std::runtime_error("Unknown runner type: " + s);
}

inline LMRunnerType lm_runner_from_string(const std::string& s) {
    if (s == "llama_cpp") {
#ifndef COMPUTE_FLOW_HAS_LLAMACPP
        throw std::runtime_error(
            "Manifest requests LM runner 'llama_cpp', but llama.cpp support was disabled at build time. "
            "Rebuild with -DCOMPUTE_FLOW_ENABLE_LLAMACPP=ON.");
#endif
        return LMRunnerType::Llama_cpp;
    }
    throw std::runtime_error("Unknown LM runner type: " + s);
}

inline LMSamplingPolicy lm_sampling_policy_from_string(const std::string& s) {
    if (s == "greedy") return LMSamplingPolicy::Greedy;
    if (s == "distribution") return LMSamplingPolicy::Distribution;
    if (s == "disabled") return LMSamplingPolicy::Disabled;
    throw std::runtime_error("Unknown LM sampling policy: " + s);
}

inline LMRNGStrategy lm_rng_strategy_from_string(const std::string& s) {
    if (s == "mt19937") return LMRNGStrategy::Mt19937;
    if (s == "lcg") return LMRNGStrategy::Lcg;
    throw std::runtime_error("Unknown LM rng strategy: " + s + " (expected: mt19937, lcg)");
}

inline LMTopKStrategy lm_top_k_strategy_from_string(const std::string& s) {
    if (s == "strict_k") return LMTopKStrategy::StrictK;
    if (s == "threshold_ties") return LMTopKStrategy::ThresholdTies;
    throw std::runtime_error(
        "Unknown LM top_k_policy: " + s + " (expected: strict_k, threshold_ties)");
}

inline int get_index(const nlohmann::json& j, const char* field, const std::string& ctx) {
    if (!j.contains(field) || !j[field].is_number_integer())
        throw std::runtime_error("Missing or invalid '" + std::string(field) + "' in " + ctx);
    return static_cast<int>(j[field].get<int64_t>());
}

inline std::string get_required_string(const nlohmann::json& j, const char* field, const std::string& ctx) {
    if (!j.contains(field) || !j[field].is_string())
        throw std::runtime_error("Missing or non-string '" + std::string(field) + "' in " + ctx);
    return j[field].get<std::string>();
}

inline ConstantValue parse_typed_manifest_value(const nlohmann::json& obj, const std::string& ctx) {
    const std::string ty = get_required_string(obj, "type", ctx);
    if (!obj.contains("value")) {
        throw std::runtime_error("Missing 'value' for " + ctx);
    }

    if (ty == "int64") {
        return static_cast<int64_t>(obj["value"].get<int64_t>());
    }
    if (ty == "int32") {
        return static_cast<int32_t>(obj["value"].get<int32_t>());
    }
    if (ty == "float" || ty == "float32") {
        return static_cast<float>(obj["value"].get<float>());
    }
    if (ty == "float64") {
        return static_cast<double>(obj["value"].get<double>());
    }
    if (ty == "string") {
        return static_cast<std::string>(obj["value"].get<std::string>());
    }
    if (ty == "bool") {
        return static_cast<bool>(obj["value"].get<bool>());
    }

    throw std::runtime_error("Unsupported value type for " + ctx + ": " + ty);
}

inline std::string trim(std::string_view value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return std::string(value.substr(start, end - start));
}

inline std::string constant_to_string(const ConstantValue& value) {
    return std::visit([](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            return v;
        } else if constexpr (std::is_same_v<T, bool>) {
            return v ? "true" : "false";
        } else {
            return std::to_string(v);
        }
    }, value);
}

inline std::string resolve_constants_string(
    const std::string& key,
    const std::unordered_map<std::string, ConstantValue>& constants,
    std::unordered_map<std::string, std::string>& cache,
    std::unordered_set<std::string>& stack);

inline std::string render_template(
    const std::string& templ,
    const std::unordered_map<std::string, ConstantValue>& constants,
    std::unordered_map<std::string, std::string>& cache,
    std::unordered_set<std::string>& stack) {
    std::string out;
    out.reserve(templ.size());

    size_t pos = 0;
    while (pos < templ.size()) {
        const size_t start = templ.find("{{", pos);
        if (start == std::string::npos) {
            out.append(templ.substr(pos));
            break;
        }
        out.append(templ.substr(pos, start - pos));

        const size_t end = templ.find("}}", start + 2);
        if (end == std::string::npos) {
            out.append(templ.substr(start));
            break;
        }

        const std::string placeholder = trim(std::string_view(templ).substr(start + 2, end - (start + 2)));
        auto it = constants.find(placeholder);
        if (it != constants.end()) {
            if (std::holds_alternative<std::string>(it->second)) {
                out.append(resolve_constants_string(placeholder, constants, cache, stack));
            } else {
                out.append(constant_to_string(it->second));
            }
        } else {
            out.append(templ.substr(start, end - start + 2));
        }

        pos = end + 2;
    }

    return out;
}

inline std::string resolve_constants_string(
    const std::string& key,
    const std::unordered_map<std::string, ConstantValue>& constants,
    std::unordered_map<std::string, std::string>& cache,
    std::unordered_set<std::string>& stack) {
    auto cached = cache.find(key);
    if (cached != cache.end()) {
        return cached->second;
    }

    if (stack.find(key) != stack.end()) {
        throw std::runtime_error("Circular constants placeholder reference: " + key);
    }
    stack.insert(key);

    auto it = constants.find(key);
    if (it == constants.end()) {
        stack.erase(key);
        return "";
    }

    std::string resolved;
    if (std::holds_alternative<std::string>(it->second)) {
        const std::string& raw = std::get<std::string>(it->second);
        resolved = render_template(raw, constants, cache, stack);
    } else {
        resolved = constant_to_string(it->second);
    }

    cache.emplace(key, resolved);
    stack.erase(key);
    return resolved;
}

inline void set_manifest_constants(ComputeFlowManifest& manifest, const std::string& key, const ConstantValue& value) {
    std::visit([&](const auto& v) {
        manifest.set_constant(key, v);
    }, value);
}

inline void parse_lm_sampler_params(
    const nlohmann::json& sampler_params,
    LMParams& params,
    const std::string& ctx) {
    if (!sampler_params.is_object()) {
        throw std::runtime_error(ctx + " must be an object");
    }

    if (!sampler_params.contains("append_generated_token") ||
        !sampler_params["append_generated_token"].is_boolean()) {
        throw std::runtime_error(ctx + ".append_generated_token is required and must be a boolean");
    }
    params.append_generated_token = sampler_params["append_generated_token"].get<bool>();

    if (sampler_params.contains("policy")) {
        if (!sampler_params["policy"].is_string()) {
            throw std::runtime_error(ctx + ".policy must be a string");
        }
        params.sampling_policy = lm_sampling_policy_from_string(
            sampler_params["policy"].get<std::string>());
    }

    if (sampler_params.contains("distribution")) {
        if (!sampler_params["distribution"].is_object()) {
            throw std::runtime_error(ctx + ".distribution must be an object");
        }
        const auto& dist = sampler_params["distribution"];
        if (!dist.contains("rng_strategy") || !dist["rng_strategy"].is_string()) {
            throw std::runtime_error(
                ctx + ".distribution.rng_strategy is required (expected: mt19937, lcg)");
        }

        params.distribution_params.rng_strategy =
            lm_rng_strategy_from_string(dist["rng_strategy"].get<std::string>());
        if (dist.contains("top_k")) {
            params.distribution_params.top_k = dist["top_k"].get<int32_t>();
        }
        if (dist.contains("top_p")) {
            params.distribution_params.top_p = dist["top_p"].get<float>();
        }
        if (dist.contains("min_p")) {
            params.distribution_params.min_p = dist["min_p"].get<float>();
        }
        if (dist.contains("temperature")) {
            params.distribution_params.temperature = dist["temperature"].get<float>();
        }
        if (dist.contains("seed")) {
            params.distribution_params.seed = dist["seed"].get<uint32_t>();
        }
        if (dist.contains("repetition_penalty")) {
            params.repetition_penalty = dist["repetition_penalty"].get<float>();
        }
        if (dist.contains("top_k_strategy")) {
            if (!dist["top_k_strategy"].is_string()) {
                throw std::runtime_error(ctx + ".distribution.top_k_strategy must be a string");
            }
            params.distribution_params.top_k_strategy =
                lm_top_k_strategy_from_string(dist["top_k_strategy"].get<std::string>());
        }
    }

    if (sampler_params.contains("repetition_penalty")) {
        params.repetition_penalty = sampler_params["repetition_penalty"].get<float>();
    }

    if (sampler_params.contains("min_generated_tokens")) {
        if (!sampler_params["min_generated_tokens"].is_number_integer()) {
            throw std::runtime_error(ctx + ".min_generated_tokens must be an integer");
        }
        params.min_generated_tokens = sampler_params["min_generated_tokens"].get<int32_t>();
        if (params.min_generated_tokens < 0) {
            throw std::runtime_error(ctx + ".min_generated_tokens must be >= 0");
        }
    }

    if (sampler_params.contains("max_tokens_to_gen")) {
        if (!sampler_params["max_tokens_to_gen"].is_number_integer()) {
            throw std::runtime_error(ctx + ".max_tokens_to_gen must be an integer");
        }
        const int64_t max_tokens_to_gen = sampler_params["max_tokens_to_gen"].get<int64_t>();
        if (max_tokens_to_gen < -1 || max_tokens_to_gen > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error(ctx + ".max_tokens_to_gen must be >= -1");
        }
        params.max_tokens_to_gen = static_cast<int32_t>(max_tokens_to_gen);
    }

    if (sampler_params.contains("suppress_token_ranges")) {
        const auto& ranges = sampler_params["suppress_token_ranges"];
        if (!ranges.is_array()) {
            throw std::runtime_error(ctx + ".suppress_token_ranges must be an array");
        }

        params.suppress_token_ranges.clear();
        params.suppress_token_ranges.reserve(ranges.size());
        for (size_t idx = 0; idx < ranges.size(); ++idx) {
            const auto& range = ranges[idx];
            if (!range.is_array() || range.size() != 2) {
                throw std::runtime_error(
                    ctx + ".suppress_token_ranges[" + std::to_string(idx) +
                    "] must be an array of two integers");
            }
            if (!range[0].is_number_integer() || !range[1].is_number_integer()) {
                throw std::runtime_error(
                    ctx + ".suppress_token_ranges[" + std::to_string(idx) +
                    "] must contain integer values");
            }
            const int64_t start64 = range[0].get<int64_t>();
            const int64_t end64 = range[1].get<int64_t>();
            if (start64 < std::numeric_limits<int32_t>::min() ||
                start64 > std::numeric_limits<int32_t>::max() ||
                end64 < std::numeric_limits<int32_t>::min() ||
                end64 > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error(
                    ctx + ".suppress_token_ranges[" + std::to_string(idx) +
                    "] values out of int32 range");
            }
            const int32_t start = static_cast<int32_t>(start64);
            const int32_t end = static_cast<int32_t>(end64);
            if (start > end) {
                throw std::runtime_error(
                    ctx + ".suppress_token_ranges[" + std::to_string(idx) +
                    "] start must be <= end");
            }
            params.suppress_token_ranges.emplace_back(start, end);
        }
    }
}

} // namespace detail
} // namespace compute_flow

namespace compute_flow::bricks {

BrickManifest load_manifest(const std::string& json_path) {
    using nlohmann::json;
    namespace fs = std::filesystem;

    // Load JSON file
    std::ifstream ifs(json_path);
    if (!ifs) throw std::runtime_error("Unable to open JSON: " + json_path);
    json root;
    root = json::parse(ifs, /* callback */ nullptr, /* allow_exceptions */ true, /* ignore_comments */ true);

    // Determine models_base_path from the JSON directory
    const auto models_base_path = fs::absolute(fs::path(json_path)).parent_path();

    // Basic sanity checks
    if (!root.contains("models") || !root["models"].is_object())
        throw std::runtime_error("JSON must contain an object 'models'");

    ComputeFlowManifest manifest;

    // Brick section
    if (root.contains("brick")) {
        if (!root["brick"].is_object()) {
            throw std::runtime_error("'brick' must be an object");
        }

        const auto& b = root["brick"];

        BrickInfo bi;
        bi.name = detail::get_required_string(b, "name", "brick");
        bi.type = detail::get_required_string(b, "type", "brick");

        const std::string version = detail::get_required_string(b, "version", "brick");
        if (!bricks::parse_brick_version(version, bi.version)) {
            throw std::runtime_error("Invalid brick version: " + version);
        }

        manifest.set_brick_info(std::move(bi));
    } else {
        throw std::runtime_error("Missing mandatory brick section");
    }

    // Declare and configure modules
    for (auto& [module_name, module_obj] : root["models"].items()) {
        manifest.declare_module(module_name);

        if (module_obj.contains("lm_runner")) {
            throw std::runtime_error(
                "models." + module_name + ".lm_runner is no longer supported; use runner");
        }
        if (module_obj.contains("sampler_params")) {
            throw std::runtime_error(
                "models." + module_name + ".sampler_params is no longer supported; use runner_params.sampler");
        }
        if (!module_obj.contains("runner")) {
            throw std::runtime_error("Missing 'runner' in models." + module_name);
        }
        if (!module_obj.contains("runner_params") || !module_obj["runner_params"].is_object()) {
            throw std::runtime_error(
                "models." + module_name + ".runner_params is required and must be an object");
        }

        const std::string runner = detail::get_required_string(module_obj, "runner", "models." + module_name);
        const auto& runner_params = module_obj["runner_params"];
        if (runner == "llama_cpp") {
            manifest[module_name].lm_runner_type = detail::lm_runner_from_string(runner);

            if (runner_params.contains("embedding_position_mode")) {
                if (!runner_params["embedding_position_mode"].is_string()) {
                    throw std::runtime_error(
                        "models." + module_name + ".runner_params.embedding_position_mode must be a string");
                }
                const std::string mode = runner_params["embedding_position_mode"].get<std::string>();
                if (mode == "single") {
                    manifest[module_name].lm_params.embedding_position_mode =
                        LMEmbeddingPositionMode::Single;
                } else if (mode == "mrope_broadcast") {
                    manifest[module_name].lm_params.embedding_position_mode =
                        LMEmbeddingPositionMode::MRoPEBroadcast;
                } else {
                    throw std::runtime_error(
                        "models." + module_name +
                        ".runner_params.embedding_position_mode must be one of: single, mrope_broadcast");
                }
            }

            if (!runner_params.contains("sampler")) {
                throw std::runtime_error(
                    "models." + module_name + ".runner_params.sampler is required for llama_cpp");
            }
            detail::parse_lm_sampler_params(
                runner_params["sampler"],
                manifest[module_name].lm_params,
                "models." + module_name + ".runner_params.sampler");
        } else {
            manifest[module_name].framework_type = detail::runner_from_string(runner);
        }

        // Default acceleration is 1 CPU
        manifest[module_name].acc_config = AcceleratorConfig::cpu_only(1);

        const std::string filename = detail::get_required_string(module_obj, "filename", "models." + module_name);
        manifest[module_name].model_path = (models_base_path / filename).string();

        if (module_obj.contains("inputs")) {
            for (auto& [iname, idef] : module_obj["inputs"].items()) {
                const int index = detail::get_index(idef, "index", "models." + module_name + ".inputs." + iname);
                manifest[module_name].set_input(iname, index);
            }
        }

        if (module_obj.contains("outputs")) {
            for (auto& [oname, odef] : module_obj["outputs"].items()) {
                const int index = detail::get_index(odef, "index", "models." + module_name + ".outputs." + oname);
                manifest[module_name].set_output(oname, index);
            }
        }

        if (module_obj.contains("states")) {
            if (!module_obj["states"].is_object()) {
                throw std::runtime_error(
                    "models." + module_name + ".states must be an object");
            }

            for (auto& [sname, sdef] : module_obj["states"].items()) {
                const std::string base_ctx =
                    "models." + module_name + ".states." + sname;

                const int index_in  = detail::get_index(sdef, "index_in",  base_ctx);
                const int index_out = detail::get_index(sdef, "index_out", base_ctx);

                manifest[module_name].set_state(sname, index_in, index_out);
            }
        }

    }

    // Assets section: simple string map of asset name -> path
    if (root.contains("assets") && root["assets"].is_object()) {
        for (auto& [asset_name, asset_value] : root["assets"].items()) {
            if (!asset_value.is_string()) {
                throw std::runtime_error("Asset '" + asset_name + "' must be a string");
            }

            const std::string path = (models_base_path / asset_value.get<std::string>()).string();
            manifest.set_asset(asset_name, path);
        }
    }

    // Constants section
    if (root.contains("constants") && root["constants"].is_object()) {
        std::unordered_map<std::string, ConstantValue> constant_values;
        for (auto& [mkey, mobj] : root["constants"].items()) {
            constant_values.emplace(
                mkey,
                detail::parse_typed_manifest_value(mobj, "constants." + mkey));
        }

        std::unordered_map<std::string, std::string> cache;
        std::unordered_set<std::string> stack;
        for (auto& [mkey, mval] : constant_values) {
            if (std::holds_alternative<std::string>(mval)) {
                mval = detail::resolve_constants_string(mkey, constant_values, cache, stack);
            }
        }

        for (const auto& [mkey, mval] : constant_values) {
            detail::set_manifest_constants(manifest, mkey, mval);
        }
    }

    // Configs section
    if (root.contains("configs")) {
        if (!root["configs"].is_object()) {
            throw std::runtime_error("'configs' must be an object");
        }

        for (auto& [ckey, cobj] : root["configs"].items()) {
            const std::string config_key = ckey;
            const ConfigValue value =
                detail::parse_typed_manifest_value(cobj, "configs." + config_key);
            std::visit([&](const auto& v) {
                manifest.set_config(config_key, v);
            }, value);
        }
    }

    return manifest;
}

}  // namespace compute_flow::bricks
