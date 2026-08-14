//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/cli_examples/internal/manifest_selector.h"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace compute_flow::bricks::internal {
namespace {

constexpr const char manifest_selector_prefix[] = "@manifest/";
constexpr const char sample_selector_prefix[] = "@sample/";

bool is_brick_type_char(char value) {
    return (value >= 'a' && value <= 'z') || (value >= '0' && value <= '9') || value == '_' || value == '-';
}

bool is_brick_type_initial_char(char value) {
    return (value >= 'a' && value <= 'z') || (value >= '0' && value <= '9') || value == '_';
}

bool is_brick_name_char(char value) {
    return (value >= 'a' && value <= 'z') || (value >= '0' && value <= '9') || value == '-';
}

bool is_brick_name_initial_char(char value) {
    return (value >= 'a' && value <= 'z') || (value >= '0' && value <= '9');
}

bool is_model_component_char(char value) {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') || (value >= '0' && value <= '9') ||
           value == '_' || value == '.' || value == '+' || value == '-';
}

bool is_model_component_initial_char(char value) {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') || (value >= '0' && value <= '9') ||
           value == '_';
}

bool contains_dot_dot(const std::string& value) {
    return value.find("..") != std::string::npos;
}

bool is_explicit_path(const std::string& value) {
    return value.rfind("./", 0) == 0 || value.rfind("../", 0) == 0 || value.rfind("~", 0) == 0 ||
           std::filesystem::path(value).is_absolute();
}

void validate_component(const std::string& label,
                        const std::string& value,
                        bool (*is_valid_initial_char)(char),
                        bool (*is_valid_char)(char)) {
    if (value.empty() || contains_dot_dot(value)) {
        throw std::invalid_argument("Invalid manifest selector " + label + ": " + value);
    }

    if (!is_valid_initial_char(value.front())) {
        throw std::invalid_argument("Invalid manifest selector " + label + ": " + value);
    }

    for (char c : value) {
        if (!is_valid_char(c)) {
            throw std::invalid_argument("Invalid manifest selector " + label + ": " + value);
        }
    }
}

std::filesystem::path compute_flow_home() {
    if (const char* compute_flow_home = std::getenv("COMPUTE_FLOW_HOME")) {
        if (compute_flow_home[0] != '\0') {
            return std::filesystem::path(compute_flow_home);
        }
    }

    if (const char* home = std::getenv("HOME")) {
        if (home[0] != '\0') {
            return std::filesystem::path(home) / ".compute-flow";
        }
    }

    if (const char* user_profile = std::getenv("USERPROFILE")) {
        if (user_profile[0] != '\0') {
            return std::filesystem::path(user_profile) / ".compute-flow";
        }
    }

    throw std::runtime_error("COMPUTE_FLOW_HOME is not set and the user home directory could not be resolved");
}

} // namespace

std::string get_manifest_path(const std::string& value, const std::string& brick_type) {
    if (std::filesystem::is_regular_file(value)) {
        return value;
    }

    if (is_explicit_path(value)) {
        throw std::runtime_error("Manifest file does not exist: " + value);
    }

    if (value.rfind(manifest_selector_prefix, 0) != 0) {
        throw std::invalid_argument(
            "Invalid manifest value: " + value +
            "\nExpected a manifest path or @manifest/<brick-name>/<model-name>/<manifest-file>.json");
    }

    const std::string selector = value.substr(sizeof(manifest_selector_prefix) - 1);
    const size_t first_slash = selector.find('/');
    const size_t second_slash = selector.find('/', first_slash == std::string::npos ? 0 : first_slash + 1);

    if (first_slash == std::string::npos || second_slash == std::string::npos ||
        selector.find('/', second_slash + 1) != std::string::npos) {
        throw std::invalid_argument(
            "Invalid manifest selector: " + value +
            "\nExpected: @manifest/<brick-name>/<model-name>/<manifest-file>.json");
    }

    const std::string brick_name = selector.substr(0, first_slash);
    const std::string model_name = selector.substr(first_slash + 1, second_slash - first_slash - 1);
    const std::string manifest_name = selector.substr(second_slash + 1);

    validate_component("type", brick_type, is_brick_type_initial_char, is_brick_type_char);
    validate_component("brick", brick_name, is_brick_name_initial_char, is_brick_name_char);
    validate_component("model", model_name, is_model_component_initial_char, is_model_component_char);
    validate_component("manifest", manifest_name, is_model_component_initial_char, is_model_component_char);

    if (manifest_name.size() < 5 || manifest_name.substr(manifest_name.size() - 5) != ".json") {
        throw std::invalid_argument("Invalid manifest selector: expected a .json file: " + manifest_name);
    }

    const std::filesystem::path manifest_path =
        compute_flow_home() / "models" / brick_type / brick_name / model_name / manifest_name;

    if (!std::filesystem::is_regular_file(manifest_path)) {
        throw std::runtime_error("Manifest is not installed: " + value + "\nExpected file: " + manifest_path.string());
    }

    return manifest_path.string();
}

std::string get_sample_path(const std::string& value, const std::string& manifest_path) {
    if (value.empty() || value.front() != '@') {
        return value;
    }

    if (value.rfind(sample_selector_prefix, 0) != 0) {
        throw std::invalid_argument("Invalid sample value: " + value + "\nExpected a path or @sample/<path>");
    }

    const std::filesystem::path relative_path = value.substr(sizeof(sample_selector_prefix) - 1);
    if (relative_path.empty() || relative_path.is_absolute()) {
        throw std::invalid_argument("Invalid sample selector: " + value + "\nExpected: @sample/<path>");
    }

    for (const auto& component : relative_path) {
        if (component == "..") {
            throw std::invalid_argument("Invalid sample selector: " + value + "\nPath traversal is not allowed");
        }
    }

    const std::filesystem::path sample_path =
        std::filesystem::path(manifest_path).parent_path() / "samples" / relative_path;
    if (!std::filesystem::is_regular_file(sample_path)) {
        throw std::runtime_error("Sample is not installed: " + value + "\nExpected file: " + sample_path.string());
    }

    return sample_path.string();
}

} // namespace compute_flow::bricks::internal
