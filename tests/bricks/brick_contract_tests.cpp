//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

#include "compute_flow/bricks/brick_factory.h"

namespace {

using compute_flow::BrickInfo;
using compute_flow::ComputeFlowManifest;
using compute_flow::ValidationResult;
using compute_flow::bricks::Brick;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_validation_mentions(const ValidationResult& result, std::string_view expected,
                                 const std::string& context) {
    require(!result, context + " unexpectedly passed validation");
    require(result.error.find(expected) != std::string::npos,
            context + " failed for an unexpected reason: " + result.error);
}

int32_t find_supported_manifest_major(const Brick& brick, ComputeFlowManifest& manifest, BrickInfo& brick_info,
                                      const std::string& context) {
    constexpr int32_t k_max_manifest_major = 64;
    for (int32_t major = 0; major <= k_max_manifest_major; ++major) {
        brick_info.version.major = major;
        manifest.set_brick_info(brick_info);
        const ValidationResult result = brick->validate(manifest);
        if (result.error.find("info.version.major") == std::string::npos) {
            return major;
        }
    }
    throw std::runtime_error(context + " rejected every supported manifest major version");
}

void test_brick_contract(const nlohmann::json& info) {
    BrickInfo brick_info;
    brick_info.type = info.at("type").get<std::string>();
    brick_info.name = info.at("name").get<std::string>();

    ComputeFlowManifest manifest;
    manifest.set_brick_info(brick_info);

    const Brick brick         = compute_flow::bricks::load_brick(manifest);
    const std::string context = brick_info.type + ":" + brick_info.name;
    require(brick != nullptr, context + " factory returned a null brick");
    require(brick->api() == info.at("api").get<std::string>(), context + " API does not match info.json");

    brick_info.version.major = find_supported_manifest_major(brick, manifest, brick_info, context);
    manifest.set_brick_info(brick_info);
    const ValidationResult incomplete_result = brick->validate(manifest);
    require(!incomplete_result, context + " accepted an incomplete manifest");
    require(incomplete_result.error.find("info.") == std::string::npos,
            context + " rejected its declared identity: " + incomplete_result.error);

    BrickInfo invalid_info = brick_info;
    invalid_info.type += "_invalid";
    manifest.set_brick_info(invalid_info);
    require_validation_mentions(brick->validate(manifest), "info.type", context + " invalid type");

    invalid_info = brick_info;
    invalid_info.name += "Invalid";
    manifest.set_brick_info(invalid_info);
    require_validation_mentions(brick->validate(manifest), "info.name", context + " invalid name");

    invalid_info = brick_info;
    ++invalid_info.version.major;
    manifest.set_brick_info(invalid_info);
    require_validation_mentions(brick->validate(manifest), "info.version.major", context + " invalid version");
}

int64_t test_installed_bricks(const nlohmann::json& cache) {
    int64_t num_tested = 0;
    for (const auto& [identity, versions] : cache.at("bricks").items()) {
        (void)identity;
        for (const auto& [version, entry] : versions.items()) {
            (void)version;
            if (!entry.at("metadata").at("installed").get<bool>()) {
                continue;
            }
            test_brick_contract(entry.at("info_json"));
            ++num_tested;
        }
    }
    return num_tested;
}

}  // namespace

int32_t main(int32_t argc, char** argv) {
    try {
        require(argc == 2, "Usage: brick_contract_tests <cache.json>");

        std::ifstream cache_stream(argv[1]);
        require(cache_stream.is_open(), std::string("Failed to open brick cache: ") + argv[1]);

        nlohmann::json cache;
        cache_stream >> cache;

        const int64_t num_tested = test_installed_bricks(cache);
        require(num_tested > 0, "Brick cache contains no installed bricks");
        std::cout << "Validated " << num_tested << " installed brick contracts\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Brick contract test failed: " << error.what() << '\n';
        return 1;
    }
}
