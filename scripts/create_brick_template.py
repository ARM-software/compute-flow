#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Create a starter Compute Flow brick under bricks/<type>/<variant>."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# REUSE-IgnoreStart
COPYRIGHT = (
    f"SPDX-FileCopyrightText: Copyright {date.today().year} Arm Limited and/or its affiliates "
    "<open-source-office@arm.com>"
)
LICENSE = "SPDX-License-Identifier: Apache-2.0"
# REUSE-IgnoreEnd

HEADER_TEMPLATE = """//
// @COPYRIGHT@
//
// @LICENSE@
//

#pragma once

#include <memory>
#include <string_view>

#include "compute_flow/bricks/brick_base_v1.h"
#include "compute_flow/memory_planner.h"
#include "compute_flow/runners/runner.h"

namespace compute_flow::bricks::@TYPE@ {

/** @brief TODO: Describe the brick capability. */
class @CLASS@ : public BrickBaseV1 {
public:
    std::string_view api() const noexcept override { return "@API@"; }
    ValidationResult validate(const ComputeFlowManifest& manifest) const override;
    bool is_generation_complete() const override;

protected:
    void load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    void prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    void warmup_impl(ProfilerView profiler) override;
    void update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    TensorDict run_impl(const TensorDict& input, ProfilerView profiler) override;
    void reset_states_impl(ProfilerView profiler) override;

private:
    // TODO: Add a runner for each module and any persistent model state.
    std::unique_ptr<IRunner> _runner;
    MemoryPlanner _memory;
    bool _is_eog = false;
};

}  // namespace compute_flow::bricks::@TYPE@
"""

SOURCE_TEMPLATE = """//
// @COPYRIGHT@
//
// @LICENSE@
//

#include "@CLASS@.h"

namespace compute_flow::bricks::@TYPE@ {

ValidationResult @CLASS@::validate(const ComputeFlowManifest& manifest) const {
    const auto& info = manifest.get_brick_info();

    COMPUTE_FLOW_VALIDATE(info.type == "@TYPE@");
    COMPUTE_FLOW_VALIDATE(info.name == "@CLASS@");
    COMPUTE_FLOW_VALIDATE(info.version.major == 1);

    // Check modules presence
    // TODO: Replace "model" and add checks for every module in the manifest.
    // COMPUTE_FLOW_VALIDATE(manifest.has_module("model"));

    // Check valid file for each module
    // COMPUTE_FLOW_VALIDATE(!manifest["model"].model_path.empty());

    // Check valid runner for each module
    // TODO: Check the supported RunnerType for every module, for example:
    // COMPUTE_FLOW_VALIDATE(manifest["model"].framework_type == RunnerType::LiteRT);

    // Check assets
    // TODO: Validate required assets with manifest.has_asset() and get_asset().
    // COMPUTE_FLOW_VALIDATE(manifest.has_asset("asset"));
    // COMPUTE_FLOW_VALIDATE(!manifest.get_asset("asset")->empty());

    // Check configs presence
    // TODO: Validate required configs with manifest.has_config().
    // COMPUTE_FLOW_VALIDATE(manifest.has_config("config"));

    // Check constants presence
    // TODO: Validate required constants with manifest.has_constant().
    // COMPUTE_FLOW_VALIDATE(manifest.has_constant("constant"));

    // Validate constants values
    // TODO: Read constants and validate their model-specific ranges.
    // COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("constant") > 0);

    return {};
}

void @CLASS@::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)manifest;
    (void)profiler;

    // TODO: Create and set up a runner for each model module.
    // _runner = make_runner(manifest["model"].framework_type);

    // _runner->setup(manifest["model"], profiler);
}

void @CLASS@::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    // TODO: Read constants and assets, and allocate reusable tensors in _memory.
    update_config_impl(manifest, profiler);
}

void @CLASS@::warmup_impl(ProfilerView profiler) {
    TensorDict input;

    // TODO: Declare representative dummy input tensors with live backing storage.
    // BrickBaseV1 resets all state after warmup.
    run_impl(input, profiler);
}

void @CLASS@::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)manifest;
    (void)profiler;

    // TODO: Read and validate model-specific runtime configs.
}

void @CLASS@::reset_states_impl(ProfilerView profiler) {
    (void)profiler;

    // TODO: Reset each runner and any additional recurrent or streaming state.
    // _runner->reset_states();
    _is_eog = false;
}

bool @CLASS@::is_generation_complete() const { return _is_eog; }

TensorDict @CLASS@::run_impl(const TensorDict& input, ProfilerView profiler) {
    (void)input;
    (void)profiler;

    // TODO: Validate inputs with COMPUTE_FLOW_CHECK, bind runner inputs, run the
    // model, and return tensors that match the @API@ brick API contract.
    // _runner->set_input("input", input["input"]);
    // _runner->run();
    // TensorDict output;
    // output.declare_tensor("output") = _runner->get_output("output");
    // _is_eog = true;
    // return output;
    COMPUTE_FLOW_CHECK(false && "TODO: implement model-specific inference");
    return {};
}

}  // namespace compute_flow::bricks::@TYPE@
"""

README_TEMPLATE = """<!--
@COPYRIGHT@

@LICENSE@
-->

# @CLASS@

TODO: Describe what this Compute Flow brick does.

## Model package

TODO: Document the model source, license, manifest, and validation results.

## Configuration

TODO: Document model-specific configs, constants, and required assets.

## Usage

TODO: Describe inputs and outputs defined by the `@API@` brick API.
"""


def variant_name(class_name: str) -> str:
    """Convert a C++ brick class name to its snake_case source directory."""
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", class_name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def render(template: str, brick_type: str, class_name: str, api: str) -> str:
    """Fill the shared identity and licensing tokens in a text template."""
    return (
        template.replace("@COPYRIGHT@", COPYRIGHT)
        .replace("@LICENSE@", LICENSE)
        .replace("@TYPE@", brick_type)
        .replace("@CLASS@", class_name)
        .replace("@API@", api)
    )


def main() -> None:
    """Parse the requested brick identity and write its starter files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("brick_type", help="Brick type, for example audio_detection")
    parser.add_argument(
        "class_name", help="C++ class name, for example Nemotron3Diarization"
    )
    parser.add_argument("--api", help="Brick API name (default: <type>.v1)")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "bricks",
        help="Parent of the brick type directory (default: repository bricks/)",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.brick_type):
        parser.error("brick_type must use lowercase letters, digits, and underscores")
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", args.class_name):
        parser.error("class_name must be a PascalCase C++ identifier")

    api = args.api or f"{args.brick_type}.v1"
    if not re.fullmatch(r"[a-z][a-z0-9_]*\.v[1-9][0-9]*", api):
        parser.error("api must look like <type>.v<version>")

    destination = args.output_root / args.brick_type / variant_name(args.class_name)
    if destination.exists():
        parser.error(f"destination already exists: {destination}")

    destination.mkdir(parents=True)
    files = {
        f"{args.class_name}.h": render(
            HEADER_TEMPLATE, args.brick_type, args.class_name, api
        ),
        f"{args.class_name}.cpp": render(
            SOURCE_TEMPLATE, args.brick_type, args.class_name, api
        ),
        "README.md": render(README_TEMPLATE, args.brick_type, args.class_name, api),
        "info.json": json.dumps(
            {
                "name": args.class_name,
                "type": args.brick_type,
                "api": api,
                "description": "TODO: Describe this brick.",
                "author": "TODO: Add publisher.",
                "version": "1.0.0",
                "models": {},
            },
            indent=2,
        )
        + "\n",
    }

    for filename, contents in files.items():
        (destination / filename).write_text(contents, encoding="utf-8")

    print(f"Created brick template: {destination}")


if __name__ == "__main__":
    main()
