#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Interactive prompts for compute-flow model import."""

from __future__ import annotations

import json
import sys
from typing import TextIO

from brick_manager.errors import BrickManagerError
from brick_manager.model_import import (
    ModelImportChoices,
    ModelImportPreparation,
    parse_model_config_value,
    parse_model_import_source,
)


def prompt_model_import(
    preparation: ModelImportPreparation,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> ModelImportChoices:
    """Collect artifact sources, config overrides, and final confirmation."""
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    require_interactive_model_import(input_stream)
    return _collect_model_import_choices(preparation, input_stream, output_stream)


def require_interactive_model_import(input_stream: TextIO | None = None) -> None:
    """Reject model import before source inspection when input is not interactive."""
    input_stream = input_stream or sys.stdin
    if not input_stream.isatty():
        raise BrickManagerError(
            "Model import requires interactive input. Non-interactive artifact and "
            "config options are not available yet."
        )


def _collect_model_import_choices(
    preparation: ModelImportPreparation,
    input_stream: TextIO,
    output_stream: TextIO,
) -> ModelImportChoices:
    _write(
        output_stream,
        f"Importing model '{preparation.new_model_name}'\n"
        f"Brick: {preparation.brick}\n"
        f"Template: {preparation.template_model}\n",
    )
    artifacts = {}
    for artifact in preparation.artifacts:
        runner = f"Runner: {artifact.runner}\n" if artifact.runner else ""
        _write(
            output_stream,
            f"\nModel artifact: {artifact.key}\n"
            f"{runner}"
            f"Current file: {artifact.current_filename}\n"
            "Enter a local path or URL:\n",
        )
        while True:
            value = _read_prompt(input_stream, output_stream)
            try:
                artifacts[artifact.key] = parse_model_import_source(value)
                break
            except BrickManagerError as error:
                _write(output_stream, f"{error}\nEnter a local path or URL:\n")

    configs = {}
    if preparation.configs:
        _write(output_stream, "\nConfigure model:\n")
        if any(config.value_type == "string" for config in preparation.configs):
            _write(
                output_stream,
                'Enter plain text or a JSON string, e.g. "line one\\nline two".\n',
            )
    for config in preparation.configs:
        current = (
            json.dumps(config.current_value, ensure_ascii=False)
            if config.value_type == "string"
            else _format_value(config.current_value)
        )
        _write(
            output_stream,
            f"\n{config.key}\nCurrent value:\n{current}\n\n"
            "New value [Enter to keep current]:\n",
        )
        while True:
            value = _read_prompt(input_stream, output_stream, allow_empty=True)
            if not value:
                break
            try:
                if config.value_type == "string" and value.lstrip().startswith('"'):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError as error:
                        raise BrickManagerError(
                            "Expected a valid JSON string enclosed in double quotes"
                        ) from error
                configs[config.key] = parse_model_config_value(
                    config.value_type,
                    value,
                )
                break
            except BrickManagerError as error:
                _write(output_stream, f"{error}\nNew value:\n")

    _write(
        output_stream,
        "\nImport summary:\n\n"
        f"  Model:     {preparation.new_model_name}\n"
        f"  Template:  {preparation.template_model}\n",
    )
    for artifact in preparation.artifacts:
        _write(
            output_stream,
            f"  {artifact.key}:  {artifacts[artifact.key].filename}\n",
        )
    _write(output_stream, "\nInstall imported model? [Y/n]: ")
    answer = input_stream.readline()
    if answer == "":
        raise BrickManagerError("Model import input ended before confirmation.")
    if answer.strip().lower() not in ("", "y", "yes"):
        raise BrickManagerError("Model import cancelled.")
    return ModelImportChoices(artifacts=artifacts, configs=configs)


def _read_prompt(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    allow_empty: bool = False,
) -> str:
    _write(output_stream, "> ")
    value = input_stream.readline()
    if value == "":
        raise BrickManagerError("Model import input ended before completion.")
    value = value.rstrip("\r\n")
    if not value and not allow_empty:
        _write(output_stream, "A value is required.\n")
        return _read_prompt(
            input_stream,
            output_stream,
            allow_empty=allow_empty,
        )
    return value


def _format_value(value: bool | float | int | str) -> str:
    if isinstance(value, str):
        return value if value else '""'
    return json.dumps(value)


def _write(output_stream: TextIO, value: str) -> None:
    output_stream.write(value)
    output_stream.flush()
