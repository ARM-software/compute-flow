# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

_SELECTOR_PREFIXES = ("@sample", "@manifest")


def load_run_configs(
    path: str, allowed_arguments: set[str], required_arguments: set[str]
) -> list[dict[str, Any]]:
    """Load and validate run settings from a JSON file."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            root = json.load(stream)
    except OSError as error:
        raise ValueError(f"Failed to open JSON runs file '{path}': {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Failed to parse JSON runs file '{path}': {error}") from error

    if not isinstance(root, list) or not root:
        raise ValueError("JSON runs file must contain a non-empty array")

    configs: list[dict[str, Any]] = []
    allowed = allowed_arguments | {"id"}
    for run_index, config in enumerate(root, start=1):
        if not isinstance(config, dict):
            raise ValueError(f"Run {run_index} must be a JSON object")

        unknown_arguments = config.keys() - allowed
        if unknown_arguments:
            argument = sorted(unknown_arguments)[0]
            raise ValueError(f"Run {run_index} contains unknown argument '{argument}'")

        missing_arguments = (required_arguments | {"id"}) - config.keys()
        if missing_arguments:
            argument = sorted(missing_arguments)[0]
            raise ValueError(
                f"Run {run_index} is missing required argument '{argument}'"
            )

        run_id = config["id"]
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(
                f"Run {run_index} argument 'id' must be a non-empty string"
            )

        configs.append(config)

    return configs


def get_run_value(
    config: dict[str, Any],
    argument: str,
    default_value: Any,
    run_index: int,
) -> Any:
    """Return an optional argument after validating its type."""
    value = config.get(argument, default_value)
    if not isinstance(value, type(default_value)) or isinstance(value, bool):
        raise ValueError(f"Run {run_index} argument '{argument}' has an invalid value")
    if isinstance(value, str) and value.startswith(_SELECTOR_PREFIXES):
        raise ValueError(
            f"Run {run_index} argument '{argument}' does not support selectors"
        )
    return value
