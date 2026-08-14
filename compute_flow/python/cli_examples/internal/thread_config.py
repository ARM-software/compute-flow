#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Shared CPU thread configuration helpers for Python CLI examples."""

import compute_flow_py as cf


def parse_hint_cpu_threads(thread_spec: str) -> tuple[int, dict[str, int]]:
    """Parse default and per-model CPU thread counts from a CLI specification."""
    default_threads = 1
    model_threads: dict[str, int] = {}

    if not thread_spec:
        return default_threads, model_threads

    first_item = True

    for item in thread_spec.split(","):
        if ":" not in item:
            if first_item:
                default_threads = int(item)
        else:
            model_name, threads = item.split(":", 1)
            model_threads[model_name] = int(threads)

        first_item = False

    return default_threads, model_threads


def set_cpu_threads(
    manifest, default_threads: int, model_threads: dict[str, int]
) -> None:
    """Configure default and per-model CPU thread counts in a brick manifest."""
    for model in manifest.models():
        manifest[model.model_name].acc_config = cf.AcceleratorConfig.cpu_only(
            default_threads
        )

    for model_name, threads in model_threads.items():
        manifest[model_name].acc_config = cf.AcceleratorConfig.cpu_only(threads)
