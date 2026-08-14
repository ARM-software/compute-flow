#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""PEP 517 build backend wrapper for Compute Flow local development installs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from setuptools import build_meta as _setuptools_build_meta

from brick_manager.completion_install import install_completion_for_project


def _install_completion() -> None:
    try:
        install_completion_for_project(Path.cwd())
    except Exception:
        # Completion install is a local developer convenience. It must not make
        # package builds fail.
        pass


def _run_with_completion(hook_name: str, *args: Any, **kwargs: Any) -> Any:
    _install_completion()
    result = getattr(_setuptools_build_meta, hook_name)(*args, **kwargs)
    _install_completion()
    return result


def build_wheel(*args: Any, **kwargs: Any) -> str:
    return _run_with_completion("build_wheel", *args, **kwargs)


def build_editable(*args: Any, **kwargs: Any) -> str:
    return _run_with_completion("build_editable", *args, **kwargs)


def prepare_metadata_for_build_wheel(*args: Any, **kwargs: Any) -> str:
    return _run_with_completion("prepare_metadata_for_build_wheel", *args, **kwargs)


def prepare_metadata_for_build_editable(*args: Any, **kwargs: Any) -> str:
    return _run_with_completion("prepare_metadata_for_build_editable", *args, **kwargs)


def get_requires_for_build_wheel(*args: Any, **kwargs: Any) -> list[str]:
    return _run_with_completion("get_requires_for_build_wheel", *args, **kwargs)


def get_requires_for_build_editable(*args: Any, **kwargs: Any) -> list[str]:
    return _run_with_completion("get_requires_for_build_editable", *args, **kwargs)


def build_sdist(*args: Any, **kwargs: Any) -> str:
    return _run_with_completion("build_sdist", *args, **kwargs)
