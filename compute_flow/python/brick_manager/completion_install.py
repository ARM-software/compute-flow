#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Install local shell completion support for Compute Flow development venvs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


_START_MARKER = "# >>> Compute Flow brick CLI completion >>>"
_END_MARKER = "# <<< Compute Flow brick CLI completion <<<"
_COMPLETION_RELATIVE_PATH = Path("share/compute-flow/brick_cli_completion.sh")


def install_completion_for_project(project_root: Path | str) -> bool:
    """Install completion into project_root/.venv and patch activate if present."""
    if os.environ.get("COMPUTE_FLOW_SKIP_COMPLETION_INSTALL"):
        return False

    root = Path(project_root).resolve()
    venv = root / ".venv"
    activate = venv / "bin" / "activate"
    if not activate.is_file():
        return False

    source_script = _find_source_script(root)
    if source_script is None:
        return False

    installed_script = venv / _COMPLETION_RELATIVE_PATH
    installed_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_script, installed_script)
    installed_script.chmod(0o644)

    _patch_activate(activate)
    return True


def _find_source_script(project_root: Path) -> Path | None:
    repo_script = project_root / "scripts" / "completion" / "brick_cli_completion.sh"
    if repo_script.is_file():
        return repo_script

    package_script = Path(__file__).with_name("completion") / "brick_cli_completion.sh"
    if package_script.is_file():
        return package_script

    return None


def _patch_activate(activate: Path) -> None:
    original = activate.read_text(encoding="utf-8")
    block = (
        f"{_START_MARKER}\n"
        f'if [ -n "${{VIRTUAL_ENV:-}}" ]; then\n'
        f'    _compute_flow_completion_script="$VIRTUAL_ENV/{_COMPLETION_RELATIVE_PATH.as_posix()}"\n'
        f'    if [ -r "$VIRTUAL_ENV/../scripts/completion/brick_cli_completion.sh" ]; then\n'
        f'        _compute_flow_completion_script="$VIRTUAL_ENV/../scripts/completion/brick_cli_completion.sh"\n'
        f"    fi\n"
        f'    if [ -r "$_compute_flow_completion_script" ]; then\n'
        f'        . "$_compute_flow_completion_script"\n'
        f"    fi\n"
        f"    unset _compute_flow_completion_script\n"
        "fi\n"
        f"{_END_MARKER}\n"
    )

    updated = _replace_or_append_block(original, block)
    if updated != original:
        activate.write_text(updated, encoding="utf-8")


def _replace_or_append_block(content: str, block: str) -> str:
    start = content.find(_START_MARKER)
    if start >= 0:
        end = content.find(_END_MARKER, start)
        if end >= 0:
            end += len(_END_MARKER)
            while end < len(content) and content[end] in "\r\n":
                end += 1
            return content[:start] + block + content[end:]

    separator = "" if content.endswith("\n") else "\n"
    return f"{content}{separator}\n{block}"
