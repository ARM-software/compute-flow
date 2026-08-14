#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Rich rendering for Compute Flow brick manager command output."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from brick_manager.catalog import (
    BrickInfo,
    BrickModelInfo,
    BrickModelList,
    BrickSummary,
)
from brick_manager.build import BuildResult
from brick_manager.check import CheckResult
from brick_manager.install import InstallAllResult, InstallResult
from brick_manager.model_install import ModelInstallResult, ModelUninstallResult
from brick_manager.paths import StatePaths
from brick_manager.uninstall import UninstallAllResult, UninstallResult
from brick_manager.update import UpdateResult


THEME = Theme(
    {
        "app.title": "bold",
        "app.subtitle": "dim",
        "section": "bold",
        "muted": "dim",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "info": "cyan",
        "key": "bold",
        "value": "",
        "path": "cyan",
        "command": "bold cyan",
        "dim": "dim",
    }
)

CONSOLE = Console(theme=THEME)


def render_overview(state_paths: StatePaths, console: Console = CONSOLE) -> None:
    console.print(_overview_panel(state_paths))


def render_update_result(result: UpdateResult, console: Console = CONSOLE) -> None:
    console.print(
        _summary_panel(
            "OK",
            "Cache Updated",
            _kv_table(
                (
                    ("Registry", _path(result.registry_path)),
                    ("Cache", _path(result.cache_path)),
                    ("Indexed versions", str(result.brick_versions)),
                )
            ),
            status_style="success",
        )
    )


def render_build_result(result: BuildResult, console: Console = CONSOLE) -> None:
    console.print(
        _summary_panel(
            "OK",
            "Build Complete",
            _kv_table(
                (
                    ("Build directory", Text(str(result.build_dir), style="bold path")),
                    ("Application SDK", Text(str(result.sdk_dir), style="bold path")),
                )
            ),
            status_style="success",
        )
    )


def render_check_result(result: CheckResult, console: Console = CONSOLE) -> None:
    if not result.has_errors:
        console.print(
            _summary_panel(
                "OK",
                "Environment Ready",
                Group(
                    Text("All required system dependencies are available."),
                    Rule("Dependencies", style="dim"),
                    _dependency_status_table(result),
                    Rule("Native Toolchain", style="dim"),
                    _native_toolchain_status_table(result),
                    Rule("Tools", style="dim"),
                    _managed_item_status_table(result.tools),
                    Rule("Toolchains", style="dim"),
                    _managed_item_status_table(result.toolchains),
                ),
                status_style="success",
            )
        )
        return

    console.print(
        _summary_panel(
            "WARN",
            "Environment Issues",
            Group(
                Text(
                    "Fix the environment problems below, then rerun compute-flow check.",
                    style="warning",
                ),
                Rule("Problems", style="dim"),
                _environment_problem_table(result),
                Rule("Detected", style="dim"),
                _dependency_status_table(result),
                Rule("Native Toolchain", style="dim"),
                _native_toolchain_status_table(result),
                Rule("Tools", style="dim"),
                _managed_item_status_table(result.tools),
                Rule("Toolchains", style="dim"),
                _managed_item_status_table(result.toolchains),
            ),
            status_style="warning",
        )
    )


def render_brick_list(
    summaries: list[BrickSummary], console: Console = CONSOLE
) -> None:
    if not summaries:
        console.print(
            _summary_panel(
                "WARN",
                "No Cached Bricks",
                Text("Run ", style="muted")
                + Text("compute-flow update", style="command")
                + Text(" to refresh the local cache.", style="muted"),
                status_style="warning",
            )
        )
        return

    console.print(
        Group(
            _section_header("Available Bricks", f"{len(summaries)} cached bricks"),
            (
                _compact_brick_list(summaries)
                if console.width < 78
                else _brick_table(summaries)
            ),
        )
    )


def render_brick_model_list(
    model_lists: list[BrickModelList], console: Console = CONSOLE
) -> None:
    model_count = sum(len(model_list.models) for model_list in model_lists)
    if model_count == 0:
        console.print(
            _summary_panel(
                "WARN",
                "Models",
                Text("No declared models.", style="muted"),
                status_style="warning",
            )
        )
        return

    console.print(
        Group(
            _section_header("Models", f"{model_count} declared models"),
            _brick_model_table(model_lists),
        )
    )


def render_brick_model_info(
    model_info: BrickModelInfo, console: Console = CONSOLE
) -> None:
    panels = []
    for model in model_info.models:
        content: list[RenderableType] = [
            _kv_table(
                (
                    ("Model", model.name),
                    ("Version", model.version),
                    ("Status", _model_info_status(model.installed)),
                    ("Size", _format_size(model.size_bytes)),
                )
            )
        ]
        if model.installed:
            content.extend(
                (
                    Rule("Location", style="dim"),
                    _path(Path(model.location or "")),
                    Rule("Manifests", style="dim"),
                    _manifest_list(model.manifests),
                )
            )

        panels.append(
            Panel(
                Padding(Group(*content), (0, 1)),
                title=_title_text("Model Details"),
                subtitle=Text(model_info.identity, style="muted"),
                border_style="muted",
                box=box.ROUNDED,
                expand=False,
            )
        )

    console.print(Group(*panels))


def render_brick_info(info: BrickInfo, console: Console = CONSOLE) -> None:
    metadata = info.metadata
    info_json = info.info_json

    content: list[RenderableType] = [
        Text(info_json.get("description", "-"), style="value"),
        Text(""),
        _kv_table(
            (
                ("Identity", info.identity),
                ("Type", _value(info_json.get("type"))),
                ("Name", _value(metadata.get("manager_name"))),
                ("Implementation", _value(info_json.get("name"))),
                ("Version", _value(info_json.get("version"))),
                ("Author", _value(info_json.get("author"))),
            )
        ),
        Rule("Manager", style="dim"),
        _kv_table(
            (
                ("Source", _value(metadata.get("source"))),
                ("Source type", _value(metadata.get("source_type"))),
                ("Source path", _path_or_dash(metadata.get("source_path"))),
                ("State", _installed_state(bool(metadata.get("installed", False)))),
                ("Install path", _path_or_dash(metadata.get("installed_source_path"))),
            )
        ),
    ]

    models = info_json.get("models", {})
    if models:
        content.extend((Rule("Models", style="dim"), _models_table(models)))
    else:
        content.extend(
            (
                Rule("Models", style="dim"),
                Text("No declared models.", style="muted"),
            )
        )

    console.print(
        Panel(
            Padding(Group(*content), (0, 1)),
            title=_title_text("Brick Details"),
            subtitle=Text(info.identity, style="muted"),
            border_style="muted",
            box=box.ROUNDED,
            expand=False,
        )
    )


def render_install_result(result: InstallResult, console: Console = CONSOLE) -> None:
    title_by_status = {
        "already_installed": "Brick Already Installed",
        "installed": "Brick Installed",
        "reinstalled_missing": "Brick Reinstalled",
        "replaced": "Brick Replaced",
    }
    content: list[RenderableType] = [
        _kv_table(
            (
                ("Brick", result.identity),
                ("State", _value(result.status.replace("_", " "))),
                ("Path", _path(result.install_path)),
            )
        )
    ]
    if result.removed_models or result.refreshed_models:
        content.extend(
            (
                Rule("Models", style="dim"),
                _model_reconcile_table(result),
            )
        )
    console.print(
        _summary_panel(
            "OK",
            title_by_status.get(result.status, "Brick installed"),
            Group(*content),
            status_style="success",
        )
    )


def render_model_install_result(
    result: ModelInstallResult, console: Console = CONSOLE
) -> None:
    brick, version = _split_versioned_identity(result.identity)
    console.print(
        _summary_panel(
            "OK",
            "Models Installed",
            Group(
                _kv_table((("Brick", brick), ("Version", version))),
                Rule("Models", style="dim"),
                _model_install_table(result),
            ),
            status_style="success",
        )
    )


def render_model_uninstall_result(
    result: ModelUninstallResult, console: Console = CONSOLE
) -> None:
    brick, version = _split_versioned_identity(result.identity)
    console.print(
        _summary_panel(
            "OK",
            "Models Uninstalled",
            Group(
                _kv_table((("Brick", brick), ("Version", version))),
                Rule("Models", style="dim"),
                _model_uninstall_table(result),
            ),
            status_style="success",
        )
    )


def render_install_all_result(
    result: InstallAllResult, console: Console = CONSOLE
) -> None:
    if result.total == 0:
        console.print(
            _summary_panel(
                "WARN",
                "No Cached Bricks to Install",
                _kv_table(
                    (
                        ("Cache", _path(result.cache_path)),
                        ("Total attempted", str(result.total)),
                    )
                ),
                status_style="warning",
            )
        )
        return

    console.print(
        _summary_panel(
            "OK",
            "Bricks Installed",
            Group(
                _kv_table(
                    (
                        ("Cache", _path(result.cache_path)),
                        ("Total attempted", str(result.total)),
                    )
                ),
                Rule("Results", style="dim"),
                _count_table(
                    (
                        ("Installed", result.statuses.get("installed", 0)),
                        (
                            "Already installed",
                            result.statuses.get("already_installed", 0),
                        ),
                        (
                            "Reinstalled missing",
                            result.statuses.get("reinstalled_missing", 0),
                        ),
                        ("Replaced", result.statuses.get("replaced", 0)),
                    )
                ),
            ),
            status_style="success",
        )
    )


def render_uninstall_result(
    result: UninstallResult, console: Console = CONSOLE
) -> None:
    title_by_status = {
        "uninstalled": "Brick Uninstalled",
        "cleared_missing": "Brick Install State Cleared",
    }
    content: list[RenderableType] = [
        _kv_table(
            (
                ("Brick", result.identity),
                ("State", _value(result.status.replace("_", " "))),
                ("Path", _path(result.install_path)),
            )
        )
    ]
    console.print(
        _summary_panel(
            "OK",
            title_by_status.get(result.status, "Brick uninstalled"),
            Group(*content),
            status_style="success",
        )
    )
    if result.kept_models:
        brick = result.identity.split("@", 1)[0]
        console.print(
            _summary_panel(
                "WARN",
                "Installed Models Kept",
                Group(
                    Text(
                        f"{len(result.kept_models)} installed model(s) were kept.",
                        style="warning",
                    ),
                    Text.assemble(
                        ("Use ", "muted"),
                        (
                            f"compute-flow model uninstall {brick} --all",
                            "command",
                        ),
                        (" to remove them.", "muted"),
                    ),
                    Rule("Models", style="dim"),
                    _kept_models_table(result.kept_models),
                ),
                status_style="warning",
            )
        )


def render_uninstall_all_result(
    result: UninstallAllResult, console: Console = CONSOLE
) -> None:
    if result.total == 0:
        console.print(
            _summary_panel(
                "WARN",
                "No Installed Bricks to Uninstall",
                _kv_table(
                    (
                        ("Cache", _path(result.cache_path)),
                        ("Total attempted", str(result.total)),
                    )
                ),
                status_style="warning",
            )
        )
        return

    console.print(
        _summary_panel(
            "OK",
            "Bricks Uninstalled",
            Group(
                _kv_table(
                    (
                        ("Cache", _path(result.cache_path)),
                        ("Total attempted", str(result.total)),
                    )
                ),
                Rule("Results", style="dim"),
                _count_table(
                    (
                        ("Uninstalled", result.statuses.get("uninstalled", 0)),
                        (
                            "Cleared missing",
                            result.statuses.get("cleared_missing", 0),
                        ),
                        ("Models kept", result.kept_models),
                    )
                ),
            ),
            status_style="success",
        )
    )
    if result.kept_models:
        console.print(
            _summary_panel(
                "WARN",
                "Installed Models Kept",
                Text.assemble(
                    (f"{result.kept_models} installed model(s) were kept. ", "warning"),
                    (
                        "Use compute-flow model uninstall <type:name> --all "
                        "to remove model artifacts for each brick.",
                        "muted",
                    ),
                ),
                status_style="warning",
            )
        )


def _overview_panel(state_paths: StatePaths) -> Panel:
    commands = Table.grid(padding=(0, 3))
    commands.add_column("Command", style="command", no_wrap=True)
    commands.add_column("Description", style="value", overflow="fold")
    commands.add_row("list", "List cached bricks")
    commands.add_row("info", "Show cached metadata for one brick")
    commands.add_row("path", "Print the installed brick source path")
    commands.add_row("install", "Install brick source files")
    commands.add_row("uninstall", "Remove installed brick source files")

    content = Group(
        Text("Manage cached and installed Compute Flow bricks.", style="app.subtitle"),
        Text(""),
        _kv_table(
            (
                ("Home", _path(state_paths.home)),
                ("Cache", _path(state_paths.cache)),
            )
        ),
        Rule("Commands", style="dim"),
        commands,
    )
    return Panel(
        Padding(content, (0, 1)),
        title=_title_text("Compute Flow Brick Manager"),
        subtitle=Text("compute-flow brick --help", style="command"),
        border_style="muted",
        box=box.ROUNDED,
        expand=False,
    )


def _operation_panel(title: str, identity: str, path: Path, status: str) -> Panel:
    return _summary_panel(
        "OK",
        title,
        _kv_table(
            (
                ("Brick", identity),
                ("State", _value(status.replace("_", " "))),
                ("Path", _path(path)),
            )
        ),
        status_style="success",
    )


def _models_table(models: dict[str, Any]) -> Table:
    table = Table(box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Model", style="key", overflow="fold", ratio=2)
    table.add_column("Ref", no_wrap=True)
    table.add_column("SHA256", overflow="fold", ratio=2)
    table.add_column("Description", overflow="fold", ratio=2)
    table.add_column("URL", overflow="fold")

    for name, metadata in sorted(models.items()):
        table.add_row(
            name,
            _value(metadata.get("ref")),
            _value(metadata.get("sha256")),
            _value(metadata.get("description")),
            _value(metadata.get("url")),
        )

    return table


def _dependency_status_table(result: CheckResult) -> Table:
    table = Table(box=box.SIMPLE_HEAD, expand=True, show_lines=False)
    table.add_column("Dependency", style="key", no_wrap=True)
    table.add_column("Executable", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Path", style="path", overflow="fold")

    for dependency in result.dependencies:
        if dependency.installed:
            status = Text("found", style="success")
        else:
            status = Text("missing", style="error")
        table.add_row(
            dependency.name,
            dependency.executable,
            status,
            dependency.path or "-",
        )

    return table


def _environment_problem_table(result: CheckResult) -> Table:
    table = Table(box=box.SIMPLE_HEAD, expand=True, show_lines=True)
    table.add_column("Item", style="key", no_wrap=True)
    table.add_column("Problem", overflow="fold", ratio=2)
    table.add_column("Action", overflow="fold", ratio=2)

    for dependency in result.missing:
        table.add_row(
            dependency.name,
            dependency.reason,
            "\n".join(dependency.install_actions),
        )

    for item in result.invalid_overrides:
        table.add_row(
            item.name,
            item.error or "Invalid override.",
            "Correct or unset the environment variable override.",
        )

    return table


def _native_toolchain_status_table(result: CheckResult) -> Table:
    table = Table(box=box.SIMPLE_HEAD, expand=True, show_lines=False)
    table.add_column("Compiler", style="key", no_wrap=True)
    table.add_column("Toolchain", overflow="fold")
    table.add_column("Version", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Path", style="path", overflow="fold", ratio=2)

    for compiler in result.native_toolchain:
        if compiler.installed:
            status = Text("found", style="success")
        else:
            status = Text("missing", style="error")
        table.add_row(
            compiler.name,
            (
                f"{compiler.toolchain} ({compiler.executable})"
                if compiler.toolchain
                else compiler.executable
            ),
            compiler.version or "-",
            status,
            compiler.path or "-",
        )

    return table


def _managed_item_status_table(items) -> Table:
    table = Table(box=box.SIMPLE_HEAD, expand=True, show_lines=False)
    table.add_column("Name", style="key", overflow="fold", ratio=2)
    table.add_column("Version", no_wrap=True)
    table.add_column("Provided by", overflow="fold", ratio=2)
    table.add_column("Status", overflow="fold", ratio=2)
    table.add_column("Path", style="path", overflow="fold", ratio=4)

    for item in items:
        if item.invalid:
            status = Text(item.status, style="error")
        elif item.installed:
            status = Text(item.status, style="success")
        else:
            status = Text(item.status, style="muted")
        table.add_row(
            item.name,
            item.version or "-",
            item.provider,
            status,
            item.path or "-",
        )

    return table


def _model_install_table(result: ModelInstallResult) -> Table:
    table = Table(box=box.SIMPLE_HEAD, expand=True, show_lines=False)
    table.add_column("Model", style="key", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("Path", style="path", overflow="fold")

    for item in result.items:
        table.add_row(item.name, item.status.replace("_", " "), str(item.path))

    return table


def _model_uninstall_table(result: ModelUninstallResult) -> Table:
    table = Table(box=box.SIMPLE_HEAD, expand=True, show_lines=False)
    table.add_column("Model", style="key", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("Path", style="path", overflow="fold")

    for item in result.items:
        table.add_row(item.name, item.status.replace("_", " "), str(item.path))

    return table


def _manifest_list(manifests: list[str]) -> Group | Text:
    if not manifests:
        return Text("No root model manifests found.", style="muted")
    return Group(*(Text(f"- {manifest}", style="value") for manifest in manifests))


def _kept_models_table(model_names: list[str]) -> Table:
    table = Table(box=box.SIMPLE_HEAD, expand=True, show_lines=False)
    table.add_column("Model", style="key", no_wrap=True)
    table.add_column("State", no_wrap=True)

    for model_name in model_names:
        table.add_row(model_name, "kept")

    return table


def _model_reconcile_table(result: InstallResult) -> Table:
    table = Table(box=box.SIMPLE_HEAD, expand=True, show_lines=False)
    table.add_column("Model", style="key", no_wrap=True)
    table.add_column("State", no_wrap=True)

    for model_name in result.removed_models:
        table.add_row(model_name, "removed")
    for model_name in result.refreshed_models:
        table.add_row(model_name, "refreshed")

    return table


def _brick_table(summaries: list[BrickSummary]) -> Table:
    brick_column_width = max(
        (len(summary.brick) for summary in summaries), default=len("Brick")
    )
    table = Table(box=box.SIMPLE_HEAD, expand=True, show_lines=False)
    table.add_column("Brick", style="key", no_wrap=True, width=brick_column_width)
    table.add_column("Version", style="value", no_wrap=True, width=7)
    table.add_column("Installed", justify="center", no_wrap=True, width=9)
    table.add_column("Description", style="value", overflow="fold")

    for summary in summaries:
        table.add_row(
            summary.brick,
            summary.version,
            _installed_version_marker(summary.installed_version),
            summary.description,
        )

    return table


def _brick_model_table(model_lists: list[BrickModelList]) -> Table:
    table = Table(box=box.SIMPLE_HEAD, expand=True, show_lines=False)
    table.add_column("Brick", style="key", no_wrap=True)
    table.add_column("Version", style="value", no_wrap=True)
    table.add_column("Model", style="key", no_wrap=True)
    table.add_column("Installed", justify="center", no_wrap=True)
    table.add_column("Description", overflow="fold", ratio=2)

    non_empty_model_lists = [
        model_list for model_list in model_lists if model_list.models
    ]
    for index, model_list in enumerate(non_empty_model_lists):
        brick, version = _split_versioned_identity(model_list.identity)
        for model in model_list.models:
            table.add_row(
                brick,
                version,
                model.name,
                _installed_version_marker(
                    model_list.version if model.installed else None
                ),
                model.description,
            )
        if index != len(non_empty_model_lists) - 1:
            table.add_row("", "", "", "", "")

    return table


def _compact_brick_list(summaries: list[BrickSummary]) -> Group:
    rows: list[RenderableType] = []
    for index, summary in enumerate(summaries):
        line = _installed_version_marker(summary.installed_version)
        line.append("  ")
        line.append(summary.brick, style="key")
        rows.append(line)
        rows.append(
            Padding(Text(f"Version {summary.version}", style="muted"), (0, 0, 0, 2))
        )
        rows.append(Padding(Text(summary.description, style="muted"), (0, 0, 0, 2)))
        if index != len(summaries) - 1:
            rows.append(Text(""))
    return Group(*rows)


def _summary_panel(
    status: str,
    title: str,
    content: RenderableType,
    *,
    status_style: str,
) -> Panel:
    return Panel(
        Padding(content, (0, 1)),
        title=_status_title(status, title, status_style),
        border_style="muted",
        box=box.ROUNDED,
        expand=False,
    )


def _kv_table(rows: Iterable[tuple[str, RenderableType | str]]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column("Field", style="key", no_wrap=True)
    table.add_column("Value", overflow="fold", ratio=1)
    for key, value in rows:
        table.add_row(key, _renderable_value(value))
    return table


def _count_table(rows: Iterable[tuple[str, int]]) -> Table:
    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column("Result", style="key")
    table.add_column("Count", justify="right", style="value")
    for key, value in rows:
        table.add_row(key, str(value))
    return table


def _section_header(title: str, subtitle: str | None = None) -> Padding:
    renderables: list[RenderableType] = [Text(title, style="app.title")]
    if subtitle:
        renderables.append(Text(subtitle, style="app.subtitle"))
    return Padding(Group(*renderables), (0, 0, 0, 0))


def _title_text(title: str) -> Text:
    return Text(title, style="app.title")


def _status_title(status: str, title: str, status_style: str) -> Text:
    return Text.assemble((status, status_style), (" ", "dim"), (title, "app.title"))


def _renderable_value(value: RenderableType | str) -> RenderableType:
    if isinstance(value, str):
        return Text(value, style="value")
    return value


def _split_versioned_identity(identity: str) -> tuple[str, str]:
    brick, separator, version = identity.rpartition("@")
    if not separator:
        return identity, "-"
    return brick, version


def _value(value: Any) -> Text:
    if value is True:
        return Text("OK yes", style="success")
    if value is False:
        return Text("no", style="muted")
    if value is None:
        return Text("-", style="muted")
    return Text(str(value))


def _path(path: Path) -> Text:
    return Text(str(path), style="path")


def _format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "-"

    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0

    if unit == "B":
        return f"{size_bytes} B"
    return f"{value:.1f} {unit} ({size_bytes:,} bytes)"


def _path_or_dash(value: Any) -> Text:
    if value is None:
        return Text("-", style="muted")
    return Text(str(value), style="path")


def _installed_state(installed: bool) -> Text:
    if installed:
        return Text("OK installed", style="success")
    return Text("not installed", style="muted")


def _model_info_status(installed: bool) -> Text:
    if installed:
        return Text("Installed", style="success")
    return Text("Not installed", style="muted")


def _installed_marker(installed: bool) -> Text:
    if installed:
        return Text("✓", style="success")
    return Text("-", style="muted")


def _installed_version_marker(version: str | None) -> Text:
    if version is None:
        return Text("-", style="muted")
    return Text(f"✓ {version}", style="success")
