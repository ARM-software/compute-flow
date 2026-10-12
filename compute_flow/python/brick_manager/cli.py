#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Command line interface for the Compute Flow brick manager."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from brick_manager.build import build_target, format_target_list
from brick_manager.catalog import (
    get_brick_info,
    get_brick_model_info,
    get_brick_model_list,
    get_brick_model_path,
    get_brick_path,
    get_brick_summaries,
)
from brick_manager.check import check_environment
from brick_manager.completion import complete_cli, complete_cli_with_descriptions
from brick_manager.errors import BrickManagerError
from brick_manager.install import install_all_bricks, install_brick
from brick_manager.model_import import import_model, prepare_model_import
from brick_manager.model_import_interactive import (
    prompt_model_import,
    require_interactive_model_import,
)
from brick_manager.model_install import (
    install_models,
    uninstall_all_models,
    uninstall_models,
    validate_models_for_install,
)
from brick_manager.paths import resolve_state_paths
from brick_manager.registry_commands import (
    add_registry_source,
    list_registry_sources,
    remove_registry_source,
)
from brick_manager.render import (
    CONSOLE,
    render_brick_info,
    render_brick_model_info,
    render_brick_list,
    render_brick_model_list,
    render_build_result,
    render_check_result,
    render_install_all_result,
    render_install_result,
    render_model_install_result,
    render_model_import_result,
    render_model_uninstall_result,
    render_overview,
    render_registry_add_result,
    render_registry_list,
    render_registry_remove_result,
    render_uninstall_all_result,
    render_uninstall_result,
    render_update_result,
)
from brick_manager.uninstall import uninstall_all_bricks, uninstall_brick
from brick_manager.update import update_cache


_STUB_COMMANDS = ()


def _parse_brick_types(value: str) -> list[str]:
    brick_types = value.split(",")
    if any(not brick_type for brick_type in brick_types):
        raise argparse.ArgumentTypeError(
            "--type requires one or more comma-separated brick types"
        )
    return brick_types


def _not_implemented(command: str) -> int:
    print(f"compute-flow brick {command} is not implemented yet.", file=sys.stderr)
    return 1


def _add_brick_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    brick_parser = subparsers.add_parser("brick", help="Manage Compute Flow bricks")
    brick_subparsers = brick_parser.add_subparsers(
        dest="brick_command", metavar="COMMAND"
    )

    list_parser = brick_subparsers.add_parser(
        "list", help="List brick variants from the local cache"
    )
    list_parser.add_argument(
        "--type",
        dest="types",
        default=[],
        type=_parse_brick_types,
        metavar="TYPE[,TYPE...]",
        help="Brick types to display, separated by commas",
    )
    list_parser.add_argument(
        "--installed",
        action="store_true",
        help="Display only installed brick variants",
    )

    info_parser = brick_subparsers.add_parser(
        "info", help="Show cached metadata for one brick"
    )
    info_parser.add_argument("brick", metavar="type:name[@version]")

    path_parser = brick_subparsers.add_parser(
        "path", help="Print the installed brick source path"
    )
    path_parser.add_argument("brick", metavar="type:name")

    install_parser = brick_subparsers.add_parser(
        "install", help="Install one brick source"
    )
    install_parser.add_argument(
        "--force", action="store_true", help="Replace an existing install"
    )
    install_parser.add_argument(
        "--all",
        action="store_true",
        help="Install every brick source from the local cache",
    )
    install_parser.add_argument("brick", nargs="?", metavar="type:name[@version]")

    uninstall_parser = brick_subparsers.add_parser(
        "uninstall", help="Uninstall one brick and its models"
    )
    uninstall_parser.add_argument(
        "--all",
        action="store_true",
        help="Uninstall every installed brick and its models",
    )
    uninstall_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Assume yes for uninstall confirmation prompts",
    )
    uninstall_parser.add_argument("brick", nargs="?", metavar="type:name")


def _add_registry_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    registry_parser = subparsers.add_parser(
        "registry", help="Manage brick source registries"
    )
    registry_subparsers = registry_parser.add_subparsers(
        dest="registry_command", metavar="COMMAND", required=True
    )

    add_parser = registry_subparsers.add_parser(
        "add", help="Register a Git brick source"
    )
    add_parser.add_argument("url", metavar="git-url")

    remove_parser = registry_subparsers.add_parser(
        "remove", help="Remove a registered brick source"
    )
    remove_parser.add_argument("source", metavar="source")

    list_parser = registry_subparsers.add_parser(
        "list", help="List configured brick sources"
    )

    for command_parser in (add_parser, remove_parser, list_parser):
        command_parser.add_argument(
            "--registry",
            help="Use only this registry.txt source list",
        )


def _add_model_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    model_parser = subparsers.add_parser(
        "model", help="Manage model artifacts declared by bricks"
    )
    model_subparsers = model_parser.add_subparsers(
        dest="model_command", metavar="COMMAND"
    )
    model_list_parser = model_subparsers.add_parser(
        "list", help="List models declared by brick versions"
    )
    model_list_parser.add_argument("bricks", nargs="+", metavar="type:name[@version]")
    model_info_parser = model_subparsers.add_parser(
        "info", help="Show model install details"
    )
    model_info_parser.add_argument("brick", metavar="type:name")
    model_info_parser.add_argument("models", nargs="+", metavar="model")
    model_path_parser = model_subparsers.add_parser(
        "path", help="Print the installed model artifact path"
    )
    model_path_parser.add_argument("brick", metavar="type:name")
    model_path_parser.add_argument("model", metavar="model")
    model_install_parser = model_subparsers.add_parser(
        "install", help="Install models and their required brick"
    )
    model_install_parser.add_argument(
        "--force", action="store_true", help="Replace existing model artifacts"
    )
    model_install_parser.add_argument("brick", metavar="type:name")
    model_install_parser.add_argument("models", nargs="+", metavar="model")
    model_import_parser = model_subparsers.add_parser(
        "import",
        help=(
            "Create a managed model from external artifacts using a supported "
            "model as a template"
        ),
    )
    model_import_parser.add_argument(
        "--force", action="store_true", help="Replace an existing managed model"
    )
    model_import_parser.add_argument("brick", metavar="type:name")
    model_import_parser.add_argument("template_model", metavar="template-model")
    model_import_parser.add_argument("new_model_name", metavar="new-model-name")
    model_uninstall_parser = model_subparsers.add_parser(
        "uninstall", help="Uninstall models for the installed brick version"
    )
    model_uninstall_parser.add_argument(
        "--all",
        action="store_true",
        help="Uninstall all model artifacts for the brick",
    )
    model_uninstall_parser.add_argument("brick", metavar="type:name")
    model_uninstall_parser.add_argument("models", nargs="*", metavar="model")


def _add_build_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    build_parser = subparsers.add_parser(
        "build",
        help="Build Compute Flow and installed bricks",
        epilog=(
            "Use COMPUTE_FLOW_CMAKE_EXECUTABLE to select CMake. Android™ uses "
            "COMPUTE_FLOW_ANDROID_NDK_ROOT when set. Linux® Arm® 64-bit builds use "
            "COMPUTE_FLOW_LINUX_ARM64_CC and COMPUTE_FLOW_LINUX_ARM64_CXX when set. "
            "Otherwise, the Brick Manager uses discovered or managed tools."
        ),
    )
    build_parser.add_argument(
        "target",
        metavar="TARGET",
        help="Build target: list, native, android_arm64, or linux_arm64",
    )
    build_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="Parallel build jobs. Default: 4",
    )
    build_parser.add_argument(
        "--from-source",
        action="store_true",
        help="Build runtime dependencies from source instead of release archives",
    )
    build_parser.add_argument(
        "--streamline",
        action="store_true",
        help="Enable Arm® Streamline timeline annotations",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compute-flow")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.add_parser("check", help="Check required system dependencies")
    update_parser = subparsers.add_parser(
        "update", help="Refresh the local brick registry cache"
    )
    update_parser.add_argument("--registry", help="Path to a registry.txt source list")
    _add_registry_parser(subparsers)
    _add_brick_parser(subparsers)
    _add_model_parser(subparsers)
    _add_build_parser(subparsers)
    return parser


def _install_models_with_progress(
    state_paths,
    brick: str,
    models: list[str],
    *,
    force: bool = False,
):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=CONSOLE,
        transient=True,
    ) as progress:
        task = progress.add_task("Preparing model install", total=len(models))

        def report(model_name: str, phase: str) -> None:
            if phase == "done":
                progress.update(task, description=f"Installed {model_name}", advance=1)
                return
            progress.update(task, description=f"{phase.title()} {model_name}")

        return install_models(state_paths, brick, models, force=force, progress=report)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "__complete":
        completion_arguments = arguments[1:]
        if completion_arguments[:1] == ["--descriptions"]:
            for candidate, description in complete_cli_with_descriptions(
                resolve_state_paths(), completion_arguments[1:]
            ):
                printable_description = "".join(
                    character if character.isprintable() else " "
                    for character in description
                )
                single_line_description = " ".join(printable_description.split())
                print(f"{candidate}\t{single_line_description}")
        else:
            for candidate in complete_cli(resolve_state_paths(), completion_arguments):
                print(candidate)
        return 0

    parser = build_parser()
    args = parser.parse_args(arguments)

    if args.command == "check":
        result = check_environment(resolve_state_paths())
        render_check_result(result)
        return 1 if result.has_errors else 0

    if args.command == "update":
        try:
            result = update_cache(resolve_state_paths(), registry=args.registry)
        except BrickManagerError as error:
            print(error, file=sys.stderr)
            return 1
        render_update_result(result)
        return 0

    if args.command == "registry":
        state_paths = resolve_state_paths()
        try:
            if args.registry_command == "add":
                result = add_registry_source(
                    state_paths, args.url, registry=args.registry
                )
                render_registry_add_result(result)
            elif args.registry_command == "remove":
                result = remove_registry_source(
                    state_paths, args.source, registry=args.registry
                )
                render_registry_remove_result(result)
            else:
                sources = list_registry_sources(state_paths, registry=args.registry)
                render_registry_list(sources)
        except BrickManagerError as error:
            print(error, file=sys.stderr)
            return 1
        return 0

    if args.command == "build":
        if args.target == "list":
            print(format_target_list())
            return 0
        if args.jobs < 1:
            parser.error("compute-flow build --jobs must be at least 1")
        try:
            build_dir = build_target(
                resolve_state_paths(),
                args.target,
                jobs=args.jobs,
                from_source=args.from_source,
                streamline=args.streamline,
            )
        except BrickManagerError as error:
            print(error, file=sys.stderr)
            return 1
        render_build_result(build_dir)
        return 0

    if args.command not in ("brick", "model"):
        parser.print_help()
        return 0

    state_paths = resolve_state_paths()

    if args.command == "model":
        if args.model_command == "list":
            try:
                model_lists = [
                    get_brick_model_list(state_paths, brick) for brick in args.bricks
                ]
            except BrickManagerError as error:
                print(error, file=sys.stderr)
                return 1
            render_brick_model_list(model_lists)
            return 0
        if args.model_command == "info":
            try:
                model_info = get_brick_model_info(
                    state_paths,
                    args.brick,
                    args.models,
                )
            except BrickManagerError as error:
                print(error, file=sys.stderr)
                return 1
            render_brick_model_info(model_info)
            return 0
        if args.model_command == "path":
            try:
                model_path = get_brick_model_path(
                    state_paths,
                    args.brick,
                    args.model,
                )
            except BrickManagerError as error:
                print(error, file=sys.stderr)
                return 1
            if model_path is not None:
                print(model_path)
            return 0
        if args.model_command == "install":
            try:
                selected_brick = validate_models_for_install(
                    state_paths,
                    args.brick,
                    args.models,
                )
                brick_result = install_brick(state_paths, selected_brick)
                installed_brick = brick_result.identity.rsplit("@", maxsplit=1)[0]
                result = _install_models_with_progress(
                    state_paths,
                    installed_brick,
                    args.models,
                    force=args.force,
                )
                model_list = get_brick_model_list(state_paths, installed_brick)
            except BrickManagerError as error:
                print(error, file=sys.stderr)
                return 1
            render_install_result(brick_result)
            render_model_install_result(result)
            render_brick_model_list([model_list])
            return 0
        if args.model_command == "import":
            try:
                require_interactive_model_import()
                preparation = prepare_model_import(
                    state_paths,
                    args.brick,
                    args.template_model,
                    args.new_model_name,
                    force=args.force,
                )
                choices = prompt_model_import(preparation)
                selected_brick = f"{preparation.brick}@{preparation.version}"
                brick_result = install_brick(state_paths, selected_brick)

                def report(phase: str, artifact: str | None) -> None:
                    messages = {
                        "creating_manifest": "Creating manifest...",
                        "fetching_template": "Fetching template package...",
                        "installing": "Installing imported model...",
                    }
                    if phase in ("copying", "downloading"):
                        print(f"{phase.title()} model artifact '{artifact}'...")
                    else:
                        print(messages[phase])

                result = import_model(
                    state_paths,
                    preparation,
                    choices,
                    progress=report,
                )
            except (BrickManagerError, OSError) as error:
                print(error, file=sys.stderr)
                return 1
            render_install_result(brick_result)
            render_model_import_result(result)
            return 0
        if args.model_command == "uninstall":
            if args.all == bool(args.models):
                parser.error(
                    "compute-flow model uninstall requires exactly one of --all or model"
                )
            try:
                if args.all:
                    result = uninstall_all_models(state_paths, args.brick)
                else:
                    result = uninstall_models(state_paths, args.brick, args.models)
                installed_brick = result.identity.rsplit("@", maxsplit=1)[0]
                model_list = get_brick_model_list(state_paths, installed_brick)
            except BrickManagerError as error:
                print(error, file=sys.stderr)
                return 1
            render_model_uninstall_result(result)
            render_brick_model_list([model_list])
            return 0
        parser.error(
            "compute-flow model requires a command: list, info, path, import, "
            "install, or uninstall"
        )

    if args.brick_command is None:
        render_overview(state_paths)
        return 0

    if args.brick_command == "list":
        try:
            summaries = get_brick_summaries(
                state_paths,
                args.types,
                installed_only=args.installed,
            )
        except BrickManagerError as error:
            print(error, file=sys.stderr)
            return 1
        render_brick_list(
            summaries, brick_types=args.types, installed_only=args.installed
        )
        return 0

    if args.brick_command == "info":
        try:
            info = get_brick_info(state_paths, args.brick)
        except BrickManagerError as error:
            print(error, file=sys.stderr)
            return 1
        render_brick_info(info)
        return 0

    if args.brick_command == "path":
        try:
            brick_path = get_brick_path(state_paths, args.brick)
        except BrickManagerError as error:
            print(error, file=sys.stderr)
            return 1
        if brick_path is not None:
            print(brick_path)
        return 0

    if args.brick_command == "install":
        if args.all == bool(args.brick):
            parser.error(
                "compute-flow brick install requires exactly one of --all or brick"
            )
        try:
            if args.all:
                result = install_all_bricks(state_paths, force=args.force)
            else:
                result = install_brick(state_paths, args.brick, force=args.force)
        except BrickManagerError as error:
            print(error, file=sys.stderr)
            return 1
        if args.all:
            render_install_all_result(result)
        else:
            render_install_result(result)
            model_list = get_brick_model_list(state_paths, result.identity)
            render_brick_model_list([model_list])
        return 0

    if args.brick_command == "uninstall":
        if args.all == bool(args.brick):
            parser.error(
                "compute-flow brick uninstall requires exactly one of --all or brick"
            )
        try:
            confirm = (lambda _path: True) if args.yes else None
            if args.all:
                result = uninstall_all_bricks(
                    state_paths,
                    confirm=confirm,
                )
            else:
                result = uninstall_brick(
                    state_paths,
                    args.brick,
                    confirm=confirm,
                )
        except BrickManagerError as error:
            print(error, file=sys.stderr)
            return 1
        if args.all:
            render_uninstall_all_result(result)
        else:
            render_uninstall_result(result)
        return 0

    if args.brick_command in _STUB_COMMANDS:
        return _not_implemented(args.brick_command)

    parser.error("unknown brick command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
