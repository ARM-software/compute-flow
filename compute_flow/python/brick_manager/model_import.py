#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Core model import planning and installation."""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from brick_manager.cache import write_cache
from brick_manager.catalog import load_required_cache, select_versioned_entry
from brick_manager.errors import BrickManagerError
from brick_manager.identity import (
    BrickSelector,
    parse_brick_selector,
    validate_model_component,
)
from brick_manager.metadata import load_installed_entry_info_json
from brick_manager.model_artifacts import (
    artifact_source_root_json,
    download_artifact_source,
    model_artifact_path,
    remove_artifact,
)
from brick_manager.paths import StatePaths


ImportProgressCallback = Callable[[str, str | None], None]
_CONFIG_TYPES = frozenset(
    ("bool", "float", "float32", "float64", "int32", "int64", "string")
)


@dataclass(frozen=True)
class ModelImportArtifact:
    key: str
    runner: str | None
    current_filename: str


@dataclass(frozen=True)
class ModelImportConfig:
    key: str
    value_type: str
    current_value: bool | float | int | str


@dataclass(frozen=True)
class ModelImportSource:
    value: str
    filename: str
    is_url: bool


@dataclass(frozen=True)
class ModelImportPreparation:
    brick: str
    version: str
    template_model: str
    new_model_name: str
    destination_path: Path
    template_path: Path | None
    template_source: dict[str, str] | None
    template_manifest_name: str
    manifest_filename: str
    manifest: dict[str, Any]
    artifacts: list[ModelImportArtifact]
    configs: list[ModelImportConfig]
    force: bool


@dataclass(frozen=True)
class ModelImportChoices:
    artifacts: dict[str, ModelImportSource]
    configs: dict[str, bool | float | int | str]


@dataclass(frozen=True)
class ModelImportResult:
    identity: str
    name: str
    template: str
    path: Path
    manifest: str


def prepare_model_import(
    state_paths: StatePaths,
    brick: str,
    template_model: str,
    new_model_name: str,
    *,
    force: bool = False,
) -> ModelImportPreparation:
    """Load and validate a declared template before collecting user choices."""
    validate_model_component("name", template_model)
    validate_model_component("name", new_model_name)
    cache = load_required_cache(state_paths)
    selector = parse_brick_selector(brick)
    if selector.version is not None:
        raise BrickManagerError(
            "Model import takes <type>:<name> without @version. "
            "You can select a particular version with:\n"
            f"compute-flow brick install {selector.identity}"
        )

    version, entry = select_versioned_entry(cache, selector)
    selected = BrickSelector(selector.brick_type, selector.name, version)
    info_json = entry["info_json"]
    if entry["metadata"].get("installed", False):
        info_json = load_installed_entry_info_json(entry, selected)
    declared_models = info_json.get("models", {})
    if template_model not in declared_models:
        available = ", ".join(sorted(declared_models)) or "none"
        raise BrickManagerError(
            f"Unknown template model for {selector.key}: {template_model}. "
            f"Available models: {available}"
        )

    installed_models = entry["metadata"].get("installed_models", {})
    template_state = installed_models.get(template_model)
    managed_template_path = model_artifact_path(
        state_paths,
        selected,
        template_model,
    )
    template_installed = (
        template_state is not None
        and template_state.get("origin", "declared") == "declared"
        and Path(template_state["path"]) == managed_template_path
        and managed_template_path.is_dir()
    )
    if template_installed:
        template_path = managed_template_path
        manifest_path, manifest = _find_template_manifest(template_path, info_json)
        template_source = None
        template_manifest_name = manifest_path.name
    else:
        template_path = None
        template_source, template_manifest_name, manifest = (
            _find_source_template_manifest(
                declared_models[template_model]["sources"],
                info_json,
            )
        )

    destination_path = model_artifact_path(state_paths, selected, new_model_name)
    if not force and (destination_path.exists() or new_model_name in installed_models):
        raise BrickManagerError(
            f"Model already exists: {new_model_name}. Use --force to replace it."
        )

    artifacts = _manifest_artifacts(manifest)
    configs = _manifest_configs(manifest)
    return ModelImportPreparation(
        brick=selector.key,
        version=version,
        template_model=template_model,
        new_model_name=new_model_name,
        destination_path=destination_path,
        template_path=template_path,
        template_source=template_source,
        template_manifest_name=template_manifest_name,
        manifest_filename=f"{new_model_name}_manifest.json",
        manifest=copy.deepcopy(manifest),
        artifacts=artifacts,
        configs=configs,
        force=force,
    )


def parse_model_import_source(value: str) -> ModelImportSource:
    """Parse and validate one local path or HTTP(S) artifact source."""
    source_path = Path(value).expanduser()
    if source_path.is_file():
        source_path = source_path.resolve()
        _validate_artifact_filename(source_path.name)
        return ModelImportSource(
            value=str(source_path),
            filename=source_path.name,
            is_url=False,
        )

    parsed = urlparse(value)
    if parsed.scheme in ("http", "https"):
        filename = Path(unquote(parsed.path)).name
        if not filename:
            raise BrickManagerError(
                f"Model artifact URL does not contain a filename: {value}"
            )
        _validate_artifact_filename(filename)
        return ModelImportSource(value=value, filename=filename, is_url=True)
    if parsed.scheme:
        raise BrickManagerError(
            f"Model artifact must be a local path or HTTP/HTTPS URL: {value}"
        )

    raise BrickManagerError(
        f"Model artifact file does not exist: {source_path.resolve()}"
    )


def parse_model_config_value(
    value_type: str,
    value: str,
) -> bool | float | int | str:
    """Parse a prompt value using a Compute Flow manifest config type."""
    if value_type == "string":
        return value
    if value_type == "bool":
        normalized = value.strip().lower()
        if normalized in ("true", "yes", "1"):
            return True
        if normalized in ("false", "no", "0"):
            return False
        raise BrickManagerError("Expected a boolean: true or false")
    if value_type in ("int32", "int64"):
        try:
            parsed = int(value)
        except ValueError as error:
            raise BrickManagerError(f"Expected an integer for {value_type}") from error
        minimum, maximum = (
            (-(2**31), 2**31 - 1) if value_type == "int32" else (-(2**63), 2**63 - 1)
        )
        if not minimum <= parsed <= maximum:
            raise BrickManagerError(f"Value is outside the {value_type} range")
        return parsed
    if value_type in ("float", "float32", "float64"):
        try:
            parsed_float = float(value)
        except ValueError as error:
            raise BrickManagerError(f"Expected a number for {value_type}") from error
        if not math.isfinite(parsed_float):
            raise BrickManagerError(f"Expected a finite number for {value_type}")
        return parsed_float
    raise BrickManagerError(f"Unsupported config type: {value_type}")


def import_model(
    state_paths: StatePaths,
    preparation: ModelImportPreparation,
    choices: ModelImportChoices,
    *,
    progress: ImportProgressCallback | None = None,
) -> ModelImportResult:
    """Create and atomically install an imported model package."""
    _validate_choices(preparation, choices)
    cache = load_required_cache(state_paths)
    selector = parse_brick_selector(preparation.brick)
    version, entry = _select_installed_entry(cache, selector)
    if version != preparation.version:
        raise BrickManagerError(
            "The installed brick changed during model import. Start the import again."
        )
    selected = BrickSelector(selector.brick_type, selector.name, version)
    info_json = load_installed_entry_info_json(entry, selected)
    _validate_manifest_brick(preparation.manifest, info_json)

    destination_path = preparation.destination_path
    installed_models = entry["metadata"].setdefault("installed_models", {})
    if not preparation.force and (
        destination_path.exists() or preparation.new_model_name in installed_models
    ):
        raise BrickManagerError(
            f"Model already exists: {preparation.new_model_name}. "
            "Use --force to replace it."
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{preparation.new_model_name}.",
        dir=destination_path.parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        package_path = temporary_root / "package"
        package_path.mkdir()
        manifest = copy.deepcopy(preparation.manifest)

        template_path = _materialize_template(
            state_paths,
            preparation,
            temporary_root,
            info_json,
            progress,
        )
        _copy_template_assets(preparation, template_path, package_path)
        for artifact in preparation.artifacts:
            manifest["models"][artifact.key]["filename"] = choices.artifacts[
                artifact.key
            ].filename
        for config in preparation.configs:
            if config.key in choices.configs:
                manifest["configs"][config.key]["value"] = choices.configs[config.key]

        _report(progress, "creating_manifest", None)
        manifest_path = package_path / preparation.manifest_filename
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        installed_filenames = set()
        for artifact in preparation.artifacts:
            source = choices.artifacts[artifact.key]
            if source.filename in installed_filenames:
                raise BrickManagerError(
                    f"Multiple model artifacts use the filename: {source.filename}"
                )
            installed_filenames.add(source.filename)
            artifact_path = package_path / source.filename
            if artifact_path.exists():
                raise BrickManagerError(
                    f"Imported artifact filename conflicts with a template asset: "
                    f"{source.filename}"
                )
            if source.is_url:
                _report(progress, "downloading", artifact.key)
                _download_file(source.value, artifact_path)
            else:
                _report(progress, "copying", artifact.key)
                shutil.copy2(source.value, artifact_path)

        _report(progress, "installing", None)
        previous_path = temporary_root / "previous"
        had_previous = destination_path.exists()
        if had_previous:
            os.replace(destination_path, previous_path)
        try:
            os.replace(package_path, destination_path)
            imported_state = {
                "path": str(destination_path),
                "origin": "imported",
                "template": preparation.template_model,
                "manifest": preparation.manifest_filename,
                "brick": copy.deepcopy(manifest["brick"]),
            }
            if isinstance(info_json.get("api"), str):
                imported_state["api"] = info_json["api"]
            installed_models[preparation.new_model_name] = imported_state
            write_cache(state_paths.cache, cache)
        except Exception:
            remove_artifact(destination_path)
            if had_previous:
                os.replace(previous_path, destination_path)
            raise

    return ModelImportResult(
        identity=f"{preparation.brick}@{preparation.version}",
        name=preparation.new_model_name,
        template=preparation.template_model,
        path=destination_path,
        manifest=preparation.manifest_filename,
    )


def imported_model_compatible(
    path: Path, state: dict[str, Any], info_json: dict[str, Any]
) -> bool:
    """Return whether imported state still identifies the installed brick."""
    recorded_api = state.get("api")
    if recorded_api is not None and recorded_api != info_json.get("api"):
        return False
    manifest_name = state.get("manifest")
    if not isinstance(manifest_name, str) or not manifest_name:
        return False
    try:
        manifest = json.loads((path / manifest_name).read_text(encoding="utf-8"))
        _validate_manifest_brick(manifest, info_json)
    except (BrickManagerError, OSError, json.JSONDecodeError):
        return False
    return True


def _select_installed_entry(
    cache: dict[str, Any],
    selector: BrickSelector,
) -> tuple[str, dict[str, Any]]:
    versions = cache.get("bricks", {}).get(selector.key)
    if not versions:
        raise BrickManagerError(f"Unknown brick: {selector.key}")
    installed = [
        (version, entry)
        for version, entry in versions.items()
        if entry["metadata"].get("installed", False)
    ]
    if not installed:
        raise BrickManagerError(
            f"{selector.key} is not installed. Install the brick first:\n"
            f"compute-flow brick install {selector.key}"
        )
    if len(installed) > 1:
        raise BrickManagerError(
            f"Invalid cache: multiple installed versions found for {selector.key}"
        )
    return installed[0]


def _find_template_manifest(
    template_path: Path,
    info_json: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for path in sorted(template_path.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            _validate_manifest_brick(document, info_json)
            _manifest_artifacts(document)
            _manifest_configs(document)
        except (BrickManagerError, OSError, json.JSONDecodeError):
            continue
        candidates.append((path, document))
    if not candidates:
        raise BrickManagerError(
            f"Template model has no compatible root manifest: {template_path}"
        )
    return candidates[0]


def _find_source_template_manifest(
    sources: list[dict[str, str]],
    info_json: dict[str, Any],
) -> tuple[dict[str, str], str, dict[str, Any]]:
    errors = []
    for source in sources:
        try:
            documents = artifact_source_root_json(source)
        except BrickManagerError as error:
            errors.append(str(error))
            continue

        candidates = _compatible_source_manifests(documents, info_json)
        if candidates:
            name, manifest = candidates[0]
            return copy.deepcopy(source), name, manifest

        source_identity = f"{source['kind']} {source['url']}"
        errors.append(f"{source_identity}: no compatible root manifest")

    details = "\n".join(f"- {error}" for error in errors)
    raise BrickManagerError(
        f"Template model sources contain no compatible root manifest:\n{details}"
    )


def _compatible_source_manifests(
    documents: list[tuple[str, str]],
    info_json: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    candidates = []
    for name, contents in documents:
        try:
            document = json.loads(contents)
            _validate_manifest_brick(document, info_json)
            _manifest_artifacts(document)
            _manifest_configs(document)
        except (BrickManagerError, json.JSONDecodeError):
            continue
        candidates.append((name, document))
    candidates.sort(key=lambda candidate: candidate[0])
    return candidates


def _validate_manifest_brick(
    manifest: Any,
    info_json: dict[str, Any],
) -> None:
    if not isinstance(manifest, dict):
        raise BrickManagerError("Model manifest root must be an object")
    brick = manifest.get("brick")
    if not isinstance(brick, dict):
        raise BrickManagerError("Model manifest is missing brick identity")
    expected = (info_json["type"], info_json["name"])
    actual = (brick.get("type"), brick.get("name"))
    if actual != expected:
        raise BrickManagerError(
            "Model manifest brick identity is incompatible: expected "
            f"{expected[0]}:{expected[1]}, got {actual[0]}:{actual[1]}"
        )
    version = brick.get("version")
    if not isinstance(version, str) or not version:
        raise BrickManagerError("Model manifest brick version must be a string")


def _manifest_artifacts(manifest: dict[str, Any]) -> list[ModelImportArtifact]:
    models = manifest.get("models")
    if not isinstance(models, dict) or not models:
        raise BrickManagerError("Model manifest must contain a non-empty models object")
    artifacts = []
    for key, model in models.items():
        if not isinstance(model, dict):
            raise BrickManagerError(f"Manifest model entry must be an object: {key}")
        filename = model.get("filename")
        if filename is None:
            continue
        if not isinstance(filename, str) or not filename:
            raise BrickManagerError(
                f"Manifest model filename must be a non-empty string: {key}"
            )
        runner = model.get("runner")
        artifacts.append(
            ModelImportArtifact(
                key=key,
                runner=runner if isinstance(runner, str) and runner else None,
                current_filename=filename,
            )
        )
    if not artifacts:
        raise BrickManagerError("Model manifest contains no model artifact filenames")
    return artifacts


def _manifest_configs(manifest: dict[str, Any]) -> list[ModelImportConfig]:
    configs = manifest.get("configs", {})
    if not isinstance(configs, dict):
        raise BrickManagerError("Model manifest configs must be an object")
    result = []
    for key, config in configs.items():
        if not isinstance(config, dict):
            raise BrickManagerError(f"Manifest config must be an object: {key}")
        value_type = config.get("type")
        if value_type not in _CONFIG_TYPES:
            raise BrickManagerError(f"Unsupported config type for {key}: {value_type}")
        value = config.get("value")
        _validate_config_value(key, value_type, value)
        result.append(ModelImportConfig(key, value_type, value))
    return result


def _validate_choices(
    preparation: ModelImportPreparation,
    choices: ModelImportChoices,
) -> None:
    expected_artifacts = {artifact.key for artifact in preparation.artifacts}
    if set(choices.artifacts) != expected_artifacts:
        raise BrickManagerError("Every model artifact needs exactly one import source")
    expected_configs = {config.key for config in preparation.configs}
    unexpected_configs = set(choices.configs) - expected_configs
    if unexpected_configs:
        names = ", ".join(sorted(unexpected_configs))
        raise BrickManagerError(f"Unknown model config overrides: {names}")
    configs_by_key = {config.key: config for config in preparation.configs}
    for key, value in choices.configs.items():
        _validate_config_value(key, configs_by_key[key].value_type, value)


def _validate_config_value(key: str, value_type: str, value: Any) -> None:
    if value_type == "string" and isinstance(value, str):
        return
    if value_type == "bool" and isinstance(value, bool):
        return
    if (
        value_type in ("int32", "int64")
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        minimum, maximum = (
            (-(2**31), 2**31 - 1) if value_type == "int32" else (-(2**63), 2**63 - 1)
        )
        if minimum <= value <= maximum:
            return
    if value_type in ("float", "float32", "float64"):
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                return
    raise BrickManagerError(
        f"Config value does not match its declared type: {key} ({value_type})"
    )


def _materialize_template(
    state_paths: StatePaths,
    preparation: ModelImportPreparation,
    temporary_root: Path,
    info_json: dict[str, Any],
    progress: ImportProgressCallback | None,
) -> Path:
    if preparation.template_path is not None:
        return preparation.template_path
    if preparation.template_source is None:
        raise BrickManagerError("Model import template source is unavailable")

    _report(progress, "fetching_template", None)
    template_path = temporary_root / "template"
    download_artifact_source(
        state_paths,
        preparation.template_source,
        template_path,
    )
    manifest_path = template_path / preparation.template_manifest_name
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BrickManagerError(
            f"Template manifest disappeared from its source: {manifest_path.name}"
        ) from error
    except json.JSONDecodeError as error:
        raise BrickManagerError(
            f"Template manifest is no longer valid JSON: {manifest_path.name}"
        ) from error
    _validate_manifest_brick(manifest, info_json)
    if manifest != preparation.manifest:
        raise BrickManagerError(
            "Template manifest changed during model import. Start the import again."
        )
    return template_path


def _copy_template_assets(
    preparation: ModelImportPreparation,
    template_path: Path,
    package_path: Path,
) -> None:
    excluded = {Path(artifact.current_filename) for artifact in preparation.artifacts}
    excluded.add(Path(preparation.template_manifest_name))
    excluded.add(Path("SHA256SUMS"))
    for source_path in template_path.rglob("*"):
        relative_path = source_path.relative_to(template_path)
        if any(part.startswith(".") for part in relative_path.parts):
            continue
        if _is_compatible_template_manifest(source_path, template_path, preparation):
            continue
        if relative_path in excluded or source_path.is_dir():
            continue
        destination_path = package_path / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def _is_compatible_template_manifest(
    source_path: Path,
    template_path: Path,
    preparation: ModelImportPreparation,
) -> bool:
    if source_path.parent != template_path or source_path.suffix != ".json":
        return False
    try:
        document = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(document, dict) or not isinstance(document.get("brick"), dict):
        return False
    selected_brick = preparation.manifest["brick"]
    brick = document["brick"]
    return (
        brick.get("type") == selected_brick["type"]
        and brick.get("name") == selected_brick["name"]
    )


def _download_file(url: str, destination_path: Path) -> None:
    temporary_path = destination_path.with_suffix(destination_path.suffix + ".download")
    try:
        with (
            urllib.request.urlopen(url) as response,
            temporary_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        os.replace(temporary_path, destination_path)
    except (OSError, urllib.error.URLError) as error:
        temporary_path.unlink(missing_ok=True)
        raise BrickManagerError(
            f"Failed to download model artifact from {url}: {error}"
        ) from error


def _validate_artifact_filename(filename: str) -> None:
    if filename in (".", "..") or Path(filename).name != filename:
        raise BrickManagerError(f"Invalid model artifact filename: {filename}")


def _report(
    progress: ImportProgressCallback | None,
    phase: str,
    artifact: str | None,
) -> None:
    if progress is not None:
        progress(phase, artifact)
