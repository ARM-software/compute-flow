#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Maintain installed custom operator declarations in the brick cache."""

from __future__ import annotations

from typing import Any

from brick_manager.identity import BrickSelector


def record_custom_ops(
    cache: dict[str, Any],
    selector: BrickSelector,
    custom_ops: list[dict[str, object]],
) -> None:
    """Record a brick's custom operators after installation or replacement."""
    remove_custom_ops(cache, selector)
    if custom_ops:
        cache["custom_ops"][selector.identity] = custom_ops
        cache["custom_ops"] = dict(sorted(cache["custom_ops"].items()))


def remove_custom_ops(cache: dict[str, Any], selector: BrickSelector) -> None:
    """Remove all custom operator records for an uninstalled brick."""
    inventory = cache.setdefault("custom_ops", {})
    for identity in list(inventory):
        if identity.startswith(f"{selector.key}@"):
            del inventory[identity]
    cache["custom_ops"] = dict(sorted(inventory.items()))
