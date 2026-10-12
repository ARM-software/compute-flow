#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Create a starter Compute Flow how-to guide under docs/howto/."""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

HOWTO_TEMPLATE = """<!--
@COPYRIGHT@

SPDX-License-Identifier: Apache-2.0
-->

# TODO: Give this guide a descriptive title

## Dependencies

- `uv`, used to create the Python environment.
- `git`, used by the Compute Flow brick manager.
- TODO: Add any other required tools or platforms; remove tools that this guide does not use.

## Goal

TODO: State what the reader will accomplish and what result they should expect.

Run all commands from the Compute Flow project root.

## TODO: Name the workflow

### Step 1 - Prepare the environment

Create and activate the Python environment:

```bash
uv sync && source .venv/bin/activate
```

Check the required system dependencies:

```bash
compute-flow check
```

If the check reports a missing dependency, follow the suggested action and run the check again.

### Step 2 - TODO: Name the action

TODO: Explain the first task-specific action.

```bash
# TODO: Add the command for this step.
```

### Step 3 - Verify the result

TODO: Explain how to verify the result. Add more steps or sections if the workflow needs them.

```bash
# TODO: Add a command that checks the result.
```
"""


def main() -> None:
    """Parse the guide name and write its template."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "name",
        help="Lowercase snake case name beginning with an -ing verb, for example developing_brick_api",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "docs" / "howto",
        help="Directory for the generated guide (default: docs/howto/)",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z]+ing(?:_[a-z0-9]+)*", args.name):
        parser.error("name must use lowercase snake case and start with an -ing verb")

    destination = args.output_root / f"{args.name}.md"
    content = HOWTO_TEMPLATE.replace(
        "@COPYRIGHT@",
        f"SPDX-FileCopyrightText: Copyright {date.today().year} Arm Limited and/or its affiliates "
        "<open-source-office@arm.com>",
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError:
        parser.error(f"destination already exists: {destination}")

    print(f"Created how-to guide template: {destination}")


if __name__ == "__main__":
    main()
