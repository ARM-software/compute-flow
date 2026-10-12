#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Create a starter Python worked example under docs/examples/python/."""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

EXAMPLE_TEMPLATE = '''<!--
@COPYRIGHT@

SPDX-License-Identifier: Apache-2.0
-->

# TODO: Give this Python example a descriptive title

## Dependencies

- A Mac® with Apple silicon, or a Linux® system with an Arm®-based CPU, such as a Raspberry Pi® 5.
- `uv`, `git`, and a native C and C++ compiler.

TODO: Adjust the platform and tool requirements for this example.

## Goal

TODO: Explain the use case and list the bricks in the order that the application runs them.

Run all commands from the Compute Flow project root.

## Install the bricks and models

### Step 1: Set up Compute Flow

Create and activate the project environment:

```bash
uv sync && source .venv/bin/activate
```

Check the required system dependencies:

```bash
compute-flow check
```

If the check reports a missing dependency, follow the suggested action and run the check again.

### Step 2: Download the models

Update the local brick registry, then install the model packages:

```bash
compute-flow update
# TODO: Add one compute-flow model install command for each required brick.
```

## Build the native Python module

### Step 3: Build Compute Flow

Build Compute Flow with the installed bricks:

```bash
compute-flow build native
```

## Create the application

### Step 4: Connect the bricks

Create an application directory:

```bash
mkdir -p @EXAMPLE_DIR@
```

Save the following as `@EXAMPLE_DIR@/@EXAMPLE_NAME@.py`:

```python
"""TODO: Describe the application built with Compute Flow bricks."""

from __future__ import annotations

import argparse

import compute_flow_py as cf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # TODO: Add manifest, input, output, and use-case arguments.
    args = parser.parse_args(argv)

    options = cf.BrickOptions()
    options.runtime.enable_warmup = True

    # Load the manifests.
    # TODO: first_manifest = cf.load_manifest(args.first_manifest)
    # TODO: Load each additional manifest here.

    # Configure the manifests.
    # TODO: Set any use-case-specific config before loading the bricks.

    # Load the bricks.
    # TODO: first_brick = cf.load_brick(first_manifest)
    # TODO: Load each additional brick here.

    # Set up the bricks.
    # TODO: first_brick.setup(first_manifest, options)
    # TODO: Set up each additional brick here.

    # TODO: Prepare typed inputs and reset states before each request.
    # TODO: Run each brick until is_generation_complete() and connect its output.
    # TODO: Keep inputs alive until each brick completes generation.
    # TODO: Save and display the final result, then return 0.
    raise NotImplementedError("TODO: Connect the bricks and produce output")


if __name__ == "__main__":
    raise SystemExit(main())
```

TODO: Explain any use-case-specific data conversion or output-buffer handling between bricks.

## Run the pipeline

### Step 5: TODO: Describe the result the application produces

Resolve the installed package paths, then run the application on a representative sample:

```bash
# TODO: Resolve each installed package with compute-flow model path.
# TODO: Run python @EXAMPLE_DIR@/@EXAMPLE_NAME@.py with manifests and sample input.
```

TODO: Describe the saved result and include a short excerpt of the displayed output:

```text
TODO: Replace this with representative output.
```

## Try another brick variant

TODO: If a compatible variant is available, show its install, build, and run commands. Otherwise,
remove this section.
'''


def main() -> None:
    """Parse the example name and write its guide template."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "name",
        help="Lowercase snake case name, for example transcribing_audio",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "docs" / "examples" / "python",
        help="Directory for the generated guide (default: docs/examples/python/)",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", args.name):
        parser.error("name must use lowercase snake case")
    if args.name.endswith("_py"):
        parser.error(
            "omit the _py suffix; Python examples use the docs/examples/python/ directory"
        )

    destination = args.output_root / f"{args.name}.md"
    content = (
        EXAMPLE_TEMPLATE.replace(
            "@COPYRIGHT@",
            f"SPDX-FileCopyrightText: Copyright {date.today().year} Arm Limited and/or its affiliates "
            "<open-source-office@arm.com>",
        )
        .replace("@EXAMPLE_DIR@", f"{args.name}_example")
        .replace("@EXAMPLE_NAME@", args.name)
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError:
        parser.error(f"destination already exists: {destination}")

    print(f"Created Python example template: {destination}")


if __name__ == "__main__":
    main()
