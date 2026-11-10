# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Generate temporary LiteRT fixtures and run the custom operator integration test."""

import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


_MODELS = (
    ("add_sub.tflite", ()),
    ("add_sub_i8.tflite", ("--dtype", "int8")),
    ("add_sub_same.tflite", ("--case", "same")),
    ("add_sub_different.tflite", ("--case", "different")),
    ("add_sub_subgraphs.tflite", ("--case", "subgraphs")),
    ("add_sub_three.tflite", ("--size", "3")),
    ("add_sub_shared_constant.tflite", ("--case", "shared_constant")),
)


def main() -> None:
    """Create fixture models outside the source tree for one test invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("test_executable", type=Path)
    args = parser.parse_args()
    generator = Path(__file__).parent / "fixtures" / "generate_add_sub.py"

    with TemporaryDirectory(prefix="compute_flow_custom_ops_") as temporary_dir:
        model_paths = []

        for filename, options in _MODELS:
            subprocess.run(
                [
                    sys.executable,
                    str(generator),
                    *options,
                    "--output-dir",
                    temporary_dir,
                ],
                check=True,
            )
            model_paths.append(str(Path(temporary_dir) / filename))

        subprocess.run([str(args.test_executable), *model_paths], check=True)


if __name__ == "__main__":
    main()
