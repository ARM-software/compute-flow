<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Python Code Conventions

These conventions apply to all Python code across Compute Flow, including bindings, Brick Manager
modules, command-line tools, tests, scripts, and examples.

**Naming**

- Classes use PascalCase.
- Functions and variables use lower snake_case.
- Counting variables use `num_<name>` instead of `n_<name>`.
- Module-level constants use upper snake case.
- Module-internal names use a leading underscore: `_<name>`.

**Types**

- Add parameter and return type annotations to public and reusable functions.
- Use built-in generic types such as `list`, `dict`, and `tuple` instead of their `typing`
  equivalents.
- Give NumPy arrays an explicit data type when a tensor or file format requires one.
- Avoid `Any` when a more precise type is practical.

**Functions And Modules**

- Keep variable definitions close to their first use.
- Add an empty line before and after a standalone function call. Calls that form one logical block
  may remain together.
- Use short comments to identify the major steps in command-line examples.
- Use an `if __name__ == "__main__":` guard for executable modules.

**Imports**

- Keep imports at the top of the file unless a dependency is intentionally optional or lazy.
- Group standard-library, third-party, and local imports, with one empty line between groups.
- Import only the names or modules the file needs.

**Formatting**

- Use Ruff for formatting and linting.
- Do not manually align code in ways that conflict with Ruff formatting.

**Comments And Docstrings**

- Write comments that explain intent rather than restating the code.
- Add docstrings to public modules, classes, and reusable functions.

**Examples**

```python
_DIMENSION_MULTIPLE = 16


def prepare_image(path: str) -> np.ndarray:
    image = load_image(path)

    padded_image = pad_image(image, _DIMENSION_MULTIPLE)

    return padded_image
```
