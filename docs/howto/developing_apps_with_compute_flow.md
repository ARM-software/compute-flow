<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Compute Flow Application Development Guidelines

### Depend on Brick Type Libraries

When developing an application, link against the **brick type library** (for example, `tts_core.a`)
rather than a specific brick variant library (for example, `Qwen3TTS_core.a`).

This ensures that your application remains portable across different implementations of the same
brick type. By depending only on the brick interface, you can switch to a different brick variant
without requiring application code changes.

The build combines installed implementations of each brick type into one generic library. For
example, all installed TTS implementations are contained in `tts_core.a`; concrete variant archives
are not exposed in the application SDK.

### Compute Flow header files to include into the application

Applications should include only the following header:

```cpp
#include <compute_flow/compute_flow.h>
```

This should be the only Compute Flow header required by your application.

Avoid including brick variant-specific header files. Applications should be written against the
generic Compute Flow interfaces rather than concrete brick implementations. This ensures that a
brick variant can be replaced with another variant of the same brick type without requiring source
code changes or recompilation of application logic.

In other words, applications should depend on the brick contract, not on a specific brick
implementation.

### Best first inference performance

To achieve optimal performance from the first inference request, it is recommended to enable warmup
during the setup() phase.

If model warmup is required, use the warmup mechanism provided by Compute Flow rather than
implementing a custom warmup procedure within your application or brick. This ensures consistent
behaviour across brick variants and allows the framework to manage warmup efficiently.

### Install Bricks and Models Using the Brick Manager

Use the **Compute Flow Brick Manager** to install the bricks and models required by your
application.

The Brick Manager provides a consistent mechanism for discovering, downloading, and managing
compatible bricks and model assets. Refer to the Brick Manager design documentation for detailed
usage instructions and workflows.

### Build Compute Flow with the Brick Manager

Build the required Compute Flow libraries using the Brick Manager:

```bash
compute-flow build <target>
```

This ensures that all required dependencies, bricks, and platform-specific components are configured
correctly for your target environment.

The command also prepares `<build-directory>/sdk` by copying the public Compute Flow headers, the
library artifacts produced by that CMake build, and the `info.json` metadata for each installed
brick included in the build. Brick metadata is stored under
`share/compute_flow/bricks/<type>/<name>/info.json`. Applications can use a declared model's ordered
sources to download it at runtime and validate its `SHA256SUMS` file against the declared `sha256`,
without maintaining a separate registry of model URLs and checksums for the installed bricks. The
SDK therefore reflects the bricks and build options selected during configuration; it does not
create or combine additional libraries.

Applications can consume the packaged files through the included CMake script:

```cmake
add_subdirectory("/path/to/sdk" compute_flow_sdk)
target_link_libraries(my_application PRIVATE compute_flow::sdk)
```

The `compute_flow::sdk` target propagates the requirement to compile the application with C++17 or
later and supplies the platform-specific link requirements of the packaged libraries.

For Android™, configure 16 KB page alignment on each final application executable or shared library.
NDK r28 and later provide compatible alignment by default. With NDK r27 or earlier, add the
following to the application's top-level `CMakeLists.txt` after `project()`:

```cmake
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    foreach(linker_flags IN ITEMS CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS)
        string(APPEND ${linker_flags} " -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384")
    endforeach()
endif()
```

See the [Android™ 16 KB page-size guide](https://developer.android.com/guide/practices/page-sizes)
for compatibility requirements and verification steps.

The SDK can also be copied as a self-contained application directory. Its `CMakeLists.txt` defines
`PROJ_NAME` and automatically builds `main.cpp` when configured as the top-level project. Change
`PROJ_NAME` from its `minimal_example` default to select another application executable name.

### Always Support Streaming-Capable Bricks

Applications should always check `is_generation_complete()`, even when working with bricks that
currently produce complete outputs in a single step.

Some future brick variants may introduce streaming capabilities while preserving the same brick
interface. By implementing generation loops that rely on `is_generation_complete()`, your
application will automatically support both current and future variants without modification.

### Keep Inputs Valid During Generation

Keep the input tensors valid and unmodified until `is_generation_complete()` returns `true`.

### Follow the Brick Interface Specification

Refer to the Brick Interface Design Documentation for the authoritative definition of each brick's
API, lifecycle, and expected usage patterns.
