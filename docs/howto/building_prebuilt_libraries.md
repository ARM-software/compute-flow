<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Building prebuilt libraries

Use the `export-prebuilts` target to build and package dependencies for a new release. It reuses the
source builds under `cmake/deps/` and exports exactly the libraries in the selected profile from
`cmake/deps/dependencies.json`.

## Prepare

```bash
uv sync
source .venv/bin/activate
compute-flow check
```

Install any missing tools. Select the compiler family and major version recorded in the manifest.
Keep each artifact's `version` set to the full upstream commit SHA pinned under `cmake/deps/`.
Update its `filename` when the library version changes; the exporter uses that filename unchanged.
The exporter uses the project's CMake 3.24 baseline. Profiles containing LiteRT require CMake 4.0.1
or newer, as enforced by its source-build module.

## Build and export

Use a fresh build directory for each release/profile. For macOS®, select the matching Xcode
installation using `DEVELOPER_DIR` if necessary:

```bash
cmake -S cmake/prebuilts -B build_prebuilts \
  -DCOMPUTE_FLOW_PREBUILT_PROFILE=macos_aarch64 \
  -DCOMPUTE_FLOW_PREBUILT_VERSION=local
cmake --build build_prebuilts --target export-prebuilts --parallel 4
```

For Linux®, select `linux_aarch64` and pass the matching `CMAKE_C_COMPILER` and
`CMAKE_CXX_COMPILER`. Build on an Arm® 64-bit host or container using the oldest supported system
libraries. The resulting LiteRT archive is shared by GCC and Clang consumers.

For Android™, select `android_aarch64` and pass
`-DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake"` from the matching NDK.

The configuration checks the compiler and target architecture, and enforces Release mode, static
libraries, position-independent code, and the Arm® v8-A CPU baseline. It sets macOS® deployment
target `15.0`; Android™ uses `arm64-v8a`, `c++_shared`, and API `24`, preserving the original
publishing recipe. These settings apply to new exports; existing release archives are unchanged.
Library-specific options remain in the existing dependency modules.

## Use the output

`build_prebuilts/exports/` contains the archives, a profile-specific `dependencies.json`, and
`SHA256SUMS`. Archives include headers, libraries, CMake configuration, required tools, and license
notices. No separate license download is needed.

Use `local` for unpublished exports, or supply a new dependency bundle version when preparing a
release. Check that consumers can use the packages on the supported target systems, including a
LiteRT model run with both GCC and Clang on Linux®.

## Publish a dependency release

1. Combine the profile manifests and unchanged archives into the complete archive set listed in
   `cmake/deps/dependencies.json`, with matching `SHA256SUMS`.
2. Create `deps-v<version>` with the release title `Compute Flow Dependencies v<version>`. Make
   every archive, the manifest snapshot, and checksums available as release assets. Keep library
   license notices inside the archives.
3. Verify the published downloads and checksums. Then update `cmake/deps/dependencies.json` and
   `cmake/deps/host.cmake` together. Update source-build commit pins and third-party notices when
   library versions change.

The export target does not publish anything automatically. Reuse the existing dependency release
when its contents have not changed. Keep published archives and tags unchanged. Dependency releases
are versioned independently of Compute Flow releases; see
[Prebuilt binaries](../design/prebuilt_binaries.md).
