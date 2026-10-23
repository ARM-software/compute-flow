<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Prebuilt binaries

Compute Flow uses prebuilt dependency libraries to reduce build time. A dependency release contains
library archives, a manifest, checksums, and license information. The build downloads the enabled
libraries from release assets, trying the configured mirrors in order. The packages are mirrored on
[GitHub](https://github.com/ARM-software/compute-flow/releases). A package registry is not required.

## Versioning and storage

Compute Flow releases and dependency releases have independent versions. Several Compute Flow
versions can use the same dependency release when their required libraries have not changed.

Each Compute Flow source revision records its dependency selection in two files:

- `cmake/deps/dependencies.json` identifies the dependency bundle version and each library's
  upstream commit SHA, target profile, archive filename, and SHA256 checksum.
- `cmake/deps/host.cmake` provides the default release download URLs in priority order.

The manifest at a Compute Flow release tag describes the dependencies selected by that release. The
download URLs and manifest change together after a new dependency release has been published and
verified. Individual library versions are independent of the bundle version.

Each artifact's `version` is the full upstream commit SHA, also pinned by its source-build module.
The separate `sha256` field verifies the downloaded archive. Archive filenames retain readable
library version labels.

Binary archives belong to dependency releases and are absent from the normal source tree. Published
archives and tags remain unchanged so older Compute Flow releases can continue to use them. An
archive contains the installed headers, libraries, CMake package configuration, required tools, and
upstream license notices.

## Selection and caching

Direct CMake builds and the Brick Manager use the same download logic. The default target profiles
are:

| Target               | Profile           |
| -------------------- | ----------------- |
| Linux® Arm® 64-bit   | `linux_aarch64`   |
| macOS® Arm® 64-bit   | `macos_aarch64`   |
| Android™ Arm® 64-bit | `android_aarch64` |

The manifest records the exact library variants available for each profile. Its compiler family,
major version, and C++ standard are inputs to the prebuilt export target. Downloads select archives
by profile; they do not check the consumer's compiler. `COMPUTE_FLOW_DEPENDENCY_PROFILE` overrides
the default selection; it does not install or change the compiler. The consuming toolchain must be
compatible with the selected binaries.

LiteRT uses one archive per platform. On Linux®, GCC and Clang consumers share the `linux_aarch64`
archive and use compatible `libstdc++` and system libraries. The compiler recorded in the manifest
is the publishing toolchain; it does not require matching the consumer's compiler. Validate new
archives with both compiler families before publishing, rather than maintaining separate GCC and
Clang packages.

CMake verifies an archive's SHA256 checksum before extracting it into
`<build-directory>/_deps/prebuilt/<sha256>/`. Completed prefixes are reused on later builds. CMake
adds those prefixes to its package search path and discovers the libraries through their installed
package configurations. Downloads run during configuration so those package configurations are
available to `find_package`.

`COMPUTE_FLOW_DEPENDENCY_BASE_URL` accepts a release download URL, a local directory, or a
semicolon-separated list of these sources in priority order. It also accepts `file://` URLs. For
each archive, CMake tries the next source if downloading or verifying the package fails. Filenames
and checksums must match the active manifest regardless of the host. An explicit value replaces the
default list. The environment variable initializes the CMake cache on first configuration; an
explicit CMake `-D` argument changes an existing cached value.

## Source-build fallback

An unavailable profile, or an archive that cannot be downloaded and verified from any configured
source, triggers the existing source-build fallback. `compute-flow build <target> --from-source`
selects that path explicitly. The next build without `--from-source` restores the prebuilt download
attempt.

Direct CMake builds control downloads with `COMPUTE_FLOW_USE_PREBUILT_DEPS` and individual source
builds with `COMPUTE_FLOW_BUILD_<DEPENDENCY>_FROM_SOURCE`. After an automatic fallback, those cached
options must be reset to retry prebuilt downloads.

The build workflows are described in [Building from source](../howto/building_from_source.md) and
[Building prebuilt libraries](../howto/building_prebuilt_libraries.md).
