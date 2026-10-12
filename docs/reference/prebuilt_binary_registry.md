<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Prebuilt binary registry

This registry lists the prebuilt dependency packages distributed for Compute Flow, their upstream
versions, licenses, and source code changes. Each row covers the package's published platform
variants. The broader dependency inventory is in
[Third-party notices](../../THIRD_PARTY_NOTICES.md).

## Packages

| Package       | Upstream version / commit                                                                                                          | Source code changes | Upstream license                                                                                                  |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| ExecuTorch    | [1.5.0 (f7140a46)](https://github.com/pytorch/executorch/tree/f7140a46ff38e919c557d45b102d3ff26097c8c9)                            | None                | [BSD-3-Clause](https://github.com/pytorch/executorch/blob/f7140a46ff38e919c557d45b102d3ff26097c8c9/LICENSE)       |
| LiteRT        | [2.1.6 (1461b6b2)](https://github.com/google-ai-edge/LiteRT/tree/1461b6b2def31713f5c71446eab844aae05d02e9)                         | None                | [Apache-2.0](https://github.com/google-ai-edge/LiteRT/blob/1461b6b2def31713f5c71446eab844aae05d02e9/LICENSE)      |
| MNN           | [3.6.1 (d407447e)](https://github.com/alibaba/MNN/tree/d407447ed56c4121a11ccbd266dc184ca1ead0c2)                                   | None                | [Apache-2.0](https://github.com/alibaba/MNN/blob/d407447ed56c4121a11ccbd266dc184ca1ead0c2/LICENSE.txt)            |
| ONNX Runtime  | [e6e048e073fd9e787195d809a66904990d48b082](https://github.com/microsoft/onnxruntime/tree/e6e048e073fd9e787195d809a66904990d48b082) | None                | [MIT](https://github.com/microsoft/onnxruntime/blob/e6e048e073fd9e787195d809a66904990d48b082/LICENSE)             |
| Protobuf      | [3.21.12 (f0dc78d7)](https://github.com/protocolbuffers/protobuf/tree/f0dc78d7e6e331b8c6bb2d5283e06aa26883ca7c)                    | None                | [BSD-3-Clause](https://github.com/protocolbuffers/protobuf/blob/f0dc78d7e6e331b8c6bb2d5283e06aa26883ca7c/LICENSE) |
| SentencePiece | [0.2.0 (17d7580d)](https://github.com/google/sentencepiece/tree/17d7580d6407802f85855d2cc9190634e2c95624)                          | None                | [Apache-2.0](https://github.com/google/sentencepiece/blob/17d7580d6407802f85855d2cc9190634e2c95624/LICENSE)       |

**Source code changes** describes modifications we make to a library's original source code before
building it. **None** means we haven't changed that source code. Otherwise, briefly describe what
changed and why, link to the patch or changed code, and mention if it only affects certain
platforms.

The licenses above cover the named upstream projects. Bundled third-party components may have
additional licenses and notices.
