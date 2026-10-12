<!--
    SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/imgs/compute_flow_logo_text_dark.png" />
    <source media="(prefers-color-scheme: light)" srcset="docs/imgs/compute_flow_logo_text_light.png" />
    <img src="docs/imgs/compute_flow_logo_text_light.png" width="650" alt="Compute Flow 标志" />
  </picture>
</div>

# Compute Flow

[English](README.md) | [简体中文](README_zh-CN.md)

Compute Flow 是一个 SDK，可通过称为 **brick** 的可复用组件，构建 AI 和生成式 AI 应用。

Compute Flow 集成了 LiteRT、ExecuTorch、ONNX Runtime 和 llama.cpp 等推理运行时，让开发者无需直接依赖各个运行时的 API，即可创建新的
brick。它面向 Arm® 平台的边缘 AI 推理，支持 **Android™**、**Linux®** 和 **macOS®** 等目标平台，不支持微控制器。

## Compute Flow 能做什么？

Compute Flow 可帮助你通过以下方式构建功能丰富的端侧 AI 应用：

1. [**查找、安装和测试 brick**](docs/howto/getting_started.md)

   使用 Brick Manager（brick 管理工具），直接在终端中查找、安装和试用开箱即用的 AI brick。只需几条命令，即可为应用添加语音识别、语言模型、文本转语音、目标检测等功能。

2. [**构建复杂的 AI 流水线**](docs/examples/python/audio_to_audio_translation.md)

   组合多个 brick，构建完整的 AI 应用体验。例如，在 Python 中串联语音转文本、翻译和文本转语音 brick，即可构建语音到语音的翻译流水线。

3. [**创建可复用的 brick**](docs/howto/developing_bricks.md)

   使用不同推理运行时的模型构建新的 brick，让社区开发者能够轻松发现并复用它们。你可以通过 Brick Manager 命令行工具，从 GitHub URL **导入 brick**。

## 快速入门

使用 `uv` 创建 Compute Flow 工具所需的 Python 环境：

```bash
uv sync && source .venv/bin/activate
```

如果尚未安装 `uv`，请参阅[安装指南](https://docs.astral.sh/uv/getting-started/installation/#pypi)。

在安装或构建 brick 之前，先检查本地环境：

```bash
compute-flow check
```

如果 `compute-flow check` 提示缺少系统依赖，请按照提示安装依赖，然后重新运行该命令。

## 使用 brick

Compute Flow 的 Brick Manager 可用于查找 brick、安装 brick 推理代码和模型包，以及为所选目标平台构建 Compute Flow。典型的本机构建和运行流程如下：

1. 更新本地 brick 注册表缓存。
2. 安装 brick 及其一个模型。
3. 为目标平台构建 Compute Flow。
4. 运行 Python 或 C++ 命令行示例。

以下示例在 Compute Flow 中安装预打包的 Qwen3-TTS 文本转语音（TTS）brick，并运行本机命令行示例。

```bash
compute-flow update
compute-flow model install tts:qwen3-tts qwen3-tts-0-6b-custom-voice
compute-flow build native
```

### 命令行自动补全

激活 `.venv` 后，Brick Manager 和命令行示例均支持自动补全： <img src="docs/imgs/compute_flow_tab_completion.gif"
alt="Brick Manager 和 brick 命令行示例的终端自动补全演示"
width="800">

- Compute Flow 的 Brick Manager 可补全命令、brick 类型、brick 名称和模型。
- Python 和 C++ 命令行示例可补全已安装的 `@manifest/...` 和 `@sample/...` 选择器。

运行支持自动补全的 Python 命令行示例：

```bash
python -m cli_examples.tts_cli \
  -m @manifest/qwen3-tts/qwen3-tts-0-6b-custom-voice/qwen3_tts_custom_voice_en_manifest.json \
  -p "Hello, I am Aiden, your personal AI assistant. How can I help?" \
  -t 1
```

也可以使用相同参数运行 C++ 命令行示例：

```bash
./build_native/tts_cli \
  -m @manifest/qwen3-tts/qwen3-tts-0-6b-custom-voice/qwen3_tts_custom_voice_en_manifest.json \
  -p "Hello, I am Aiden, your personal AI assistant. How can I help?" \
  -t 1
```

`@manifest` 用于从已安装的模型包中选择清单文件。

如需了解本机和 Android™ 平台的完整操作流程，请参阅[快速入门指南](docs/howto/getting_started.md)。

### 使用开源社区的 brick

Compute Flow 提供了预打包的 brick，帮助开发者快速上手。我们相信，开发者分享成果有助于打造更智能的应用。因此，Compute Flow 从设计之初就致力于让社区开发的 brick
易于使用。

要添加社区 Git 仓库并安装其中一个 brick 模型，请运行：

```bash
compute-flow registry add https://example.com/team/example-tts.git
compute-flow update
compute-flow model install tts:example-tts example-model
```

请将示例 URL、brick 标识符和模型名称替换为 brick 作者提供的相应内容。更多信息请参阅[添加第三方 brick 指南](docs/howto/adding_3p_bricks.md)。

### 相关资料

#### [操作指南](docs/howto/)

| 文档                                                             | 说明                                         |
| ---------------------------------------------------------------- | -------------------------------------------- |
| [**快速入门**](docs/howto/getting_started.md)                    | 在本机和 Android™ 目标平台上构建并运行 brick |
| [**添加第三方 brick**](docs/howto/adding_3p_bricks.md)           | 注册并安装 Git 仓库中的 brick                |
| [**开发 brick**](docs/howto/developing_bricks.md)                | 准备模型包并实现 brick                       |
| [**从源码构建**](docs/howto/building_from_source.md)             | 使用上游源码构建依赖项                       |
| [**运行 llama.cpp 模型**](docs/howto/running_llamacpp_models.md) | 导入并运行与 llama.cpp 兼容的 GGUF 模型      |
| [**更新清单配置**](docs/howto/updating_manifest_config.md)       | 无需重新加载模型即可修改运行时配置值         |

#### [示例](docs/examples/)

| 文档                                                                                                   | 说明                                     |
| ------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| [**最小 C++ 应用**](docs/examples/cpp/minimal_example.md)                                              | 使用打包好的 SDK 构建最小应用            |
| [**Python 翻译流水线**](docs/examples/python/audio_to_audio_translation.md)                            | 使用 Python 实现语音到语音的翻译         |
| [**使用 Python 生成播客**](docs/examples/python/generating_a_podcast.md)                               | 从文本文件生成双人播客                   |
| [**使用 Python 转录音频并标注说话人**](docs/examples/python/transcribing_audio_with_speaker_labels.md) | 将音频转录为带有说话人标签和时间戳的文本 |

其他开发、设计和参考文档请参阅[文档索引](docs/README.md)。

## 支持

如需帮助或提供反馈，请通过项目的问题跟踪系统提出，或直接联系维护者。

## 参与贡献

Compute Flow 正在快速发展，因此目前暂不接受对 `compute_flow`
核心代码的贡献。欢迎提出想法、功能需求、问题报告及其他反馈。详情请参阅[贡献指南](CONTRIBUTING.md)。

## 作者

- **Gian Marco Iodice**（*gianmarco.iodice@arm.com*），杰出工程师（CE-ML），剑桥

- **Jakub Sujak**（*jakub.sujak@arm.com*），高级软件工程师（CE-ML），剑桥

- **Yunus Kalkan**（*yunus.kalkan@arm.com*），高级软件工程师（STE），剑桥

- **Adnan AlSinan**（*adnan.alsinan@arm.com*），高级软件工程师（CE-ML），剑桥

## 致谢

我们衷心感谢以下人员（按姓名字母顺序排列）为本项目作出的贡献、给予的支持和提供的宝贵反馈：**Abbey Noble**、**Aliyha Ajmal**、**Anand
Kakad**、**Anitha Raj**、**Anton Kachatkou**、**Aude Vuilliomenet**、**Carey Williams**、**Charlotte
Chen**、**Declan Cox**、**Evie Wright**、**Govind Narasimman**、**Joe Alderson**、**Kshitij
Sisodia**、**Marlon Jacobs**、**Maurice De Vries**、**Murad Almakhzangi**、**Nina Drozd**、**Ronan
Naughton** 和 **Sicong Li**。

## 商标

本项目所用名称的规范写法、法律声明和使用规则，请参阅[商标参考文档](TRADEMARKS/README.md)。

## 许可证

Compute Flow 采用 [Apache 许可证 2.0](LICENSES/Apache-2.0.txt)。

第三方组件仍受各自许可证的约束。直接依赖项清单请参阅[第三方声明](THIRD_PARTY_NOTICES.md)。

## 引用

如果你在工作中使用了本项目，请按以下格式引用：

```bibtex
@software{compute-flow,
  author  = {Iodice, Gian Marco and Sujak, Jakub and Kalkan, Yunus and AlSinan, Adnan},
  title   = {compute-flow: a cross-platform SDK to streamline the creation of AI applications at the Edge},
  url     = {https://github.com/ARM-software/ComputeFlow},
  year    = {2026},
  organization = {Arm},
  address = {Cambridge, United Kingdom},
}
```

## 项目状态

⚠️ 本项目目前处于原型阶段，开发工作十分活跃，API 和功能可能发生变化。欢迎提出想法、报告问题并提供反馈！
