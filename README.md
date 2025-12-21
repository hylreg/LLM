# LLM 项目

一个基于 Python 和 LangChain 的大语言模型集成项目，演示如何集成 Silicon Flow（硅基流动）和 Ollama 等不同的大语言模型服务。

## 项目简介

本项目展示了如何在 Python 项目中使用 LangChain 框架集成多种大语言模型服务：

- **Silicon Flow（硅基流动）**：提供多种大语言模型 API 服务的平台，兼容 OpenAI API 规范
- **Ollama**：本地运行大语言模型的工具，无需网络连接即可使用
- **VibeVoice-Realtime-0.5B**：微软开发的轻量级实时文本转语音（TTS）模型

项目提供了完整的示例代码，包括基础调用、流式输出、结构化输出、文本转语音等功能。

## 项目结构

```
.
├── main.py                    # 主程序入口
├── pyproject.toml             # 项目配置文件（使用 uv 管理依赖）
├── README.md                  # 项目说明文档
├── test_langchain/           # LangChain 集成示例模块
│   ├── README.md             # 示例代码详细说明
│   ├── siliconflow_example.py        # Silicon Flow 模型集成示例
│   ├── ollama_example.py              # Ollama 本地模型集成示例
│   ├── streaming_output_example.py    # 流式输出示例
│   └── structured_output_example.py   # 结构化输出示例
├── examples/                 # 其他示例模块
│   └── vibevoice/           # VibeVoice-Realtime-0.5B 示例
│       ├── README.md        # VibeVoice 使用指南
│       └── vibevoice_example.py  # VibeVoice 示例代码
└── uv.lock                    # 依赖锁定文件
```

## 快速开始

### 环境要求

- Python 3.10 或更高版本
- [uv](https://github.com/astral-sh/uv)（推荐的包管理工具）

### 安装依赖

```bash
uv sync
```

### 环境配置

#### Silicon Flow

设置环境变量（使用 Silicon Flow 示例时需要）：

```bash
export SILICONFLOW_API_KEY=your_actual_api_key_here
export SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1/  # 可选
```

#### Ollama

安装并运行 Ollama 服务，然后下载模型：

```bash
ollama pull qwen3:0.6b
```

#### VibeVoice-Realtime-0.5B

安装 VibeVoice 包（详细说明请参考 [examples/vibevoice/README.md](examples/vibevoice/README.md)）：

```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice/
pip install -e .
```

### 运行示例

**推荐使用 uv 运行**（推荐）：

```bash
# 运行主程序
uv run main.py

# 运行 LangChain 示例（详细说明请参考 test_langchain/README.md）
uv run test_langchain/siliconflow_example.py
uv run test_langchain/ollama_example.py
uv run test_langchain/streaming_output_example.py
uv run test_langchain/structured_output_example.py

# 运行 VibeVoice TTS 示例（详细说明请参考 examples/vibevoice/README.md）
# 先下载模型（推荐，约 1GB）
uv run examples/vibevoice/vibevoice_example.py --download-only

# 下载并运行基础示例
uv run examples/vibevoice/vibevoice_example.py --download

# 运行基础示例（首次运行会自动下载模型）
uv run examples/vibevoice/vibevoice_example.py

# 运行流式示例
uv run examples/vibevoice/vibevoice_example.py --streaming
```

**或使用传统 Python 方式**：

```bash
# 运行主程序
python main.py

# 运行 LangChain 示例（详细说明请参考 test_langchain/README.md）
python test_langchain/siliconflow_example.py
python test_langchain/ollama_example.py
python test_langchain/streaming_output_example.py
python test_langchain/structured_output_example.py

# 运行 VibeVoice TTS 示例（详细说明请参考 examples/vibevoice/README.md）
# 先下载模型（推荐，约 1GB）
python examples/vibevoice/vibevoice_example.py --download-only

# 下载并运行基础示例
python examples/vibevoice/vibevoice_example.py --download

# 运行基础示例（首次运行会自动下载模型）
python examples/vibevoice/vibevoice_example.py

# 运行流式示例
python examples/vibevoice/vibevoice_example.py --streaming
```

## 依赖说明

- `langchain>=1.1.3` - LangChain 核心框架
- `langchain-openai>=1.1.3` - OpenAI 兼容接口支持
- `langchain-ollama>=1.0.1` - Ollama 本地模型支持
- `langchain-anthropic>=1.3.0` - Anthropic 模型支持
- `transformers>=4.40.0` - Hugging Face Transformers（用于 VibeVoice）
- `torch>=2.0.0` - PyTorch（用于 VibeVoice）
- `soundfile>=0.12.0` - 音频文件处理（用于 VibeVoice）
- `huggingface_hub>=0.20.0` - Hugging Face Hub（用于下载模型）

## 更多信息

详细的使用说明、代码示例和技术实现请参考 [test_langchain/README.md](test_langchain/README.md)。

相关资源：
- [Silicon Flow 官网](https://www.siliconflow.cn/)
- [Ollama 官网](https://ollama.com/)
- [LangChain 文档](https://python.langchain.com/)
- [VibeVoice GitHub](https://github.com/microsoft/VibeVoice)
- [VibeVoice-Realtime-0.5B 模型页面](https://hf-mirror.com/microsoft/VibeVoice-Realtime-0.5B)
