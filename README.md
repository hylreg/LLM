# LLM 项目

一个基于 Python 和 LangChain 的大语言模型集成项目，演示如何集成 Silicon Flow（硅基流动）和 Ollama 等不同的大语言模型服务。

## 项目简介

本项目展示了如何在 Python 项目中使用 LangChain 框架集成多种大语言模型服务：

- **Silicon Flow（硅基流动）**：提供多种大语言模型 API 服务的平台，兼容 OpenAI API 规范
- **Ollama**：本地运行大语言模型的工具，无需网络连接即可使用

项目提供了完整的示例代码，包括基础调用、流式输出、结构化输出等功能。

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

### 运行示例

```bash
# 运行主程序
python main.py

# 运行 LangChain 示例（详细说明请参考 test_langchain/README.md）
python test_langchain/siliconflow_example.py
python test_langchain/ollama_example.py
python test_langchain/streaming_output_example.py
python test_langchain/structured_output_example.py
```

## 依赖说明

- `langchain>=1.1.3` - LangChain 核心框架
- `langchain-openai>=1.1.3` - OpenAI 兼容接口支持
- `langchain-ollama>=1.0.1` - Ollama 本地模型支持
- `langchain-anthropic>=1.3.0` - Anthropic 模型支持

## 更多信息

详细的使用说明、代码示例和技术实现请参考 [test_langchain/README.md](test_langchain/README.md)。

相关资源：
- [Silicon Flow 官网](https://www.siliconflow.cn/)
- [Ollama 官网](https://ollama.com/)
- [LangChain 文档](https://python.langchain.com/)
