# LangChain 集成示例

本目录包含了使用 LangChain 集成不同大语言模型的完整示例代码，涵盖了多种使用场景。

## 文件说明

- **siliconflow_example.py** - 使用 LangChain 集成 Silicon Flow 模型的示例
- **ollama_example.py** - 使用 LangChain 集成本地 Ollama 模型的示例
- **structured_output_example.py** - 展示如何使用结构化输出的示例
- **streaming_output_example.py** - 展示如何实现流式输出的示例

## 环境配置

### Silicon Flow

对于 Silicon Flow 相关示例（`siliconflow_example.py`、`structured_output_example.py`、`streaming_output_example.py`），需要设置环境变量：

```bash
export SILICONFLOW_API_KEY=your_siliconflow_api_key
export SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1/  # 可选，默认值
```

**获取 API 密钥**：访问 [Silicon Flow 官网](https://www.siliconflow.cn/)，注册账号后可在控制台获取。

### Ollama

对于 Ollama 示例（`ollama_example.py`），需要：

1. **安装 Ollama**：访问 [Ollama 官网](https://ollama.com/) 获取安装方法
2. **启动服务**：Ollama 安装后通常会自动启动，也可手动运行 `ollama serve`
3. **下载模型**：
   ```bash
   ollama pull qwen3:0.6b
   ```

**查看可用模型**：`ollama list`

## 示例使用说明

### 1. Silicon Flow 基础示例

**文件**：`siliconflow_example.py`

演示如何使用 LangChain 的 `init_chat_model` 函数集成 Silicon Flow 模型。

```bash
python test_langchain/siliconflow_example.py
```

**代码示例**：

```python
from test_langchain.siliconflow_example import create_siliconflow_model
from langchain_core.messages import HumanMessage

model = create_siliconflow_model()
messages = [HumanMessage(content="你好，请简单介绍一下你自己")]
response = model.invoke(messages)
print(response.content)
```

### 2. Ollama 本地模型示例

**文件**：`ollama_example.py`

演示如何使用 LangChain 集成本地运行的 Ollama 模型，包括简单调用、流式输出、提示模板和多轮对话等功能。

```bash
# 使用默认模型 (qwen3:0.6b)
python test_langchain/ollama_example.py

# 使用指定模型
python test_langchain/ollama_example.py llama3
```

**代码示例**：

```python
from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen3:0.6b", temperature=0.7)
response = model.invoke("请简单介绍一下Python编程语言")
print(response.content)
```

### 3. 流式输出示例

**文件**：`streaming_output_example.py`

演示如何实现流式输出，模拟打字机效果逐步显示模型的回复。

```bash
# 查看演示
python test_langchain/streaming_output_example.py

# 进入交互模式
python test_langchain/streaming_output_example.py interactive
```

**代码示例**：

```python
from test_langchain.streaming_output_example import create_siliconflow_model
from langchain_core.messages import HumanMessage

model = create_siliconflow_model()
messages = [HumanMessage(content="请写一首关于人工智能的五言诗")]
for chunk in model.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### 4. 结构化输出示例

**文件**：`structured_output_example.py`

演示如何使用 Pydantic 模型从非结构化文本中提取结构化信息。

```bash
python test_langchain/structured_output_example.py
```

**代码示例**：

```python
from test_langchain.structured_output_example import extract_person_info

text = "张三今年30岁，他擅长Python和Go语言。"
result = extract_person_info(text)
print(f"姓名: {result.name}, 年龄: {result.age}, 技能: {result.skills}")
```

## 技术实现

### Silicon Flow 模型集成

Silicon Flow 兼容 OpenAI API 规范，使用 LangChain 的 `init_chat_model` 函数：

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key="your_api_key",
    temperature=0.7
)
```

**优势**：
- LangChain 原生支持，可与其他 LangChain 组件无缝集成
- 兼容 OpenAI API，使用方式简单
- 支持流式输出、结构化输出等高级功能

### Ollama 模型集成

使用 `ChatOllama` 类集成本地模型：

```python
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="qwen3:0.6b",
    temperature=0.7,
    num_predict=256
)
```

**特点**：
- 本地运行，无需网络连接
- 不需要 API 密钥
- 支持多种开源模型

### 结构化输出

使用 `with_structured_output` 方法结合 Pydantic 模型：

```python
from pydantic import BaseModel, Field
from typing import List

class PersonInfo(BaseModel):
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄")
    skills: List[str] = Field(description="技能列表")

structured_llm = model.with_structured_output(PersonInfo)
result = structured_llm.invoke("从文本中提取信息...")
```

### 流式输出

使用 `stream` 方法实现流式输出：

```python
for chunk in model.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

## 可用模型

### Silicon Flow 模型

- `Qwen/Qwen3-8B` - 通义千问 3 8B 模型（示例中默认使用）
- `deepseek-ai/DeepSeek-V3` - DeepSeek V3 模型
- `Qwen/Qwen2.5-72B-Instruct` - 通义千问 2.5 72B 指令模型
- `THUDM/glm-4-9b-chat` - GLM-4 9B 对话模型

更多模型信息请访问 [Silicon Flow 官网](https://www.siliconflow.cn/)。

### Ollama 模型

- `qwen3:0.6b` - 通义千问 3 0.6B 模型（示例中默认使用）
- `llama3` - Meta Llama 3 模型
- `mistral` - Mistral 模型
- `gemma` - Google Gemma 模型

更多模型信息请访问 [Ollama 官网](https://ollama.com/)。

## 常见问题

**Q: 流式输出不显示怎么办？**  
A: 确保在打印时使用 `flush=True` 参数，并检查终端是否支持实时输出。

**Q: 如何查看可用的 Ollama 模型？**  
A: 运行 `ollama list` 命令。

**Q: 如何下载新的 Ollama 模型？**  
A: 运行 `ollama pull <model_name>` 命令。

## 更多资源

- [LangChain 官方文档](https://python.langchain.com/)
- [Silicon Flow 文档](https://www.siliconflow.cn/)
- [Ollama 文档](https://ollama.com/)
