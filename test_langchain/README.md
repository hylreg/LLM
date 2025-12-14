# LangChain 集成示例

这个目录包含了使用 LangChain 集成 SiliconFlow 模型的示例代码。

## 文件说明

- [siliconflow_example.py](file:///Users/admin/Projects/LLM/test_langchain/siliconflow_example.py) - 使用 LangChain 原生方式集成 SiliconFlow 模型的示例

## 使用方法

### 环境变量设置

在运行示例之前，需要设置以下环境变量：

```bash
export SILICONFLOW_API_KEY=your_siliconflow_api_key
export SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1/  # 可选，默认值
```

### 运行示例

```bash
python siliconflow_example.py
```

或者在其他代码中导入使用：

```python
from test_langchain.siliconflow_example import create_siliconflow_model

model = create_siliconflow_model()
# 现在可以将此模型用于 LangChain 的各种功能，如 create_agent 等
```

## 代码说明

示例中使用了 LangChain 的 `init_chat_model` 函数来初始化 SiliconFlow 模型：

```python
model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url=base_url,
    api_key=api_key,
    temperature=0.7
)
```

这种方法的优势在于它是 LangChain 原生支持的方式，可以与其他 LangChain 组件无缝集成。

## LangChain集成示例详细介绍

### 使用 `init_chat_model` 函数（推荐）

这是LangChain较新版本提供的统一接口，可以直接指定模型提供商：

```python
from langchain.chat_models import init_chat_model

# 初始化硅基流动的模型
model = init_chat_model(
    model="Qwen/Qwen3-8B",           # 指定具体模型
    model_provider="openai",          # 指定提供商（硅基流动兼容OpenAI API）
    base_url="https://api.siliconflow.cn/v1/",  # 硅基流动API地址
    api_key="your_siliconflow_api_key"          # 你的API密钥
)

# 使用模型创建agent
from langchain.agents import create_agent
agent = create_agent(
    llm=model,
    # 其他参数...
)
```

### 使用 `ChatOpenAI` 类

由于硅基流动兼容OpenAI API，您可以直接使用LangChain的ChatOpenAI类：

```python
from langchain_openai import ChatOpenAI

# 配置硅基流动模型
llm = ChatOpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key="your_siliconflow_api_key",
    model="Qwen/Qwen3-8B",
    temperature=0.7
)

# 使用该模型创建agent
from langchain.agents import create_agent
agent = create_agent(
    llm=llm,
    # 其他参数...
)
```

## Silicon Flow模型使用说明

Silicon Flow提供了多种大语言模型，包括但不限于:
- Qwen/Qwen3-8B (默认使用)
- deepseek-ai/DeepSeek-V3
- Qwen/Qwen2.5-72B-Instruct
- THUDM/glm-4-9b-chat

这些模型都可以通过上述两种方式与LangChain原生集成。

### 在代码中使用模型

```python
from langchain.models import get_silicon_flow_model, QWEN3_8B

# 获取模型实例（默认使用Qwen/Qwen3-8B）
model = get_silicon_flow_model()

# 或者显式指定模型
model = get_silicon_flow_model(QWEN3_8B, temperature=0.7)

# 调用模型
response = model.invoke("你好，请简单介绍一下你自己")
print(response.content)
```

### 使用对话链

```python
from langchain.models import get_silicon_flow_model
from langchain.chains import chat_with_model
from langchain_core.messages import HumanMessage, AIMessage

model = get_silicon_flow_model()
history = [
    HumanMessage(content="你能告诉我什么是大语言模型吗？"),
    AIMessage(content="大语言模型是一种基于大量文本数据训练的深度学习模型...")
]

response = chat_with_model(model, "那它们有什么应用场景呢？", history)
```