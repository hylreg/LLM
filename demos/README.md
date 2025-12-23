# Demos 目录

本目录包含各种演示和示例代码。

## 目录结构

```
demos/
├── main.py                    # LangChain Agent 示例
├── voice_assistant/           # 端到端语音大模型助手模块
│   ├── __init__.py           # 模块初始化
│   ├── core.py               # 核心实现
│   ├── cli.py                # 命令行工具
│   ├── examples.py           # 使用示例
│   └── README.md             # 详细文档
└── speech_to_text/            # 语音转文字功能模块
    ├── __init__.py           # 模块初始化
    ├── core.py               # 核心实现
    ├── cli.py                # 命令行工具
    ├── examples.py           # 使用示例
    └── README.md             # 详细文档
```

## 端到端语音大模型助手 ⭐

完整的语音对话系统，实现：**语音输入 → 大语言模型 → 语音输出**

### 快速开始

```bash
# 交互式模式（推荐）
uv run python -m demos.voice_assistant.cli

# 或运行示例
uv run python -m demos.voice_assistant.examples
```

### 在代码中使用

```python
from demos.voice_assistant import VoiceAssistant

assistant = VoiceAssistant(
    llm_provider="siliconflow",  # 或 "ollama"
    temperature=0.7
)

# 处理语音输入（从麦克风）
result = assistant.process_voice_input(
    audio_source="microphone",
    language="zh",
    play_response=True
)

print(f"用户输入: {result['user_text']}")
print(f"LLM回复: {result['llm_reply']}")
```

详细使用说明请参考 [voice_assistant/README.md](voice_assistant/README.md)

## 语音转文字功能

详细使用说明请参考 [speech_to_text/README.md](speech_to_text/README.md)

### 快速开始

```bash
# 使用命令行工具
uv run python -m demos.speech_to_text.cli

# 运行示例
uv run python -m demos.speech_to_text.examples
```

### 在代码中使用

```python
from demos.speech_to_text import SpeechToText

stt = SpeechToText(model_name="openai/whisper-small")
result = stt.transcribe("audio.wav", language="zh")
print(result)
```

