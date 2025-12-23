# Demos 目录

本目录包含各种演示和示例代码。

## 目录结构

```
demos/
├── main.py                    # LangChain Agent 示例
└── speech_to_text/            # 语音转文字功能模块
    ├── __init__.py           # 模块初始化
    ├── core.py               # 核心实现
    ├── cli.py                # 命令行工具
    ├── examples.py           # 使用示例
    └── README.md             # 详细文档
```

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

