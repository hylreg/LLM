# 语音转文字（Speech-to-Text）模块

本模块使用 OpenAI Whisper 模型实现语音转文字功能，支持多种音频格式和麦克风录音。

## 功能特性

- ✅ 支持多种音频格式（WAV、MP3、M4A、FLAC、OGG等）
- ✅ 支持从文件转录
- ✅ 支持麦克风实时录音并转录
- ✅ 支持多语言（中文、英文等）
- ✅ 支持自动语言检测
- ✅ 提供多种模型选择（tiny/base/small/medium/large）
- ✅ 可集成到LangChain工作流

## 技术选型

- **模型**：OpenAI Whisper（通过 transformers 库）
- **优势**：
  1. 开源免费，无需API密钥
  2. 支持多语言（包括中文），准确度高
  3. 可在本地运行，保护隐私
  4. 支持多种音频格式
  5. 项目已包含 transformers 和 torch 依赖

## 快速开始

### 1. 安装依赖

```bash
# 在项目根目录执行
cd /home/lab/Projects/LLM
uv sync
```

### 2. 使用命令行工具

```bash
# 运行命令行工具（交互式）
uv run python -m demos.speech_to_text.cli

# 或从文件转录
uv run python -m demos.speech_to_text.cli audio.wav
```

### 3. 在代码中使用

```python
from demos.speech_to_text import SpeechToText

# 创建转录器
stt = SpeechToText(model_name="openai/whisper-small")

# 从文件转录
result = stt.transcribe("audio.wav", language="zh")
print(result)

# 从麦克风录音并转录
result = stt.transcribe_from_microphone(duration=5.0, language="zh")
print(result)
```

## 使用示例

### 示例1：基础文件转录

```python
from demos.speech_to_text import SpeechToText

stt = SpeechToText(model_name="openai/whisper-small")
result = stt.transcribe("audio.wav", language="zh")
print(result)
```

### 示例2：麦克风录音并转录

```python
from demos.speech_to_text import SpeechToText

stt = SpeechToText(model_name="openai/whisper-small")
result = stt.transcribe_from_microphone(duration=5.0, language="zh")
print(result)
```

### 示例3：自动检测语言

```python
from demos.speech_to_text import SpeechToText

stt = SpeechToText(model_name="openai/whisper-small")
result = stt.transcribe("audio.wav", language="auto")
print(result)
```

### 示例4：运行示例脚本

```bash
uv run python -m demos.speech_to_text.examples
```

## 模型选择

| 模型 | 参数 | 速度 | 准确度 | 内存 | 推荐场景 |
|------|------|------|--------|------|----------|
| tiny | ~39M | 最快 | 较低 | ~1GB | 实时场景，快速测试 |
| base | ~74M | 快 | 中等 | ~1-2GB | 平衡选择 |
| **small** | **~244M** | **中等** | **较高** | **~2-4GB** | **推荐使用** |
| medium | ~769M | 较慢 | 高 | ~4-8GB | 高准确度需求 |
| large | ~1550M | 慢 | 最高 | ~8GB+ | 最高准确度需求 |

## API 参考

### SpeechToText 类

#### `__init__(model_name="openai/whisper-small", device=None)`

初始化语音转文字模型。

**参数**：
- `model_name` (str): Whisper模型名称
- `device` (str, optional): 计算设备，"cuda" 或 "cpu"，None 自动选择

#### `transcribe(audio_path, language="zh", return_timestamps=False)`

将音频文件转换为文字。

**参数**：
- `audio_path` (str): 音频文件路径
- `language` (str): 语言代码，"zh"、"en"、"auto"等
- `return_timestamps` (bool): 是否返回时间戳

**返回**：转录的文字字符串

#### `transcribe_from_microphone(duration=5.0, language="zh", sample_rate=16000)`

从麦克风录音并直接转录为文字。

**参数**：
- `duration` (float): 录制时长（秒）
- `language` (str): 语言代码
- `sample_rate` (int): 采样率，默认16000

**返回**：转录的文字字符串

#### `record_from_microphone(duration=5.0, sample_rate=16000, channels=1)`

从麦克风录制音频。

**参数**：
- `duration` (float): 录制时长（秒）
- `sample_rate` (int): 采样率
- `channels` (int): 声道数，1=单声道，2=立体声

**返回**：numpy array格式的音频数据

#### `transcribe_stream(audio_data, sample_rate=16000)`

实时流式音频转文字。

**参数**：
- `audio_data`: 音频数据（numpy array 或 bytes）
- `sample_rate` (int): 采样率

**返回**：转录的文字字符串

## 注意事项

1. **首次运行**：首次使用时会自动下载 Whisper 模型，可能需要一些时间
2. **模型存储**：模型会下载到 `~/.cache/huggingface/` 目录
3. **GPU支持**：如果有GPU，会自动使用GPU加速（需要安装CUDA版本的PyTorch）
4. **麦克风功能**：需要系统有可用的麦克风设备，Linux系统可能需要安装 `portaudio` 相关库

## 常见问题

### Q: 模型下载很慢怎么办？

A: 可以设置HuggingFace镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 内存不足怎么办？

A: 使用更小的模型（tiny或base），或者使用CPU而不是GPU。

### Q: 支持哪些音频格式？

A: 支持所有librosa支持的格式，包括WAV、MP3、M4A、FLAC、OGG等。

### Q: 可以实时转录吗？

A: 可以，使用 `transcribe_stream()` 方法处理音频流数据，或使用 `transcribe_from_microphone()` 进行实时录音转录。

## 文件结构

```
demos/speech_to_text/
├── __init__.py          # 模块初始化，导出SpeechToText类
├── core.py              # 核心实现
├── cli.py               # 命令行工具
├── examples.py          # 使用示例
└── README.md            # 本文档
```

## 许可证

本项目遵循 MIT 许可证。

