# 端到端语音大模型助手

完整的语音对话系统，实现语音输入 -> 大语言模型 -> 语音输出的完整流程。

## 功能特性

- ✅ **语音输入**：支持麦克风录音或音频文件
- ✅ **语音转文字**：使用 Whisper 模型
- ✅ **大语言模型**：支持 Silicon Flow 和 Ollama
- ✅ **文字转语音**：使用 edge-tts（支持中文）
- ✅ **语音播放**：自动播放生成的语音
- ✅ **多轮对话**：支持对话历史管理

## 系统流程

```
语音输入 (麦克风/文件)
    ↓
语音转文字 (Whisper)
    ↓
大语言模型 (LLM)
    ↓
文字转语音 (edge-tts)
    ↓
语音输出 (播放/保存)
```

## 安装依赖

```bash
# 确保所有依赖已安装
cd /home/lab/Projects/LLM
uv sync
```

如果 edge-tts 未自动安装，请手动安装：

```bash
uv pip install edge-tts
```

## 快速开始

### 方式一：交互式模式（推荐）

```bash
# 运行命令行工具（交互式）
uv run python -m demos.voice_assistant.cli

# 或使用示例脚本
uv run python -m demos.voice_assistant.examples
```

### 方式二：处理音频文件

```bash
# 处理音频文件
uv run python -m demos.voice_assistant.cli audio.wav
```

### 方式三：在代码中使用

```python
from demos.voice_assistant import VoiceAssistant

# 初始化助手
assistant = VoiceAssistant(
    stt_model_name="openai/whisper-small",
    llm_provider="siliconflow",  # 或 "ollama"
    temperature=0.7
)

# 处理语音输入
result = assistant.process_voice_input(
    audio_source="microphone",  # 或音频文件路径
    language="zh",
    play_response=True
)

print(f"用户输入: {result['user_text']}")
print(f"LLM回复: {result['llm_reply']}")
```

## 环境配置

### Silicon Flow（推荐用于中文）

```bash
export SILICONFLOW_API_KEY=your_api_key_here
export SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1/  # 可选
```

### Ollama（本地运行）

1. 安装 Ollama：https://ollama.com/
2. 下载模型：
   ```bash
   ollama pull qwen3:0.6b
   ```

## 使用示例

### 示例1：基础对话

```python
from demos.voice_assistant import VoiceAssistant

assistant = VoiceAssistant(
    llm_provider="siliconflow",
    temperature=0.7
)

# 从麦克风输入并处理
result = assistant.process_voice_input(
    audio_source="microphone",
    language="zh",
    play_response=True
)
```

### 示例2：多轮对话

```python
from demos.voice_assistant import VoiceAssistant
from langchain_core.messages import HumanMessage, AIMessage

assistant = VoiceAssistant(llm_provider="siliconflow")
conversation_history = []

# 第一轮
user_text = assistant.speech_to_text("microphone", language="zh")
reply = assistant.llm_response(user_text, conversation_history)
conversation_history.append(HumanMessage(content=user_text))
conversation_history.append(AIMessage(content=reply))

# 第二轮（保留历史）
user_text = assistant.speech_to_text("microphone", language="zh")
reply = assistant.llm_response(user_text, conversation_history)
```

### 示例3：从文件处理

```python
assistant = VoiceAssistant(llm_provider="siliconflow")

result = assistant.process_voice_input(
    audio_source="input_audio.wav",
    language="zh",
    play_response=True,
    save_audio=True  # 保存生成的语音
)
```

## 配置选项

### VoiceAssistant 参数

- `stt_model_name`: Whisper模型名称（默认："openai/whisper-small"）
- `llm_provider`: LLM提供商，"siliconflow" 或 "ollama"
- `llm_model_name`: LLM模型名称（可选，使用默认值）
- `tts_voice`: TTS语音（默认："zh-CN-XiaoxiaoNeural"）
- `temperature`: LLM温度参数（默认：0.7）

### TTS 语音选择

edge-tts 支持多种中文语音，可以查看可用语音：

```python
import edge_tts
import asyncio

async def list_voices():
    voices = await edge_tts.list_voices()
    chinese_voices = [v for v in voices if "zh-CN" in v["Locale"]]
    for voice in chinese_voices:
        print(f"{voice['Name']}: {voice['Locale']}")

asyncio.run(list_voices())
```

常用中文语音：
- `zh-CN-XiaoxiaoNeural` - 女声（推荐）
- `zh-CN-YunxiNeural` - 男声
- `zh-CN-XiaoyiNeural` - 女声
- `zh-CN-YunjianNeural` - 男声

## 完整工作流程

1. **语音输入**：用户通过麦克风说话或提供音频文件
2. **语音转文字**：Whisper 模型将语音转换为文字
3. **LLM处理**：大语言模型理解问题并生成回复
4. **文字转语音**：edge-tts 将回复转换为语音
5. **语音输出**：自动播放生成的语音

## 注意事项

1. **首次运行**：首次使用时会下载 Whisper 模型（约几百MB）
2. **API密钥**：使用 Silicon Flow 需要设置 API 密钥
3. **网络连接**：edge-tts 需要网络连接来获取语音
4. **麦克风权限**：使用麦克风需要系统麦克风权限
5. **语音质量**：edge-tts 是云端服务，质量较高但需要网络

## 故障排除

### 问题：edge-tts 安装失败

```bash
# 使用 pip 安装
pip install edge-tts

# 或使用 uv
uv pip install edge-tts
```

### 问题：无法播放音频

确保已安装 sounddevice 和 soundfile：

```bash
uv sync
```

### 问题：LLM连接失败

- 检查 API 密钥是否正确设置
- 检查网络连接
- 如果是 Ollama，确保服务正在运行

### 问题：麦克风无法使用

- Linux：可能需要安装 portaudio
- 检查系统麦克风权限
- 使用音频文件测试是否正常工作

## 扩展功能

### 自定义系统提示词

可以修改 LLM 的系统提示词来改变助手的行为：

```python
from langchain_core.messages import SystemMessage

system_msg = SystemMessage(content="你是一个友好的助手...")
conversation_history = [system_msg]
```

### 流式输出

支持 LLM 流式输出以获得更好的用户体验。

### 保存对话记录

可以保存对话历史以便后续分析或恢复。

## 相关资源

- [Whisper 文档](https://github.com/openai/whisper)
- [LangChain 文档](https://python.langchain.com/)
- [edge-tts 文档](https://github.com/rany2/edge-tts)
- [Silicon Flow](https://www.siliconflow.cn/)

