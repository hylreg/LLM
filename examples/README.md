# 示例目录

本目录包含各种模型和工具的示例代码。

## 目录结构

```
examples/
├── README.md              # 本文件
└── vibevoice/            # VibeVoice-Realtime-0.5B 示例
    ├── README.md         # VibeVoice 使用指南
    └── vibevoice_example.py  # VibeVoice 示例代码
```

## 可用示例

### VibeVoice-Realtime-0.5B

微软开发的轻量级实时文本转语音（TTS）模型示例。

**位置**: `examples/vibevoice/`

**快速开始**:
```bash
# 查看详细说明
cat examples/vibevoice/README.md

# 运行示例（推荐使用 uv）
uv run examples/vibevoice/vibevoice_example.py

# 或使用传统 Python 方式
python examples/vibevoice/vibevoice_example.py
```

**更多信息**: 请参考 [examples/vibevoice/README.md](vibevoice/README.md)

## 其他示例

- **LangChain 集成示例**: 请参考项目根目录下的 `test_langchain/` 文件夹

