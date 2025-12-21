# VibeVoice-Realtime-0.5B 使用指南

本指南介绍如何在项目中运行 Microsoft VibeVoice-Realtime-0.5B 模型。

## 模型简介

VibeVoice-Realtime-0.5B 是微软开发的轻量级实时文本转语音（TTS）模型，具有以下特点：

- **参数量**: 0.5B（部署友好）
- **实时 TTS**: ~300ms 首次可听延迟
- **流式文本输入**: 支持增量编码输入文本块
- **长文本生成**: 可生成约 10 分钟的语音
- **主要语言**: 英语（也支持部分其他语言）

## 安装步骤

### 方法 1：从 GitHub 安装（推荐）

```bash
# 克隆 VibeVoice 仓库
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice/

# 安装依赖
pip install -e .
```

### 方法 2：使用 Docker（需要 NVIDIA GPU）

```bash
# 启动 Docker 容器
sudo docker run --privileged --net=host --ipc=host \
  --ulimit memlock=-1:-1 --ulimit stack=-1:-1 \
  --gpus all --rm -it nvcr.io/nvidia/pytorch:24.07-py3

# 在容器中安装
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice/
pip install -e .
```

## 运行示例

### 1. 下载模型（推荐先下载）

**推荐使用镜像站点下载**（适合中国大陆用户，速度更快）：

```bash
# 使用镜像站点下载模型（推荐，约 2GB）
uv run examples/vibevoice/vibevoice_example.py --download-only --mirror

# 使用镜像站点下载并运行基础示例
uv run examples/vibevoice/vibevoice_example.py --download --mirror

# 使用镜像站点下载并运行流式示例
uv run examples/vibevoice/vibevoice_example.py --download --streaming --mirror
```

**或使用官方站点下载**：

```bash
# 仅下载模型（约 2GB，首次运行建议先下载）
uv run examples/vibevoice/vibevoice_example.py --download-only

# 下载模型并运行基础示例
uv run examples/vibevoice/vibevoice_example.py --download

# 下载模型并运行流式示例
uv run examples/vibevoice/vibevoice_example.py --download --streaming
```

**或使用传统 Python 方式**：

```bash
# 使用镜像站点下载（推荐）
python examples/vibevoice/vibevoice_example.py --download-only --mirror

# 仅下载模型
python examples/vibevoice/vibevoice_example.py --download-only

# 下载模型并运行基础示例
python examples/vibevoice/vibevoice_example.py --download

# 下载模型并运行流式示例
python examples/vibevoice/vibevoice_example.py --download --streaming

# 查看模型路径信息
python examples/vibevoice/vibevoice_example.py --show-path
```

**镜像站点说明**：
- 镜像站点：https://hf-mirror.com
- 适合中国大陆用户，下载速度更快
- 使用 `--mirror` 参数会自动从镜像站点下载
- 也可以通过环境变量设置：`export HF_ENDPOINT=https://hf-mirror.com`

### 2. 查看模型路径

```bash
# 查看模型下载路径和缓存信息
uv run examples/vibevoice/vibevoice_example.py --show-path

# 或使用传统 Python 方式
python examples/vibevoice/vibevoice_example.py --show-path
```

**模型默认存储位置**：
- **macOS/Linux**: `~/.cache/huggingface/hub/models--microsoft--VibeVoice-Realtime-0.5B`
- **Windows**: `C:\Users\<用户名>\.cache\huggingface\hub\models--microsoft--VibeVoice-Realtime-0.5B`

**自定义缓存路径**：
可以通过设置环境变量来更改缓存目录：
```bash
export HF_HOME=/path/to/your/cache
```

### 3. 运行示例（模型会自动下载）

**推荐使用 uv 运行**（推荐）：

```bash
# 从项目根目录运行基础示例（首次运行会自动下载模型）
uv run examples/vibevoice/vibevoice_example.py

# 运行流式输入示例
uv run examples/vibevoice/vibevoice_example.py --streaming

# 查看官方演示脚本使用方法
uv run examples/vibevoice/vibevoice_example.py --demo
```

**或使用传统 Python 方式**：

```bash
# 从项目根目录运行基础示例（首次运行会自动下载模型）
python examples/vibevoice/vibevoice_example.py

# 运行流式输入示例
python examples/vibevoice/vibevoice_example.py --streaming

# 查看官方演示脚本使用方法
python examples/vibevoice/vibevoice_example.py --demo
```

### 4. 使用官方演示脚本

```bash
# 在 VibeVoice 目录中运行
cd VibeVoice/
python demo/vibevoice_realtime_demo.py --model_path microsoft/VibeVoice-Realtime-0.5B
```

## 系统要求

- **Python**: 3.10 或更高版本
- **GPU**: 推荐使用 NVIDIA GPU（CUDA 支持）
- **内存**: 至少 4GB RAM（使用 GPU 时）
- **磁盘空间**: 至少 2GB 可用空间（用于模型文件）

## 依赖项

项目已包含以下基础依赖：
- `transformers>=4.40.0`
- `torch>=2.0.0`
- `soundfile>=0.12.0`
- `numpy>=1.24.0`
- `scipy>=1.10.0`
- `huggingface_hub>=0.20.0` - 用于下载模型

安装 VibeVoice 包时会自动安装其他必需的依赖。

## 使用示例代码

### 基础文本转语音

```python
from vibevoice import VibeVoiceRealtime
import torch
import soundfile as sf

# 加载模型
model = VibeVoiceRealtime.from_pretrained(
    "microsoft/VibeVoice-Realtime-0.5B",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 生成语音
text = "Hello, this is a test of the VibeVoice Realtime model."
audio = model.generate(text)

# 保存音频文件（24kHz 采样率）
sf.write("output.wav", audio, 24000)
```

### 流式文本输入

VibeVoice-Realtime 支持流式文本输入，可以在文本生成的同时开始语音合成，实现超低延迟。

## 模型存储位置

模型下载后会存储在 Hugging Face 的默认缓存目录中：

- **默认路径**: `~/.cache/huggingface/hub/`
- **模型路径**: `~/.cache/huggingface/hub/models--microsoft--VibeVoice-Realtime-0.5B/`

**查看模型路径**：
```bash
uv run examples/vibevoice/vibevoice_example.py --show-path
```

**自定义缓存路径**：
```bash
# 设置环境变量
export HF_HOME=/path/to/your/cache

# 或在代码中指定
from huggingface_hub import snapshot_download
snapshot_download(repo_id="microsoft/VibeVoice-Realtime-0.5B", cache_dir="/custom/path")
```

## 注意事项

1. **语言支持**: 模型主要针对英语优化，其他语言可能产生不可预测的结果
2. **首次运行**: 首次运行时会自动从 Hugging Face 下载模型文件，需要网络连接
3. **GPU 推荐**: 虽然可以在 CPU 上运行，但推荐使用 GPU 以获得更好的性能
4. **模型大小**: 模型文件约 2GB，下载和加载需要一些时间
5. **缓存管理**: 模型下载后会缓存在本地，下次使用时无需重新下载
6. **镜像站点**: 如果访问 Hugging Face 官方站点困难，建议使用 `--mirror` 参数从镜像站点下载

## 常见问题

### Q: 如何检查 GPU 是否可用？
A: 运行以下 Python 代码：
```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
```

### Q: 模型下载失败怎么办？
A: 可以尝试使用 Hugging Face 镜像站点，或手动下载模型文件。

### Q: 支持中文吗？
A: 模型主要针对英语训练，中文支持有限。如需中文 TTS，建议使用专门的中文模型。

## 相关资源

- **GitHub 仓库**: https://github.com/microsoft/VibeVoice
- **Hugging Face 模型**: https://hf-mirror.com/microsoft/VibeVoice-Realtime-0.5B
- **技术报告**: 参考模型页面上的技术报告链接
- **项目主页**: https://microsoft.github.io/VibeVoice/

## 许可证

MIT License

## 联系方式

如有问题或建议，请联系 Microsoft Research 团队。

