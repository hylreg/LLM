#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VibeVoice-Realtime-0.5B 示例程序

本程序演示如何使用 Microsoft VibeVoice-Realtime-0.5B 模型进行实时文本转语音。

注意：使用此模型需要先安装 VibeVoice 包：
1. git clone https://github.com/microsoft/VibeVoice.git
2. cd VibeVoice/
3. pip install -e .

或者使用官方提供的 Docker 镜像。
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def check_vibevoice_installed():
    """检查 VibeVoice 是否已安装"""
    try:
        import vibevoice
        return True
    except ImportError:
        return False


def install_instructions():
    """显示安装说明"""
    print("=" * 60)
    print("VibeVoice 未安装，请按照以下步骤安装：")
    print("=" * 60)
    print("\n为什么需要安装 VibeVoice 库？")
    print("-" * 60)
    print("虽然模型文件存储在 Hugging Face 上，但 VibeVoice-Realtime-0.5B")
    print("使用了自定义的模型架构（vibevoice_streaming），不能直接用")
    print("transformers.AutoModel 加载。需要安装 VibeVoice 库才能使用。")
    print("\n模型页面: https://hf-mirror.com/microsoft/VibeVoice-Realtime-0.5B")
    print("-" * 60)
    print("\n方法 1：从 GitHub 安装（推荐）")
    print("-" * 60)
    print("git clone https://github.com/microsoft/VibeVoice.git")
    print("cd VibeVoice/")
    print("pip install -e .")
    print("\n或者使用 uv（如果使用 uv 管理依赖）:")
    print("git clone https://github.com/microsoft/VibeVoice.git")
    print("cd VibeVoice/")
    print("uv pip install -e .")
    print("\n方法 2：使用 Docker（需要 NVIDIA GPU）")
    print("-" * 60)
    print("sudo docker run --privileged --net=host --ipc=host \\")
    print("  --ulimit memlock=-1:-1 --ulimit stack=-1:-1 \\")
    print("  --gpus all --rm -it nvcr.io/nvidia/pytorch:24.07-py3")
    print("\n然后在容器中执行：")
    print("git clone https://github.com/microsoft/VibeVoice.git")
    print("cd VibeVoice/")
    print("pip install -e .")
    print("\n更多信息请访问：https://github.com/microsoft/VibeVoice")
    print("=" * 60)


def try_load_with_transformers(model_name="microsoft/VibeVoice-Realtime-0.5B"):
    """尝试使用 transformers.AutoModel 加载模型（通常会失败）"""
    if not TRANSFORMERS_AVAILABLE:
        return False, "transformers 库未安装"
    
    try:
        from transformers import AutoModel
        import torch
        
        print("\n尝试使用 transformers.AutoModel 加载模型...")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        return True, model
    except Exception as e:
        error_msg = str(e)
        if "trust_remote_code" in error_msg.lower() or "code" in error_msg.lower():
            return False, "模型需要自定义代码（VibeVoice 库）才能加载"
        elif "vibevoice" in error_msg.lower():
            return False, "无法识别模型类型 'vibevoice_streaming'，需要 VibeVoice 库"
        else:
            return False, f"加载失败: {error_msg}"


def run_basic_example(download_first=False, use_mirror=False):
    """运行基础示例"""
    if not check_vibevoice_installed():
        # 尝试用 transformers 加载，展示为什么需要 VibeVoice 库
        print("\n" + "=" * 60)
        print("检测到 VibeVoice 库未安装")
        print("=" * 60)
        success, result = try_load_with_transformers()
        if not success:
            print(f"\n✗ 直接使用 transformers 加载失败: {result}")
            print("\n说明：此模型使用自定义架构，必须通过 VibeVoice 库加载")
        print()
        install_instructions()
        return
    
    # 如果需要先下载模型
    if download_first:
        is_downloaded, model_path = check_model_downloaded()
        if not is_downloaded:
            print("\n模型未下载，正在下载...")
            if not download_model(use_mirror=use_mirror):
                print("\n模型下载失败，将尝试在运行时自动下载")
        else:
            print("\n✓ 检测到模型已下载")
            if model_path:
                print(f"模型路径: {model_path}")
    
    try:
        # 尝试导入 VibeVoice 相关模块
        # 注意：实际导入路径可能因版本而异
        from vibevoice import VibeVoiceRealtime
        import torch
        
        print("\n正在加载 VibeVoice-Realtime-0.5B 模型...")
        is_downloaded, model_path = check_model_downloaded()
        if not download_first and not is_downloaded:
            print("首次运行将自动下载模型，这可能需要几分钟时间，请耐心等待...")
        else:
            print("正在从缓存加载模型...")
            if model_path:
                print(f"从路径加载: {model_path}")
        
        # 初始化模型
        model = VibeVoiceRealtime.from_pretrained(
            "microsoft/VibeVoice-Realtime-0.5B",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print("模型加载成功！")
        print(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # 示例文本
        text = "Hello, this is a test of the VibeVoice Realtime model. It can generate speech in real-time with low latency."
        
        print(f"\n正在生成语音: {text}")
        print("-" * 60)
        
        # 生成语音
        audio = model.generate(text)
        
        # 保存音频文件
        output_path = "output_vibevoice.wav"
        import soundfile as sf
        sf.write(output_path, audio, 24000)  # 24kHz 采样率
        
        print(f"\n语音已保存到: {output_path}")
        print("=" * 60)
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("\n请确保已正确安装 VibeVoice 包")
        install_instructions()
    except Exception as e:
        print(f"运行错误: {e}")
        print("\n请检查：")
        print("1. 是否已安装所有依赖")
        print("2. 是否有足够的 GPU 内存（如果使用 GPU）")
        print("3. 网络连接是否正常（首次运行需要下载模型）")


def run_streaming_example(download_first=False, use_mirror=False):
    """运行流式输入示例"""
    if not check_vibevoice_installed():
        # 尝试用 transformers 加载，展示为什么需要 VibeVoice 库
        print("\n" + "=" * 60)
        print("检测到 VibeVoice 库未安装")
        print("=" * 60)
        success, result = try_load_with_transformers()
        if not success:
            print(f"\n✗ 直接使用 transformers 加载失败: {result}")
            print("\n说明：此模型使用自定义架构，必须通过 VibeVoice 库加载")
        print()
        install_instructions()
        return
    
    # 如果需要先下载模型
    if download_first:
        is_downloaded, model_path = check_model_downloaded()
        if not is_downloaded:
            print("\n模型未下载，正在下载...")
            if not download_model(use_mirror=use_mirror):
                print("\n模型下载失败，将尝试在运行时自动下载")
        else:
            print("\n✓ 检测到模型已下载")
            if model_path:
                print(f"模型路径: {model_path}")
    
    try:
        from vibevoice import VibeVoiceRealtime
        import torch
        
        print("\n正在加载 VibeVoice-Realtime-0.5B 模型（流式模式）...")
        is_downloaded, model_path = check_model_downloaded()
        if not download_first and not is_downloaded:
            print("首次运行将自动下载模型，这可能需要几分钟时间，请耐心等待...")
        else:
            print("正在从缓存加载模型...")
            if model_path:
                print(f"从路径加载: {model_path}")
        
        model = VibeVoiceRealtime.from_pretrained(
            "microsoft/VibeVoice-Realtime-0.5B",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print("模型加载成功！")
        print("\n流式文本输入示例：")
        print("-" * 60)
        
        # 模拟流式文本输入
        text_chunks = [
            "Hello, ",
            "this is ",
            "a streaming ",
            "text-to-speech ",
            "demonstration."
        ]
        
        print("正在处理流式文本...")
        for i, chunk in enumerate(text_chunks, 1):
            print(f"处理块 {i}/{len(text_chunks)}: {chunk}")
            # 在实际使用中，这里会调用流式生成方法
            # audio_chunk = model.generate_streaming(chunk)
        
        print("\n流式处理完成！")
        print("=" * 60)
        
    except ImportError as e:
        print(f"导入错误: {e}")
        install_instructions()
    except Exception as e:
        print(f"运行错误: {e}")


def get_hf_cache_dir():
    """获取 Hugging Face 缓存目录"""
    if not HF_HUB_AVAILABLE:
        return None
    
    try:
        from huggingface_hub import constants
        # 获取默认缓存目录
        cache_dir = constants.HF_HUB_CACHE
        return cache_dir
    except Exception:
        # 如果无法获取，使用默认路径
        import os
        home = os.path.expanduser("~")
        # Hugging Face 默认缓存路径
        default_cache = os.path.join(home, ".cache", "huggingface", "hub")
        return default_cache


def download_model(model_name="microsoft/VibeVoice-Realtime-0.5B", use_mirror=False):
    """主动下载模型
    
    Args:
        model_name: 模型名称
        use_mirror: 是否使用镜像站点 (hf-mirror.com)
    """
    print("=" * 60)
    print(f"正在下载模型: {model_name}")
    if use_mirror:
        print("使用镜像站点: https://hf-mirror.com")
    print("=" * 60)
    
    if not HF_HUB_AVAILABLE:
        print("\n错误: 需要安装 huggingface_hub 来下载模型")
        print("请运行: pip install huggingface_hub")
        print("或使用 uv: uv pip install huggingface_hub")
        return False
    
    # 设置镜像站点
    if use_mirror:
        import os
        # 设置环境变量使用镜像站点
        original_endpoint = os.environ.get("HF_ENDPOINT")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("\n已设置使用镜像站点: https://hf-mirror.com")
        print("（适合中国大陆用户，下载速度更快）")
    
    # 显示缓存目录信息
    cache_dir = get_hf_cache_dir()
    if cache_dir:
        print(f"\nHugging Face 缓存目录: {cache_dir}")
        print(f"模型将下载到: {cache_dir}/models--{model_name.replace('/', '--')}")
        print("-" * 60)
    
    try:
        print(f"\n开始下载模型到本地缓存...")
        print("这可能需要一些时间，请耐心等待...")
        print("（模型大小约 2GB）")
        print("-" * 60)
        
        # 使用 snapshot_download 下载整个模型仓库
        download_path = snapshot_download(
            repo_id=model_name,
            resume_download=True,
            local_files_only=False
        )
        
        # 恢复原始环境变量
        if use_mirror:
            if original_endpoint:
                os.environ["HF_ENDPOINT"] = original_endpoint
            else:
                os.environ.pop("HF_ENDPOINT", None)
        
        print(f"\n✓ 模型下载成功！")
        print(f"\n模型完整路径: {download_path}")
        if cache_dir:
            print(f"Hugging Face 缓存根目录: {cache_dir}")
        print("\n提示: 模型已缓存，下次使用时无需重新下载")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n请检查：")
        print("1. 网络连接是否正常")
        print("2. 是否有足够的磁盘空间（模型约 2GB）")
        if use_mirror:
            print("3. 是否可以访问镜像站点 https://hf-mirror.com")
        else:
            print("3. 是否可以访问 Hugging Face")
            print("   如果访问困难，可以尝试使用 --mirror 参数使用镜像站点")
        if cache_dir:
            print(f"4. 缓存目录是否有写入权限: {cache_dir}")
        print("=" * 60)
        return False
    finally:
        # 确保恢复环境变量
        if use_mirror:
            import os
            original_endpoint = os.environ.get("HF_ENDPOINT")
            if original_endpoint and original_endpoint != "https://hf-mirror.com":
                os.environ["HF_ENDPOINT"] = original_endpoint
            elif not original_endpoint:
                os.environ.pop("HF_ENDPOINT", None)


def check_model_downloaded(model_name="microsoft/VibeVoice-Realtime-0.5B"):
    """检查模型是否已下载，并返回模型路径"""
    if not HF_HUB_AVAILABLE:
        return False, None
    
    try:
        from huggingface_hub import try_to_load_from_cache, snapshot_download
        # 先尝试从缓存加载
        cache_path = try_to_load_from_cache(
            repo_id=model_name,
            filename="config.json"  # 检查配置文件是否存在
        )
        if cache_path is not None:
            # 如果配置文件存在，尝试获取完整路径
            try:
                model_path = snapshot_download(
                    repo_id=model_name,
                    local_files_only=True
                )
                return True, model_path
            except Exception:
                return True, None
        return False, None
    except Exception:
        return False, None


def show_model_path(model_name="microsoft/VibeVoice-Realtime-0.5B"):
    """显示模型路径信息"""
    print("=" * 60)
    print("模型路径信息")
    print("=" * 60)
    
    if not HF_HUB_AVAILABLE:
        print("\n错误: 需要安装 huggingface_hub")
        print("请运行: pip install huggingface_hub")
        return
    
    # 显示缓存目录
    cache_dir = get_hf_cache_dir()
    if cache_dir:
        print(f"\nHugging Face 缓存根目录: {cache_dir}")
        print(f"模型存储路径: {cache_dir}/models--{model_name.replace('/', '--')}")
    
    # 检查模型是否已下载
    is_downloaded, model_path = check_model_downloaded(model_name)
    
    if is_downloaded:
        print(f"\n✓ 模型已下载")
        if model_path:
            print(f"模型完整路径: {model_path}")
    else:
        print(f"\n✗ 模型未下载")
        print(f"运行以下命令下载: uv run examples/vibevoice/vibevoice_example.py --download-only")
    
    print("\n提示: 可以通过设置环境变量 HF_HOME 来更改缓存目录")
    print("例如: export HF_HOME=/path/to/your/cache")
    print("=" * 60)


def run_demo_script():
    """运行官方演示脚本"""
    print("=" * 60)
    print("运行官方演示脚本")
    print("=" * 60)
    print("\n请确保已安装 VibeVoice，然后运行：")
    print("-" * 60)
    print("python demo/vibevoice_realtime_demo.py --model_path microsoft/VibeVoice-Realtime-0.5B")
    print("\n或者如果已克隆到当前目录：")
    print("-" * 60)
    print("cd VibeVoice/")
    print("python demo/vibevoice_realtime_demo.py --model_path microsoft/VibeVoice-Realtime-0.5B")
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VibeVoice-Realtime-0.5B 示例程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法（推荐使用 uv）:
  uv run examples/vibevoice/vibevoice_example.py                    # 运行基础示例
  uv run examples/vibevoice/vibevoice_example.py --streaming      # 运行流式示例
  uv run examples/vibevoice/vibevoice_example.py --download-only   # 仅下载模型
  uv run examples/vibevoice/vibevoice_example.py --download-only --mirror  # 使用镜像站点下载（推荐）
  uv run examples/vibevoice/vibevoice_example.py --download        # 下载并运行
  uv run examples/vibevoice/vibevoice_example.py --show-path       # 显示模型路径
  uv run examples/vibevoice/vibevoice_example.py --demo            # 显示官方演示脚本使用方法

或使用传统 Python 方式:
  python examples/vibevoice/vibevoice_example.py                   # 运行基础示例
  python examples/vibevoice/vibevoice_example.py --streaming       # 运行流式示例
  python examples/vibevoice/vibevoice_example.py --download-only   # 仅下载模型
  python examples/vibevoice/vibevoice_example.py --download-only --mirror  # 使用镜像站点下载（推荐）
  python examples/vibevoice/vibevoice_example.py --download         # 下载并运行
  python examples/vibevoice/vibevoice_example.py --show-path         # 显示模型路径
  python examples/vibevoice/vibevoice_example.py --demo              # 显示官方演示脚本使用方法
        """
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="运行流式输入示例"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="显示官方演示脚本使用方法"
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="先下载模型再运行（如果模型已存在则跳过下载）"
    )
    
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="仅下载模型，不运行示例"
    )
    
    parser.add_argument(
        "--show-path",
        action="store_true",
        help="显示模型路径信息"
    )
    
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="使用镜像站点下载（hf-mirror.com，适合中国大陆用户）"
    )
    
    args = parser.parse_args()
    
    # 如果指定了使用镜像，设置环境变量
    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    if args.show_path:
        show_model_path()
    elif args.download_only:
        download_model(use_mirror=args.mirror)
    elif args.demo:
        run_demo_script()
    elif args.streaming:
        run_streaming_example(download_first=args.download, use_mirror=args.mirror)
    else:
        run_basic_example(download_first=args.download, use_mirror=args.mirror)


if __name__ == "__main__":
    main()

