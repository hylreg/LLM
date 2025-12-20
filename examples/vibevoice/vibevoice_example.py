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
    print("\n方法 1：从 GitHub 安装（推荐）")
    print("-" * 60)
    print("git clone https://github.com/microsoft/VibeVoice.git")
    print("cd VibeVoice/")
    print("pip install -e .")
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


def run_basic_example():
    """运行基础示例"""
    if not check_vibevoice_installed():
        install_instructions()
        return
    
    try:
        # 尝试导入 VibeVoice 相关模块
        # 注意：实际导入路径可能因版本而异
        from vibevoice import VibeVoiceRealtime
        import torch
        
        print("正在加载 VibeVoice-Realtime-0.5B 模型...")
        print("这可能需要几分钟时间，请耐心等待...")
        
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


def run_streaming_example():
    """运行流式输入示例"""
    if not check_vibevoice_installed():
        install_instructions()
        return
    
    try:
        from vibevoice import VibeVoiceRealtime
        import torch
        
        print("正在加载 VibeVoice-Realtime-0.5B 模型（流式模式）...")
        
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
  uv run examples/vibevoice/vibevoice_example.py              # 运行基础示例
  uv run examples/vibevoice/vibevoice_example.py --streaming  # 运行流式示例
  uv run examples/vibevoice/vibevoice_example.py --demo       # 显示官方演示脚本使用方法

或使用传统 Python 方式:
  python examples/vibevoice/vibevoice_example.py              # 运行基础示例
  python examples/vibevoice/vibevoice_example.py --streaming  # 运行流式示例
  python examples/vibevoice/vibevoice_example.py --demo       # 显示官方演示脚本使用方法
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
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo_script()
    elif args.streaming:
        run_streaming_example()
    else:
        run_basic_example()


if __name__ == "__main__":
    main()

