#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
语音转文字功能使用示例

本文件展示了如何使用 SpeechToText 模块进行语音转文字的各种场景。
"""

import sys
import os
from pathlib import Path

# 确保可以导入模块（从项目根目录或demos目录运行都能正常工作）
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
demos_dir = current_dir.parent

# 添加项目根目录和demos目录到路径
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(demos_dir) not in sys.path:
    sys.path.insert(0, str(demos_dir))

try:
    from demos.speech_to_text import SpeechToText
except ImportError:
    try:
        from speech_to_text import SpeechToText
    except ImportError:
        # 使用相对导入
        from . import SpeechToText


def example_file_transcription():
    """示例1：基础文件转录"""
    print("=" * 60)
    print("示例1：基础文件转录")
    print("=" * 60)
    
    # 创建转录器（使用small模型，推荐）
    stt = SpeechToText(model_name="openai/whisper-small")
    
    try:
        # 转录音频文件（请替换为你的音频文件路径）
        audio_file = "test_audio.wav"
        
        if os.path.exists(audio_file):
            result = stt.transcribe(audio_file, language="zh")
            print(f"\n音频文件: {audio_file}")
            print(f"转录结果: {result}\n")
        else:
            print(f"\n提示: 请准备音频文件 '{audio_file}' 进行测试")
            print("支持的格式: WAV, MP3, M4A, FLAC, OGG 等\n")
    finally:
        stt.cleanup()


def example_english_transcription():
    """示例2：英文转录"""
    print("=" * 60)
    print("示例2：英文转录")
    print("=" * 60)
    
    stt = SpeechToText(model_name="openai/whisper-small")
    
    try:
        audio_file = "test_english.wav"
        
        if os.path.exists(audio_file):
            result = stt.transcribe(audio_file, language="en")
            print(f"\n音频文件: {audio_file}")
            print(f"转录结果: {result}\n")
        else:
            print(f"\n提示: 请准备英文音频文件 '{audio_file}' 进行测试\n")
    finally:
        stt.cleanup()


def example_auto_detect_language():
    """示例3：自动检测语言"""
    print("=" * 60)
    print("示例3：自动检测语言")
    print("=" * 60)
    
    stt = SpeechToText(model_name="openai/whisper-small")
    
    try:
        audio_file = "test_audio.wav"
        
        if os.path.exists(audio_file):
            # 使用 "auto" 让模型自动检测语言
            result = stt.transcribe(audio_file, language="auto")
            print(f"\n音频文件: {audio_file}")
            print(f"转录结果: {result}\n")
        else:
            print(f"\n提示: 请准备音频文件 '{audio_file}' 进行测试\n")
    finally:
        stt.cleanup()


def example_microphone_transcription():
    """示例4：麦克风录音并转录"""
    print("=" * 60)
    print("示例4：麦克风录音并转录")
    print("=" * 60)
    
    stt = SpeechToText(model_name="openai/whisper-small")
    
    try:
        # 从麦克风录音并转录（默认5秒，中文）
        result = stt.transcribe_from_microphone(
            duration=5.0,
            language="zh"
        )
        
        print("\n转录结果:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        print()
    finally:
        stt.cleanup()


def example_different_models():
    """示例5：使用不同模型"""
    print("=" * 60)
    print("示例5：使用不同模型")
    print("=" * 60)
    
    audio_file = "test_audio.wav"
    
    if not os.path.exists(audio_file):
        print(f"提示: 请准备音频文件 '{audio_file}' 进行测试\n")
        return
    
    models = [
        ("openai/whisper-tiny", "Tiny - 最快"),
        ("openai/whisper-base", "Base - 平衡"),
        ("openai/whisper-small", "Small - 推荐"),
    ]
    
    for model_name, description in models:
        print(f"\n使用模型: {description} ({model_name})")
        stt = None
        try:
            stt = SpeechToText(model_name=model_name)
            result = stt.transcribe(audio_file, language="zh")
            print(f"转录结果: {result}")
        except Exception as e:
            print(f"错误: {e}")
        finally:
            if stt is not None:
                stt.cleanup()


def example_langchain_integration():
    """示例6：与LangChain集成"""
    print("=" * 60)
    print("示例6：与LangChain集成")
    print("=" * 60)
    
    stt = None
    try:
        from langchain.tools import Tool
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage
        
        # 创建语音转文字工具
        stt = SpeechToText(model_name="openai/whisper-small")
        
        def transcribe_audio(audio_path: str) -> str:
            """转录音频文件为文字"""
            return stt.transcribe(audio_path, language="zh")
        
        # 将语音转文字功能封装为LangChain工具
        transcription_tool = Tool(
            name="speech_to_text",
            description="将音频文件转换为文字，支持WAV、MP3等格式",
            func=transcribe_audio
        )
        
        # 创建LangChain模型
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            print("提示: 需要设置 SILICONFLOW_API_KEY 环境变量\n")
            return
        
        model = init_chat_model(
            model="Qwen/Qwen3-8B",
            model_provider="openai",
            base_url="https://api.siliconflow.cn/v1/",
            api_key=api_key,
            temperature=0.7
        )
        
        # 绑定工具到模型
        model_with_tools = model.bind_tools([transcription_tool])
        
        print("\n已创建LangChain模型并绑定语音转文字工具")
        print("现在可以在对话中使用语音转文字功能了\n")
        
        # 示例：用户询问转录结果
        messages = [HumanMessage(content="请帮我转录音频文件 audio.wav")]
        response = model_with_tools.invoke(messages)
        print(f"模型回复: {response.content}\n")
        
    except Exception as e:
        print(f"集成示例出错: {e}")
        print("提示: 需要设置 SILICONFLOW_API_KEY 环境变量\n")
    finally:
        if stt is not None:
            stt.cleanup()


def main():
    """主函数"""
    print("\n语音转文字功能使用示例\n")
    
    print("请选择要运行的示例:")
    print("1. 基础文件转录（默认）")
    print("2. 英文转录")
    print("3. 自动检测语言")
    print("4. 麦克风录音并转录")
    print("5. 使用不同模型")
    print("6. 与LangChain集成")
    print()
    
    choice = input("请选择 (1-6，默认1): ").strip() or "1"
    
    examples = {
        "1": example_file_transcription,
        "2": example_english_transcription,
        "3": example_auto_detect_language,
        "4": example_microphone_transcription,
        "5": example_different_models,
        "6": example_langchain_integration,
    }
    
    func = examples.get(choice, example_file_transcription)
    try:
        func()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

