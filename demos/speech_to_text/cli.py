#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
语音转文字命令行工具

提供交互式命令行界面，支持从文件或麦克风进行语音转文字。
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


def _run_microphone_mode():
    """麦克风录音模式"""
    # 选择语言
    print("\n语言选择:")
    print("1. 中文 (zh)")
    print("2. 英文 (en)")
    print("3. 自动检测 (auto)")
    lang_choice = input("请选择 (1/2/3，默认1): ").strip() or "1"
    
    lang_map = {"1": "zh", "2": "en", "3": "auto"}
    language = lang_map.get(lang_choice, "zh")
    
    # 选择模型
    print("\n模型选择:")
    print("1. tiny (最快，准确度较低)")
    print("2. base (平衡)")
    print("3. small (推荐，默认)")
    print("4. medium (高准确度)")
    print("5. large (最高准确度，较慢)")
    model_choice = input("请选择 (1-5，默认3): ").strip() or "3"
    
    model_map = {
        "1": "openai/whisper-tiny",
        "2": "openai/whisper-base",
        "3": "openai/whisper-small",
        "4": "openai/whisper-medium",
        "5": "openai/whisper-large",
    }
    model_name = model_map.get(model_choice, "openai/whisper-small")
    
    # 录音时长
    try:
        duration = float(input("\n请输入录音时长（秒，默认5秒）: ").strip() or "5")
        if duration <= 0:
            print("时长必须大于0，使用默认值5秒")
            duration = 5.0
    except ValueError:
        print("无效输入，使用默认值5秒")
        duration = 5.0
    
    print(f"\n使用模型: {model_name}")
    print(f"语言设置: {language}")
    print(f"录音时长: {duration} 秒")
    print("\n准备开始录音...")
    print("-" * 60)
    
    stt = None
    try:
        # 创建转录器
        stt = SpeechToText(model_name=model_name)
        
        # 录音并转录
        result = stt.transcribe_from_microphone(
            duration=duration,
            language=language
        )
        
        print("\n转录结果:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        # 询问是否保存到文件
        save = input("\n是否保存到文件? (y/n，默认n): ").strip().lower()
        if save == 'y':
            output_file = f"microphone_transcription_{os.path.basename(model_name)}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"已保存到: {output_file}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if stt is not None:
            stt.cleanup()


def main():
    """主函数"""
    print("=" * 60)
    print("语音转文字工具")
    print("=" * 60)
    print()
    
    # 选择输入方式
    print("请选择输入方式:")
    print("1. 从音频文件转录")
    print("2. 从麦克风录音并转录")
    input_choice = input("请选择 (1/2，默认1): ").strip() or "1"
    
    if input_choice == "2":
        # 麦克风录音模式
        _run_microphone_mode()
        return
    
    # 文件转录模式
    # 获取音频文件路径
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = input("请输入音频文件路径: ").strip()
    
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        print(f"错误: 文件不存在: {audio_path}")
        return
    
    # 选择语言
    print("\n语言选择:")
    print("1. 中文 (zh)")
    print("2. 英文 (en)")
    print("3. 自动检测 (auto)")
    lang_choice = input("请选择 (1/2/3，默认1): ").strip() or "1"
    
    lang_map = {"1": "zh", "2": "en", "3": "auto"}
    language = lang_map.get(lang_choice, "zh")
    
    # 选择模型
    print("\n模型选择:")
    print("1. tiny (最快，准确度较低)")
    print("2. base (平衡)")
    print("3. small (推荐，默认)")
    print("4. medium (高准确度)")
    print("5. large (最高准确度，较慢)")
    model_choice = input("请选择 (1-5，默认3): ").strip() or "3"
    
    model_map = {
        "1": "openai/whisper-tiny",
        "2": "openai/whisper-base",
        "3": "openai/whisper-small",
        "4": "openai/whisper-medium",
        "5": "openai/whisper-large",
    }
    model_name = model_map.get(model_choice, "openai/whisper-small")
    
    print(f"\n使用模型: {model_name}")
    print(f"语言设置: {language}")
    print(f"音频文件: {audio_path}")
    print("\n开始转录...")
    print("-" * 60)
    
    stt = None
    try:
        # 创建转录器
        stt = SpeechToText(model_name=model_name)
        
        # 执行转录
        result = stt.transcribe(audio_path, language=language)
        
        print("\n转录结果:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        # 询问是否保存到文件
        save = input("\n是否保存到文件? (y/n，默认n): ").strip().lower()
        if save == 'y':
            output_file = audio_path.rsplit('.', 1)[0] + '_transcription.txt'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"已保存到: {output_file}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if stt is not None:
            stt.cleanup()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

