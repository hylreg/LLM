#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
端到端语音大模型助手命令行工具

提供交互式命令行界面，支持语音对话和文件处理。
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
demos_dir = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(demos_dir) not in sys.path:
    sys.path.insert(0, str(demos_dir))

try:
    from demos.voice_assistant import VoiceAssistant
except ImportError:
    from voice_assistant import VoiceAssistant


def interactive_mode(llm_provider: str = "siliconflow"):
    """交互式模式"""
    print("=" * 60)
    print("端到端语音大模型助手 - 交互模式")
    print("=" * 60)
    print()
    
    # 初始化助手
    try:
        assistant = VoiceAssistant(
            stt_model_name="openai/whisper-small",
            llm_provider=llm_provider,
            temperature=0.7
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 对话历史
    conversation_history = []
    from langchain_core.messages import HumanMessage, AIMessage
    
    print("\n进入交互模式")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空对话历史")
    print("-" * 60)
    
    while True:
        try:
            # 选择输入方式
            print("\n请选择输入方式:")
            print("1. 麦克风录音")
            print("2. 文本输入")
            input_choice = input("请选择 (1/2，默认1): ").strip() or "1"
            
            if input_choice == "2":
                # 文本输入
                user_text = input("\n请输入问题: ").strip()
                if not user_text:
                    continue
                if user_text.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                if user_text.lower() == 'clear':
                    conversation_history.clear()
                    print("对话历史已清空")
                    continue
            else:
                # 麦克风输入
                user_text = assistant.speech_to_text("microphone", language="zh")
                if not user_text.strip():
                    continue
            
            # LLM处理
            llm_reply = assistant.llm_response(user_text, conversation_history)
            
            # 更新对话历史
            conversation_history.append(HumanMessage(content=user_text))
            conversation_history.append(AIMessage(content=llm_reply))
            
            # 生成并播放语音
            audio_file = assistant.text_to_speech(llm_reply)
            if audio_file:
                assistant.play_audio(audio_file)
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()


def single_file_mode(audio_file: str, llm_provider: str = "siliconflow"):
    """单文件处理模式"""
    print("=" * 60)
    print("端到端语音大模型助手 - 文件处理模式")
    print("=" * 60)
    print()
    
    try:
        assistant = VoiceAssistant(
            stt_model_name="openai/whisper-small",
            llm_provider=llm_provider,
            temperature=0.7
        )
        
        result = assistant.process_voice_input(
            audio_source=audio_file,
            language="zh",
            play_response=True,
            save_audio=True
        )
        
        print("\n处理完成！")
        print(f"用户输入: {result['user_text']}")
        print(f"LLM回复: {result['llm_reply']}")
        print(f"语音文件: {result['audio_file']}")
        
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="端到端语音大模型助手")
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="音频文件路径（可选，不提供则进入交互模式）"
    )
    parser.add_argument(
        "--llm",
        choices=["siliconflow", "ollama"],
        default="siliconflow",
        help="LLM提供商"
    )
    parser.add_argument(
        "--stt-model",
        default="openai/whisper-small",
        help="Whisper模型名称（暂未使用）"
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="不播放生成的语音（暂未使用）"
    )
    
    args = parser.parse_args()
    
    if args.audio_file:
        single_file_mode(args.audio_file, args.llm)
    else:
        # 交互模式：让用户选择LLM提供商
        print("请选择LLM提供商:")
        print("1. Silicon Flow (需要API密钥)")
        print("2. Ollama (本地运行)")
        provider_choice = input("请选择 (1/2，默认1): ").strip() or "1"
        llm_provider = "siliconflow" if provider_choice == "1" else "ollama"
        interactive_mode(llm_provider)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

