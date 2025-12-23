#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
端到端语音大模型助手示例

演示如何使用 VoiceAssistant 类进行语音对话。
"""

import sys
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


def example_basic_conversation():
    """示例1：基础对话"""
    print("=" * 60)
    print("示例1：基础语音对话")
    print("=" * 60)
    print()
    
    try:
        # 初始化助手（使用Silicon Flow）
        assistant = VoiceAssistant(
            stt_model_name="openai/whisper-small",
            llm_provider="siliconflow",
            temperature=0.7
        )
        
        # 从麦克风输入
        result = assistant.process_voice_input(
            audio_source="microphone",
            language="zh",
            play_response=True
        )
        
        print("\n处理结果:")
        print(f"用户输入: {result['user_text']}")
        print(f"LLM回复: {result['llm_reply']}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def example_file_input():
    """示例2：从文件输入"""
    print("=" * 60)
    print("示例2：从音频文件输入")
    print("=" * 60)
    print()
    
    audio_file = input("请输入音频文件路径: ").strip()
    
    if not audio_file:
        print("未提供音频文件路径")
        return
    
    try:
        assistant = VoiceAssistant(
            stt_model_name="openai/whisper-small",
            llm_provider="siliconflow",
            temperature=0.7
        )
        
        result = assistant.process_voice_input(
            audio_source=audio_file,
            language="zh",
            play_response=True,
            save_audio=True
        )
        
        print("\n处理结果:")
        print(f"用户输入: {result['user_text']}")
        print(f"LLM回复: {result['llm_reply']}")
        print(f"语音文件: {result['audio_file']}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def example_multi_turn_conversation():
    """示例3：多轮对话"""
    print("=" * 60)
    print("示例3：多轮对话")
    print("=" * 60)
    print()
    
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        
        assistant = VoiceAssistant(
            stt_model_name="openai/whisper-small",
            llm_provider="siliconflow",
            temperature=0.7
        )
        
        conversation_history = []
        
        print("开始多轮对话（输入'quit'退出）\n")
        
        for turn in range(3):  # 示例：3轮对话
            print(f"\n--- 第 {turn + 1} 轮对话 ---")
            
            # 用户输入
            user_text = input("请输入问题（或按回车使用麦克风）: ").strip()
            if not user_text:
                user_text = assistant.speech_to_text("microphone", language="zh")
            
            if user_text.lower() in ['quit', 'exit', '退出']:
                break
            
            # LLM回复
            llm_reply = assistant.llm_response(user_text, conversation_history)
            
            # 更新对话历史
            conversation_history.append(HumanMessage(content=user_text))
            conversation_history.append(AIMessage(content=llm_reply))
            
            # 生成并播放语音
            audio_file = assistant.text_to_speech(llm_reply)
            if audio_file:
                assistant.play_audio(audio_file)
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def example_ollama_local():
    """示例4：使用本地Ollama模型"""
    print("=" * 60)
    print("示例4：使用本地Ollama模型")
    print("=" * 60)
    print()
    
    try:
        assistant = VoiceAssistant(
            stt_model_name="openai/whisper-small",
            llm_provider="ollama",
            llm_model_name="qwen3:0.6b",
            temperature=0.7
        )
        
        result = assistant.process_voice_input(
            audio_source="microphone",
            language="zh",
            play_response=True
        )
        
        print("\n处理结果:")
        print(f"用户输入: {result['user_text']}")
        print(f"LLM回复: {result['llm_reply']}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("\n端到端语音大模型助手示例\n")
    
    print("请选择示例:")
    print("1. 基础语音对话（默认）")
    print("2. 从音频文件输入")
    print("3. 多轮对话")
    print("4. 使用本地Ollama模型")
    print()
    
    choice = input("请选择 (1-4，默认1): ").strip() or "1"
    
    examples = {
        "1": example_basic_conversation,
        "2": example_file_input,
        "3": example_multi_turn_conversation,
        "4": example_ollama_local,
    }
    
    func = examples.get(choice, example_basic_conversation)
    
    try:
        func()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

