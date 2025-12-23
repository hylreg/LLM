#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
端到端语音大模型助手核心模块

完整的语音对话系统，支持：
1. 语音输入（麦克风或文件）
2. 语音转文字（Whisper）
3. 大语言模型处理（LangChain + Silicon Flow/Ollama）
4. 文字转语音（edge-tts）
5. 语音播放

使用流程：
  语音输入 -> STT -> LLM -> TTS -> 语音输出
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
demos_dir = current_dir.parent

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
        raise ImportError("无法导入 SpeechToText，请检查 speech_to_text 模块")


class VoiceAssistant:
    """语音助手类，整合STT、LLM和TTS"""
    
    def __init__(
        self,
        stt_model_name: str = "openai/whisper-small",
        llm_provider: str = "siliconflow",  # "siliconflow" 或 "ollama"
        llm_model_name: Optional[str] = None,
        tts_voice: str = "zh-CN-XiaoxiaoNeural",  # 中文语音
        temperature: float = 0.7
    ):
        """
        初始化语音助手
        
        Args:
            stt_model_name: Whisper模型名称
            llm_provider: LLM提供商，"siliconflow" 或 "ollama"
            llm_model_name: LLM模型名称，None则使用默认值
            tts_voice: TTS语音名称
            temperature: LLM温度参数
        """
        print("正在初始化语音助手...")
        
        # 初始化STT
        print("  - 加载语音转文字模型...")
        self.stt = SpeechToText(model_name=stt_model_name)
        
        # 初始化LLM
        print("  - 加载大语言模型...")
        self.llm = self._init_llm(llm_provider, llm_model_name, temperature)
        
        # 初始化TTS
        print("  - 初始化文字转语音...")
        self.tts_voice = tts_voice
        self._check_tts_available()
        
        print("语音助手初始化完成！\n")
    
    def _init_llm(self, provider: str, model_name: Optional[str], temperature: float):
        """初始化大语言模型"""
        if provider == "siliconflow":
            return self._init_siliconflow_llm(model_name, temperature)
        elif provider == "ollama":
            return self._init_ollama_llm(model_name, temperature)
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")
    
    def _init_siliconflow_llm(self, model_name: Optional[str], temperature: float):
        """初始化Silicon Flow模型"""
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage
        
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError("请设置环境变量 SILICONFLOW_API_KEY")
        
        base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1/")
        model_name = model_name or "Qwen/Qwen3-8B"
        
        model = init_chat_model(
            model=model_name,
            model_provider="openai",
            base_url=base_url,
            api_key=api_key,
            temperature=temperature
        )
        
        # 测试连接
        try:
            test_msg = [HumanMessage(content="你好")]
            response = model.invoke(test_msg)
            print(f"    LLM连接成功: {model_name}")
        except Exception as e:
            print(f"    LLM连接警告: {e}")
        
        return model
    
    def _init_ollama_llm(self, model_name: Optional[str], temperature: float):
        """初始化Ollama模型"""
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage
        
        model_name = model_name or "qwen3:0.6b"
        
        model = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=512
        )
        
        # 测试连接
        try:
            test_msg = [HumanMessage(content="你好")]
            response = model.invoke(test_msg)
            print(f"    LLM连接成功: {model_name}")
        except Exception as e:
            print(f"    LLM连接警告: {e}")
        
        return model
    
    def _check_tts_available(self):
        """检查TTS是否可用"""
        try:
            import edge_tts
            print(f"    TTS可用: edge-tts")
        except ImportError:
            print("    警告: edge-tts 未安装，将使用简单TTS")
            print("    安装方法: pip install edge-tts")
    
    def speech_to_text(self, audio_source: str, language: str = "zh") -> str:
        """
        语音转文字
        
        Args:
            audio_source: 音频文件路径或"microphone"（使用麦克风）
            language: 语言代码
            
        Returns:
            转录的文字
        """
        if audio_source.lower() == "microphone":
            print("\n请说话（默认5秒）...")
            text = self.stt.transcribe_from_microphone(duration=5.0, language=language)
        else:
            print(f"\n正在转录音频文件: {audio_source}")
            text = self.stt.transcribe(audio_source, language=language)
        
        print(f"转录结果: {text}\n")
        return text
    
    def llm_response(self, text: str, conversation_history: Optional[list] = None) -> str:
        """
        大语言模型回复
        
        Args:
            text: 用户输入的文本
            conversation_history: 对话历史
            
        Returns:
            LLM的回复文本
        """
        from langchain_core.messages import HumanMessage
        
        # 构建消息列表
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append(HumanMessage(content=text))
        
        # 调用LLM
        print("正在思考...")
        response = self.llm.invoke(messages)
        reply = response.content
        
        print(f"LLM回复: {reply}\n")
        return reply
    
    def text_to_speech(self, text: str, output_file: Optional[str] = None) -> Optional[str]:
        """
        文字转语音
        
        Args:
            text: 要转换的文字
            output_file: 输出文件路径，None则自动生成临时文件
            
        Returns:
            生成的音频文件路径，失败返回None
        """
        try:
            import edge_tts
            import asyncio
            
            async def generate_speech():
                communicate = edge_tts.Communicate(text, self.tts_voice)
                if output_file is None:
                    # 使用临时文件
                    fd, output_path = tempfile.mkstemp(suffix=".mp3")
                    os.close(fd)
                else:
                    output_path = output_file
                await communicate.save(output_path)
                return output_path
            
            print("正在生成语音...")
            output_path = asyncio.run(generate_speech())
            print(f"语音已保存到: {output_path}\n")
            return output_path
            
        except ImportError:
            # 如果edge-tts不可用，使用简单的TTS（仅Linux）
            print("使用简单TTS（需要系统TTS支持）...")
            if output_file is None:
                output_path = tempfile.mktemp(suffix=".wav")
            else:
                output_path = output_file
            
            try:
                import subprocess
                # 使用系统TTS（Linux）
                subprocess.run(
                    ["espeak", "-v", "zh", "-s", "150", "-w", output_path, text],
                    check=True,
                    capture_output=True
                )
                print(f"语音已保存到: {output_path}\n")
                return output_path
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("警告: 无法生成语音，请安装 edge-tts 或系统TTS工具")
                return None
    
    def play_audio(self, audio_file: str):
        """播放音频文件"""
        try:
            import sounddevice as sd
            import soundfile as sf
            
            print("正在播放语音...")
            data, samplerate = sf.read(audio_file)
            sd.play(data, samplerate)
            sd.wait()  # 等待播放完成
            print("播放完成\n")
            
        except Exception as e:
            print(f"播放音频失败: {e}")
            print("请手动播放音频文件")
    
    def process_voice_input(
        self,
        audio_source: str,
        language: str = "zh",
        conversation_history: Optional[list] = None,
        play_response: bool = True,
        save_audio: bool = False
    ) -> dict:
        """
        处理完整的语音输入流程
        
        Args:
            audio_source: 音频文件路径或"microphone"
            language: 语言代码
            conversation_history: 对话历史
            play_response: 是否播放回复
            save_audio: 是否保存音频文件
            
        Returns:
            包含所有步骤结果的字典
        """
        result = {}
        
        # 1. 语音转文字
        user_text = self.speech_to_text(audio_source, language=language)
        result["user_text"] = user_text
        
        # 2. LLM处理
        llm_reply = self.llm_response(user_text, conversation_history)
        result["llm_reply"] = llm_reply
        
        # 3. 文字转语音
        audio_file = self.text_to_speech(llm_reply, output_file=None if not save_audio else "reply.mp3")
        result["audio_file"] = audio_file
        
        # 4. 播放语音
        if play_response and audio_file:
            self.play_audio(audio_file)
        
        return result

