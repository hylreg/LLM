#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
语音转文字（Speech-to-Text）核心模块

本模块使用 OpenAI Whisper 模型实现语音转文字功能。
支持多种音频格式，包括 WAV、MP3、M4A、FLAC 等。

技术选型说明：
- 模型：OpenAI Whisper（通过 transformers 库）
- 优势：
  1. 开源免费，无需API密钥
  2. 支持多语言（包括中文），准确度高
  3. 可在本地运行，保护隐私
  4. 支持多种音频格式
  5. 项目已包含 transformers 和 torch 依赖
"""

import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Optional, Union
import warnings
import atexit
import multiprocessing

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

# 忽略一些警告信息
warnings.filterwarnings("ignore")

# 设置多进程启动方法，避免资源泄漏
if hasattr(multiprocessing, 'set_start_method'):
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过，忽略错误
        pass

# 注册清理函数，在程序退出时清理资源
def _cleanup_resources():
    """清理多进程资源"""
    try:
        # 清理 PyTorch 的多进程资源
        if hasattr(torch.multiprocessing, 'shutdown'):
            torch.multiprocessing.shutdown()
    except Exception:
        pass

atexit.register(_cleanup_resources)


class SpeechToText:
    """语音转文字转换器"""
    
    def __init__(self, model_name: str = "openai/whisper-small", device: Optional[str] = None):
        """
        初始化语音转文字模型
        
        Args:
            model_name: Whisper 模型名称，可选值：
                - "openai/whisper-tiny" (最快，准确度较低)
                - "openai/whisper-base" (平衡速度与准确度)
                - "openai/whisper-small" (推荐，平衡性好)
                - "openai/whisper-medium" (更高准确度)
                - "openai/whisper-large" (最高准确度，速度较慢)
            device: 计算设备，"cuda" 使用GPU，"cpu" 使用CPU，None 自动选择
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"正在加载 Whisper 模型: {model_name}")
        print(f"使用设备: {self.device}")
        
        # 加载处理器和模型
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("模型加载完成！")
    
    def cleanup(self):
        """清理资源，释放模型和内存"""
        try:
            # 清理模型
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            
            # 清理 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 强制垃圾回收
            import gc
            gc.collect()
        except Exception:
            pass
    
    def __del__(self):
        """析构函数，确保资源被清理"""
        self.cleanup()
    
    def transcribe(
        self, 
        audio_path: str, 
        language: Optional[str] = "zh",
        return_timestamps: bool = False
    ) -> Union[str, dict]:
        """
        将音频文件转换为文字
        
        Args:
            audio_path: 音频文件路径
            language: 音频语言代码，例如 "zh" (中文), "en" (英文), "auto" (自动检测)
            return_timestamps: 是否返回时间戳信息
            
        Returns:
            如果 return_timestamps=False，返回文字字符串
            如果 return_timestamps=True，返回包含文字和时间戳的字典
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        if not LIBROSA_AVAILABLE:
            raise ImportError("请安装 librosa 库: pip install librosa")
        
        try:
            # 读取音频文件（自动转换为16kHz采样率）
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 预处理音频
            inputs = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成转录
            with torch.no_grad():
                # 设置生成参数以加快速度
                generate_kwargs = {
                    "max_length": 448,  # 限制最大长度
                    "num_beams": 1,  # 不使用beam search以加快速度
                    "do_sample": False,  # 使用贪心解码
                }
                
                # 使用现代方法设置语言和任务
                if language and language != "auto":
                    generate_kwargs["language"] = language
                    generate_kwargs["task"] = "transcribe"
                
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    **generate_kwargs
                )
            
            # 解码转录结果
            transcription = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            if return_timestamps:
                # 如果需要时间戳，使用不同的解码方法
                # 注意：这需要更复杂的处理，这里简化处理
                return {
                    "text": transcription,
                    "language": language if language != "auto" else "detected"
                }
            
            return transcription
            
        except Exception as e:
            raise Exception(f"转录失败: {str(e)}")
    
    def transcribe_stream(
        self, 
        audio_data, 
        sample_rate: int = 16000,
        language: Optional[str] = None
    ) -> str:
        """
        实时流式音频转文字（适用于实时录音场景）
        
        Args:
            audio_data: 音频数据（numpy array 或 bytes）
            sample_rate: 采样率
            language: 音频语言代码，例如 "zh" (中文), "en" (英文), "auto" (自动检测)
            
        Returns:
            转录的文字
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("请安装 librosa 库: pip install librosa")
        
        import numpy as np
        
        try:
            print("开始处理音频数据...", flush=True)
            
            # 如果输入是bytes，转换为numpy array
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_array = audio_data
            
            # 确保音频数组是1D的
            if len(audio_array.shape) > 1:
                print(f"检测到多维数组，形状: {audio_array.shape}，正在转换为1D数组...", flush=True)
                audio_array = audio_array.flatten() if audio_array.shape[1] == 1 else np.mean(audio_array, axis=1)
            
            print(f"音频数组形状: {audio_array.shape}, 数据类型: {audio_array.dtype}", flush=True)
            
            # 确保采样率为16kHz
            if sample_rate != 16000:
                print(f"重新采样: {sample_rate} -> 16000 Hz", flush=True)
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                print(f"重新采样后形状: {audio_array.shape}", flush=True)
            
            # 预处理
            print("正在预处理音频...", flush=True)
            inputs = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            print(f"预处理完成，输入形状: {inputs['input_features'].shape}", flush=True)
            
            # 生成转录
            print("正在设置语言参数...", flush=True)
            if language and language != "auto":
                print(f"已设置语言: {language}", flush=True)
            else:
                print("使用自动语言检测", flush=True)
            
            print("正在生成转录结果（这可能需要一些时间，请耐心等待）...", flush=True)
            with torch.no_grad():
                # 设置生成参数以加快速度
                generate_kwargs = {
                    "max_length": 448,  # 限制最大长度
                    "num_beams": 1,  # 不使用beam search以加快速度
                    "do_sample": False,  # 使用贪心解码
                }
                
                # 使用现代方法设置语言和任务
                if language and language != "auto":
                    generate_kwargs["language"] = language
                    generate_kwargs["task"] = "transcribe"
                
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    **generate_kwargs
                )
            
            print(f"生成完成，生成了 {len(generated_ids[0])} 个token", flush=True)
            
            # 解码
            print("正在解码结果...", end="", flush=True)
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            print(" 完成")
            
            return transcription
            
        except Exception as e:
            import traceback
            print(f"\n错误详情: {traceback.format_exc()}")
            raise Exception(f"流式转录失败: {str(e)}")
    
    def record_from_microphone(
        self,
        duration: float = 5.0,
        sample_rate: int = 16000,
        channels: int = 1
    ):
        """
        从麦克风录制音频
        
        Args:
            duration: 录制时长（秒）
            sample_rate: 采样率，默认16000（Whisper推荐）
            channels: 声道数，1=单声道，2=立体声
            
        Returns:
            numpy array格式的音频数据
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError("请安装 sounddevice 库: pip install sounddevice")
        
        import numpy as np
        
        print(f"开始录音，时长 {duration} 秒...")
        print("正在录音...", end="", flush=True)
        
        # 录制音频
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype=np.float32
        )
        sd.wait()  # 等待录制完成
        
        print("完成！\n")
        
        # 确保返回1D数组
        if len(audio_data.shape) > 1:
            if channels == 2:
                # 如果是立体声，转换为单声道（取平均值）
                audio_data = np.mean(audio_data, axis=1)
            else:
                # 如果是单声道但返回2D数组，展平它
                audio_data = audio_data.flatten()
        
        return audio_data
    
    def transcribe_from_microphone(
        self,
        duration: float = 5.0,
        language: Optional[str] = "zh",
        sample_rate: int = 16000
    ) -> str:
        """
        从麦克风录音并直接转录为文字
        
        Args:
            duration: 录制时长（秒）
            language: 音频语言代码，例如 "zh" (中文), "en" (英文), "auto" (自动检测)
            sample_rate: 采样率，默认16000
            
        Returns:
            转录的文字
        """
        try:
            # 录音
            audio_data = self.record_from_microphone(
                duration=duration,
                sample_rate=sample_rate,
                channels=1
            )
            
            # 检查音频数据是否有效
            import numpy as np
            if audio_data is None or len(audio_data) == 0:
                raise ValueError("录音数据为空，请检查麦克风是否正常工作")
            
            # 检查音频是否全是静音
            audio_max = np.max(np.abs(audio_data))
            if audio_max < 0.01:  # 如果最大音量很小，可能是静音
                print("警告: 检测到音频音量很小，可能是静音或麦克风未正常工作")
            
            print(f"音频数据长度: {len(audio_data)} 采样点")
            print(f"音频数据范围: [{np.min(audio_data):.4f}, {np.max(audio_data):.4f}]")
            
            # 转录
            print("正在转录...")
            print(f"使用语言设置: {language}")
            result = self.transcribe_stream(
                audio_data, 
                sample_rate=sample_rate,
                language=language
            )
            
            return result
            
        except Exception as e:
            import traceback
            print(f"\n错误详情: {traceback.format_exc()}")
            raise Exception(f"麦克风转录失败: {str(e)}")

