"""
语音转文字（Speech-to-Text）模块

本模块使用 OpenAI Whisper 模型实现语音转文字功能。
支持多种音频格式，包括 WAV、MP3、M4A、FLAC 等。
支持从音频文件或麦克风录音进行转录。
"""

from .core import SpeechToText

__all__ = ["SpeechToText"]

