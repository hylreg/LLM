#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
流式输出示例程序

本程序演示如何使用LangChain和硅基流动模型实现流式输出功能，
模拟打字机效果逐步显示模型的回复。
"""

import os
import sys
import time
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


def create_siliconflow_model():
    """
    使用LangChain原生方式创建硅基流动模型实例
    
    从环境变量中读取:
    - SILICONFLOW_API_KEY: 硅基流动API密钥
    - SILICONFLOW_BASE_URL: 硅基流动API基础URL (可选，默认为https://api.siliconflow.cn/v1)
    
    Returns:
        模型实例
    """
    # 从环境变量获取API密钥
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量SILICONFLOW_API_KEY")
    
    # 从环境变量获取基础URL，如果未设置则使用默认值
    base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1/")
    
    # 使用init_chat_model初始化硅基流动模型
    model = init_chat_model(
        model="Qwen/Qwen3-8B",
        model_provider="openai",
        base_url=base_url,
        api_key=api_key,
        temperature=0.7
    )
    
    return model


def stream_response(messages, model):
    """
    流式输出模型的响应
    
    Args:
        messages: 发送给模型的消息
        model: 模型实例
    """
    # 使用stream方法获取流式响应
    stream = model.stream(messages)
    
    print("模型回复: ", end="", flush=True)
    
    # 逐个token输出响应
    for chunk in stream:
        if chunk.content:
            print(chunk.content, end="", flush=True)
            # 添加一个小延迟以获得更好的视觉效果
            time.sleep(0.02)
    
    print()  # 输出换行


def demo_streaming():
    """
    演示流式输出功能
    """
    try:
        # 创建模型实例
        model = create_siliconflow_model()
        print("成功创建硅基流动模型实例\n")
        
        # 示例1: 简单对话
        print("=== 流式输出对话示例 ===")
        messages = [HumanMessage(content="请写一首关于人工智能的五言诗")]
        print(f"用户: {messages[0].content}")
        stream_response(messages, model)
        
        print("\n" + "="*50 + "\n")
        
        # 示例2: 技术解释
        print("=== 技术解释流式输出示例 ===")
        messages = [HumanMessage(content="请用通俗易懂的语言解释什么是大语言模型，大概100字")]
        print(f"用户: {messages[0].content}")
        stream_response(messages, model)
        
    except Exception as e:
        print(f"出现错误: {e}")


def interactive_mode():
    """
    交互模式，允许用户输入问题并实时查看流式输出结果
    """
    try:
        model = create_siliconflow_model()
        print("成功创建硅基流动模型实例")
        print("进入交互模式（输入'退出'结束对话）\n")
        
        while True:
            user_input = input("用户: ").strip()
            
            if user_input.lower() in ['退出', 'quit', 'exit']:
                print("再见！")
                break
            
            if user_input:
                messages = [HumanMessage(content=user_input)]
                print("模型回复: ", end="", flush=True)
                
                # 流式输出响应
                stream = model.stream(messages)
                for chunk in stream:
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                        time.sleep(0.01)
                
                print("\n")  # 输出换行
                
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"出现错误: {e}")


def main():
    """
    主函数
    """
    print("硅基流动模型流式输出示例")
    print("="*30)
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        demo_streaming()
        print("\n提示：运行 'python streaming_output_example.py interactive' 进入交互模式")


if __name__ == "__main__":
    main()