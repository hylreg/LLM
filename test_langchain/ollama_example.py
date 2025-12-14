#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ollama本地模型示例程序

本程序演示如何在LangChain中使用本地运行的Ollama模型。
Ollama允许你在本地运行大语言模型，无需网络连接。
"""

import sys
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate


def create_ollama_model(model_name="qwen3:0.6b"):
    """
    创建Ollama模型实例
    
    Args:
        model_name (str): 要使用的模型名称，默认为'qwen3:0.6b'
    
    Returns:
        ChatOllama: 配置好的模型实例
    """
    # 创建Ollama模型实例
    model = ChatOllama(
        model=model_name,
        temperature=0.7,
        num_predict=256
    )
    
    return model


def simple_invoke_example(model):
    """
    简单调用示例
    
    Args:
        model: Ollama模型实例
    """
    print("=== 简单调用示例 ===")
    
    # 用户输入
    user_input = "请简单介绍一下Python编程语言"
    print(f"用户: {user_input}")
    
    try:
        # 调用模型
        response = model.invoke(user_input)
        print(f"模型回复: {response.content}\n")
    except Exception as e:
        print(f"调用模型时出错: {e}\n")


def streaming_example(model):
    """
    流式输出示例
    
    Args:
        model: Ollama模型实例
    """
    print("=== 流式输出示例 ===")
    
    # 用户输入
    user_input = "请写一首关于春天的七言绝句"
    print(f"用户: {user_input}")
    print("模型回复: ", end="", flush=True)
    
    try:
        # 流式输出模型响应
        for chunk in model.stream(user_input):
            print(chunk.content, end="", flush=True)
            time.sleep(0.02)  # 添加小延迟以获得更好的视觉效果
        print("\n")  # 换行
    except Exception as e:
        print(f"\n流式输出时出错: {e}\n")


def chat_prompt_template_example(model):
    """
    使用ChatPromptTemplate示例
    
    Args:
        model: Ollama模型实例
    """
    print("=== ChatPromptTemplate使用示例 ===")
    
    try:
        # 创建提示模板
        template = """
        你是一个乐于助人的AI助手，请用简洁明了的语言回答问题。
        问题：{question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # 创建处理链
        chain = prompt | model
        
        # 使用处理链
        question = "什么是人工智能？"
        response = chain.invoke({"question": question})
        print(f"问题: {question}")
        print(f"回答: {response.content}\n")
        
    except Exception as e:
        print(f"使用ChatPromptTemplate时出错: {e}\n")


def conversation_example(model):
    """
    多轮对话示例
    
    Args:
        model: Ollama模型实例
    """
    print("=== 多轮对话示例 ===")
    
    try:
        # 第一轮对话
        messages = [HumanMessage(content="你好，我叫小明")]
        response = model.invoke(messages)
        print(f"用户: {messages[0].content}")
        print(f"模型: {response.content}")
        
        # 第二轮对话
        messages.append(response)
        messages.append(HumanMessage(content="我刚才告诉你我叫什么名字了吗？"))
        response = model.invoke(messages)
        print(f"用户: {messages[-1].content}")
        print(f"模型: {response.content}\n")
        
    except Exception as e:
        print(f"多轮对话时出错: {e}\n")


def main():
    """
    主函数
    """
    # 检查命令行参数
    model_name = "qwen3:0.6b"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    print(f"Ollama本地模型示例 (使用模型: {model_name})")
    print("=" * 50)
    
    try:
        # 创建模型实例
        model = create_ollama_model(model_name)
        print(f"成功创建Ollama模型实例: {model_name}\n")
        
        # 运行各种示例
        simple_invoke_example(model)
        streaming_example(model)
        chat_prompt_template_example(model)
        conversation_example(model)
        
    except Exception as e:
        print(f"创建模型实例时出错: {e}")
        print("请确保：")
        print("1. Ollama服务正在运行")
        print("2. 指定的模型已下载 (可以使用 'ollama pull {model_name}' 下载)")
        print("3. 模型名称正确")


if __name__ == "__main__":
    main()