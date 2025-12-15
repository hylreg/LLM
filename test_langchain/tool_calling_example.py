#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
工具调用示例程序

本程序演示如何使用LangChain进行工具调用（Function Calling）。
模型可以根据用户需求自动选择并调用相应的工具。
"""

import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


def create_siliconflow_model():
    """
    使用LangChain原生方式创建硅基流动模型实例
    
    从环境变量中读取:
    - SILICONFLOW_API_KEY: 硅基流动API密钥
    - SILICONFLOW_BASE_URL: 硅基流动API基础URL (可选，默认为https://api.siliconflow.cn/v1)
    
    Returns:
        模型实例，可用于调用create_agent等函数
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


# 定义工具函数
@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    
    Args:
        city: 城市名称
        
    Returns:
        该城市的天气信息字符串
    """
    # 这里是一个模拟实现，实际应用中应该调用真实的天气API
    weather_data = {
        "北京": "晴天，温度 15-25°C",
        "上海": "多云，温度 18-26°C",
        "广州": "小雨，温度 20-28°C",
        "深圳": "晴天，温度 22-30°C",
    }
    return weather_data.get(city, f"{city}的天气信息暂时不可用")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式的结果
    
    Args:
        expression: 数学表达式字符串，例如 "2 + 3 * 4"
        
    Returns:
        计算结果字符串
    """
    try:
        # 安全的数学表达式计算
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_current_time() -> str:
    """
    获取当前时间
    
    Returns:
        当前时间的字符串表示
    """
    from datetime import datetime
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# 工具映射字典，用于根据工具名称执行相应的工具
TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_current_time": get_current_time,
}


def execute_tool_call(tool_call, tools_dict=TOOL_MAP):
    """
    执行工具调用
    
    Args:
        tool_call: 工具调用对象，包含name和args
        tools_dict: 工具名称到工具函数的映射字典
        
    Returns:
        工具执行结果字符串
    """
    tool_name = tool_call['name']
    tool_args = tool_call['args']
    
    if tool_name in tools_dict:
        tool_func = tools_dict[tool_name]
        return tool_func.invoke(tool_args)
    else:
        return f"未知的工具: {tool_name}"


def process_tool_calls(model_with_tools, messages, response):
    """
    处理模型返回的工具调用，执行工具并获取最终回复
    
    Args:
        model_with_tools: 绑定工具的模型实例
        messages: 消息历史列表
        response: 模型的响应，可能包含工具调用
        
    Returns:
        最终回复消息
    """
    if not response.tool_calls:
        return response
    
    print(f"\n模型请求调用工具: {len(response.tool_calls)} 个")
    
    # 将模型的响应添加到消息历史
    messages.append(response)
    
    # 执行所有工具调用
    for tool_call in response.tool_calls:
        print(f"  工具名称: {tool_call['name']}")
        print(f"  工具参数: {tool_call['args']}")
        
        # 执行工具
        tool_result = execute_tool_call(tool_call)
        print(f"  工具执行结果: {tool_result}")
        
        # 将工具结果添加到消息历史
        messages.append(ToolMessage(
            content=tool_result,
            tool_call_id=tool_call['id']
        ))
    
    # 再次调用模型，让它基于工具结果生成最终回复
    final_response = model_with_tools.invoke(messages)
    return final_response


def demo_tool_calling():
    """
    演示如何使用工具调用功能
    """
    try:
        # 创建模型实例
        model = create_siliconflow_model()
        print("成功创建硅基流动模型实例")
        
        # 定义工具列表
        tools = [get_weather, calculate, get_current_time]
        
        # 将工具绑定到模型
        model_with_tools = model.bind_tools(tools)
        
        # 示例1: 查询天气
        print("\n=== 示例1: 查询天气 ===")
        messages = [HumanMessage(content="北京今天天气怎么样？")]
        response = model_with_tools.invoke(messages)
        print(f"用户: {messages[0].content}")
        print(f"模型回复: {response.content}")
        
        # 处理工具调用
        final_response = process_tool_calls(model_with_tools, messages, response)
        if final_response != response:
            print(f"\n最终回复: {final_response.content}")
        
        # 示例2: 数学计算
        print("\n=== 示例2: 数学计算 ===")
        messages = [HumanMessage(content="请帮我计算 25 * 4 + 10 的结果")]
        response = model_with_tools.invoke(messages)
        print(f"用户: {messages[0].content}")
        print(f"模型回复: {response.content}")
        
        # 处理工具调用
        final_response = process_tool_calls(model_with_tools, messages, response)
        if final_response != response:
            print(f"最终回复: {final_response.content}")
        
        # 示例3: 获取当前时间
        print("\n=== 示例3: 获取当前时间 ===")
        messages = [HumanMessage(content="现在几点了？")]
        response = model_with_tools.invoke(messages)
        print(f"用户: {messages[0].content}")
        print(f"模型回复: {response.content}")
        
        # 处理工具调用
        final_response = process_tool_calls(model_with_tools, messages, response)
        if final_response != response:
            print(f"最终回复: {final_response.content}")
        
        return model_with_tools
        
    except Exception as e:
        print(f"出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    demo_tool_calling()

