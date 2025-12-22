#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
长短期记忆（Memory）示例程序

本程序演示如何使用LangChain的create_agent和InMemorySaver来实现记忆功能。
包括：
1. 完整记忆 - 使用InMemorySaver保存所有对话历史
2. 多会话记忆 - 使用不同的thread_id管理多个会话
3. 记忆持久化 - 展示如何在不同调用间保持记忆
"""

import os
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage


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
    if base_url and not base_url.endswith("/"):
        base_url += "/"
    
    # 使用init_chat_model初始化硅基流动模型
    model = init_chat_model(
        model="Qwen/Qwen3-8B",
        model_provider="openai",
        base_url=base_url,
        api_key=api_key,
        temperature=0.7
    )
    
    return model


# 定义一些示例工具
@tool
def get_user_info(name: str) -> str:
    """获取用户信息（模拟工具）"""
    return f"{name} 的信息：年龄25岁，职业软件工程师"


@tool
def calculate_age(birth_year: int) -> int:
    """计算年龄"""
    return 2024 - birth_year


def basic_memory_example():
    """
    示例1: 基础记忆功能 - 使用InMemorySaver保存对话历史
    
    使用相同的thread_id可以在多次调用间保持记忆
    """
    print("=" * 60)
    print("示例1: 基础记忆功能 (使用InMemorySaver)")
    print("=" * 60)
    
    # 创建模型
    model = create_siliconflow_model()
    
    # 创建checkpointer（记忆存储）
    checkpointer = InMemorySaver()
    
    # 创建agent，传入checkpointer
    agent = create_agent(
        model=model,
        tools=[get_user_info, calculate_age],
        checkpointer=checkpointer,
    )
    
    # 使用相同的thread_id来保持对话记忆
    config = {"configurable": {"thread_id": "conversation_1"}}
    
    # 第一轮对话
    print("\n[第1轮对话]")
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好，我叫张三"}]},
        config=config
    )
    print(f"用户: 你好，我叫张三")
    print(f"助手: {response1['messages'][-1].content}\n")
    
    # 第二轮对话 - agent会记住之前的对话
    print("[第2轮对话]")
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我今年25岁"}]},
        config=config
    )
    print(f"用户: 我今年25岁")
    print(f"助手: {response2['messages'][-1].content}\n")
    
    # 第三轮对话 - 测试记忆
    print("[第3轮对话]")
    response3 = agent.invoke(
        {"messages": [{"role": "user", "content": "你还记得我的名字吗？"}]},
        config=config
    )
    print(f"用户: 你还记得我的名字吗？")
    print(f"助手: {response3['messages'][-1].content}\n")
    
    # 显示完整的对话历史
    print("-" * 60)
    print("完整对话历史:")
    print("-" * 60)
    for msg in response3['messages']:
        if isinstance(msg, HumanMessage):
            print(f"用户: {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"助手: {msg.content}")
    print()


def multi_session_memory_example():
    """
    示例2: 多会话记忆 - 使用不同的thread_id管理多个独立会话
    
    每个thread_id都有独立的记忆空间
    """
    print("=" * 60)
    print("示例2: 多会话记忆 (不同thread_id)")
    print("=" * 60)
    
    # 创建模型
    model = create_siliconflow_model()
    
    # 创建checkpointer
    checkpointer = InMemorySaver()
    
    # 创建agent
    agent = create_agent(
        model=model,
        tools=[get_user_info],
        checkpointer=checkpointer,
    )
    
    # 会话1：用户张三
    print("\n[会话1 - 用户张三]")
    config1 = {"configurable": {"thread_id": "session_zhang"}}
    
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好，我叫张三，我住在北京"}]},
        config=config1
    )
    print(f"用户: 你好，我叫张三，我住在北京")
    print(f"助手: {response1['messages'][-1].content}\n")
    
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我住在哪里？"}]},
        config=config1
    )
    print(f"用户: 我住在哪里？")
    print(f"助手: {response2['messages'][-1].content}\n")
    
    # 会话2：用户李四（独立的记忆空间）
    print("[会话2 - 用户李四]")
    config2 = {"configurable": {"thread_id": "session_li"}}
    
    response3 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好，我叫李四，我住在上海"}]},
        config=config2
    )
    print(f"用户: 你好，我叫李四，我住在上海")
    print(f"助手: {response3['messages'][-1].content}\n")
    
    response4 = agent.invoke(
        {"messages": [{"role": "user", "content": "我住在哪里？"}]},
        config=config2
    )
    print(f"用户: 我住在哪里？")
    print(f"助手: {response4['messages'][-1].content}\n")
    
    # 验证会话1的记忆仍然存在
    print("[验证会话1的记忆]")
    response5 = agent.invoke(
        {"messages": [{"role": "user", "content": "我的名字是什么？"}]},
        config=config1
    )
    print(f"用户: 我的名字是什么？")
    print(f"助手: {response5['messages'][-1].content}\n")
    
    print("说明: 每个thread_id都有独立的记忆空间，互不干扰")
    print()


def persistent_memory_example():
    """
    示例3: 持久化记忆 - 展示如何在多次调用间保持记忆
    
    即使重新创建agent，只要使用相同的checkpointer和thread_id，记忆就会保留
    """
    print("=" * 60)
    print("示例3: 持久化记忆 (跨调用保持)")
    print("=" * 60)
    
    # 创建checkpointer（可以共享）
    checkpointer = InMemorySaver()
    
    # 第一次创建agent并进行对话
    print("\n[第一次调用]")
    model1 = create_siliconflow_model()
    agent1 = create_agent(
        model=model1,
        tools=[get_user_info],
        checkpointer=checkpointer,
    )
    
    config = {"configurable": {"thread_id": "persistent_session"}}
    
    response1 = agent1.invoke(
        {"messages": [{"role": "user", "content": "你好，我叫王五，我喜欢编程"}]},
        config=config
    )
    print(f"用户: 你好，我叫王五，我喜欢编程")
    print(f"助手: {response1['messages'][-1].content}\n")
    
    # 第二次调用 - 使用相同的checkpointer和thread_id
    # 即使重新创建agent，记忆也会保留
    print("[第二次调用 - 重新创建agent]")
    model2 = create_siliconflow_model()
    agent2 = create_agent(
        model=model2,
        tools=[get_user_info],
        checkpointer=checkpointer,  # 使用相同的checkpointer
    )
    
    response2 = agent2.invoke(
        {"messages": [{"role": "user", "content": "你还记得我的名字和爱好吗？"}]},
        config=config  # 使用相同的thread_id
    )
    print(f"用户: 你还记得我的名字和爱好吗？")
    print(f"助手: {response2['messages'][-1].content}\n")
    
    print("说明: 只要使用相同的checkpointer和thread_id，记忆就会跨调用保持")
    print()


def memory_with_tools_example():
    """
    示例4: 带工具的记忆 - agent可以记住之前的工具调用结果
    """
    print("=" * 60)
    print("示例4: 带工具的记忆")
    print("=" * 60)
    
    # 创建模型
    model = create_siliconflow_model()
    
    # 创建checkpointer
    checkpointer = InMemorySaver()
    
    # 创建agent，包含工具
    agent = create_agent(
        model=model,
        tools=[get_user_info, calculate_age],
        checkpointer=checkpointer,
    )
    
    config = {"configurable": {"thread_id": "tool_memory_session"}}
    
    # 第一轮：使用工具获取信息
    print("\n[第1轮 - 使用工具]")
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "获取张三的信息"}]},
        config=config
    )
    print(f"用户: 获取张三的信息")
    print(f"助手: {response1['messages'][-1].content}\n")
    
    # 第二轮：询问之前获取的信息
    print("[第2轮 - 询问之前的信息]")
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "张三的职业是什么？"}]},
        config=config
    )
    print(f"用户: 张三的职业是什么？")
    print(f"助手: {response2['messages'][-1].content}\n")
    
    # 第三轮：计算年龄
    print("[第3轮 - 计算年龄]")
    response3 = agent.invoke(
        {"messages": [{"role": "user", "content": "如果张三1999年出生，他今年多少岁？"}]},
        config=config
    )
    print(f"用户: 如果张三1999年出生，他今年多少岁？")
    print(f"助手: {response3['messages'][-1].content}\n")
    
    print()


def main():
    """主函数"""
    print("LangChain 记忆（Memory）功能示例")
    print("使用 create_agent 和 InMemorySaver")
    print("=" * 60)
    print()
    
    try:
        # 运行各种记忆示例
        basic_memory_example()
        multi_session_memory_example()
        persistent_memory_example()
        memory_with_tools_example()
        
        print("=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        print("\n记忆功能总结：")
        print("1. InMemorySaver: 在内存中保存对话历史")
        print("2. thread_id: 每个thread_id有独立的记忆空间")
        print("3. 持久化: 使用相同的checkpointer和thread_id可以跨调用保持记忆")
        print("4. 工具记忆: agent可以记住之前的工具调用结果")
        print("\n使用方式：")
        print("```python")
        print("from langchain.agents import create_agent")
        print("from langgraph.checkpoint.memory import InMemorySaver")
        print("")
        print("checkpointer = InMemorySaver()")
        print("agent = create_agent(")
        print("    model=model,")
        print("    tools=[...],")
        print("    checkpointer=checkpointer,")
        print(")")
        print("")
        print("config = {'configurable': {'thread_id': '1'}}")
        print("agent.invoke({'messages': [...]}, config=config)")
        print("```")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请确保：")
        print("1. 已设置环境变量 SILICONFLOW_API_KEY")
        print("2. 网络连接正常")
        print("3. API密钥有效")


if __name__ == "__main__":
    main()
