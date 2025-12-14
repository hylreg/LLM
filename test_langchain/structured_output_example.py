#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
结构化输出示例程序

本程序演示如何使用LangChain和Pydantic从非结构化文本中提取结构化信息。
"""

import os
from typing import List
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model


class PersonInfo(BaseModel):
    """个人信息结构"""
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄")
    skills: List[str] = Field(description="技能列表")


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


def extract_person_info(text: str, model=None) -> PersonInfo:
    """
    从文本中提取人物信息
    
    Args:
        text: 包含人物信息的文本
        model: 可选的模型实例
        
    Returns:
        PersonInfo: 提取的人物信息结构化数据
        
    Raises:
        Exception: 当模型调用或解析失败时抛出
    """
    # 如果没有提供模型，则创建一个新的模型实例
    if model is None:
        model = create_siliconflow_model()
    
    # 创建支持结构化输出的模型
    structured_llm = model.with_structured_output(PersonInfo)
    
    # 构造提示词
    prompt = f"请根据以下文本提取信息。\n{text}\n请严格按照要求的格式返回JSON。"
    
    # 调用模型提取结构化信息
    result = structured_llm.invoke(prompt)
    
    return result


def main():
    """主函数"""
    # 测试输入文本
    test_texts = [
        "张三今年30岁，他擅长Python和Go语言。",
        "李四是一名25岁的设计师，精通Photoshop和Illustrator。",
        "王五是一位40岁的数据科学家，专长是Python、R和机器学习算法。"
    ]
    
    print("结构化输出示例程序")
    print("=" * 30)
    
    try:
        # 获取模型实例
        model = create_siliconflow_model()
        print("成功创建硅基流动模型实例\n")
        
        for i, text in enumerate(test_texts, 1):
            print(f"测试用例 {i}:")
            print(f"输入文本: {text}")
            print("正在提取结构化信息...")
            
            # 调用模型提取结构化信息
            result = extract_person_info(text, model)
            
            # 输出结果
            print("提取结果:")
            print(f"  姓名: {result.name}")
            print(f"  年龄: {result.age}")
            print(f"  技能: {', '.join(result.skills)}")
            print("-" * 30)
            
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()