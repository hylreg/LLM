import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


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


def demo_usage():
    """
    演示如何使用硅基流动模型
    """
    try:
        # 创建模型实例
        model = create_siliconflow_model()
        print("成功创建硅基流动模型实例")
        
        # 简单对话示例
        print("\n=== 简单对话示例 ===")
        messages = [HumanMessage(content="你好，请简单介绍一下你自己")]
        response = model.invoke(messages)
        print(f"用户: {messages[0].content}")
        print(f"模型回复: {response.content}")
        
        return model
    except Exception as e:
        print(f"出现错误: {e}")
        return None


if __name__ == "__main__":
    demo_usage()