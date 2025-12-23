from langchain.tools import tool
import os
from pydantic import BaseModel, Field
from typing import Optional

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.
    
    Args:
        city: The name of the city to get weather for.
    
    Returns:
        A string describing the weather in the city.
    """
    return f"The weather in {city} is sunny."


from langchain.chat_models import init_chat_model

# 确保 base_url 以 / 结尾
base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1/")
if base_url and not base_url.endswith("/"):
    base_url += "/"

model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url=base_url,
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    temperature=0.7
)


from langchain.agents import create_agent


agent = create_agent(
    model,
    tools=[get_weather],
)

from langchain.messages import HumanMessage,AIMessage,SystemMessage

ai_message = AIMessage(content="The weather in Tokyo is sunny.")
system_message = SystemMessage(content="You are a weather assistant.")
human_message = HumanMessage(content="What is the weather in Tokyo?")

messages = [system_message, human_message, ai_message]

response = agent.invoke({
    "messages": messages
})
print(response)

# ========== 结构化输出示例 ==========
class WeatherInfo(BaseModel):
    """天气信息结构化数据"""
    city: str = Field(description="城市名称")
    temperature: int = Field(description="温度（摄氏度）")
    condition: str = Field(description="天气状况，如：晴天、多云、雨天等")
    humidity: Optional[int] = Field(default=None, description="湿度（百分比）")
    wind_speed: Optional[int] = Field(default=None, description="风速（公里/小时）")

# 创建支持结构化输出的模型
structured_model = model.with_structured_output(WeatherInfo)

# 使用结构化输出获取天气信息
weather_text = "今天北京的天气是晴天，温度25度，湿度60%，风速15公里每小时。"
weather_result = structured_model.invoke(f"请从以下文本中提取天气信息：{weather_text}")

print("\n========== 结构化输出结果 ==========")
print(f"城市: {weather_result.city}")
print(f"温度: {weather_result.temperature}°C")
print(f"天气状况: {weather_result.condition}")
print(f"湿度: {weather_result.humidity}%")
print(f"风速: {weather_result.wind_speed} km/h")
print(f"\n完整结构化对象: {weather_result}")

