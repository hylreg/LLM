from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import Tool
from langchain.messages import HumanMessage,AIMessage,SystemMessage
from langchain.agents import create_agent
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any

class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState

model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    temperature=0.7
)

agent = create_agent(
    model,
    tools=[Tool(name="get_weather", func=get_weather)],
    middleware=[CustomMiddleware()],
    state_schema=CustomState,
)


# 删除消息
