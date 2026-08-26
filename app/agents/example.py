from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


load_dotenv()


# 连接sqlite
connection = sqlite3.connect("resources/personal_chief.db", check_same_thread=False)
# 初始化checkpointer
checkpointer = SqliteSaver(connection)
# 自动建表
checkpointer.setup()

# 2.web搜索工具，使用tavily作为web搜索工具
web_search = TavilySearch(
    max_results=5,
    topic="general"
)

# 3.多模态模型
model = init_chat_model(
    model="qwen3.7-plus-2026-05-26",  # 模型名称，这是一个多模态模型，支持图片、文本、音频、视频
    model_provider="openai",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 4.Agent系统提示词
system_prompt = """
你是一名私人厨师。收到用户提供的食材照片或清单后，请按以下流程操作：
1.识别和评估食材：若用户提供照片，首先辨识所有可见食材。基于食材的外观状态，评估其新鲜度与可用量，整理出一份“当前可用食材清单”。
2.智能食谱检索：优先调用 web_search 工具，以“可用食材清单”为核心关键词，查找可行菜谱。
3.多维度评估与排序：从营养价值和制作难度两个维度对检索到的候选食谱进行量化打分，并根据得分排序，制作简单且营养丰富的排名靠前。
4.结构化方案输出：把排序后的食谱整理为一份结构清晰的建议报告，要包含食谱信息、得分、推荐理由、食谱的参考图片，帮助用户快速做出决策。

请严格按照流程，优先调用 web_search 工具搜索食谱，搜索不到的情况下才能自己发挥。
"""

# 5.创建Agent
agent = create_agent(
    model=model,  # 模型
    tools=[web_search],  # 工具
    system_prompt=system_prompt,  # 系统提示词
    checkpointer=checkpointer
)



from langchain.messages import HumanMessage

# 准备多模态消息，图片是网络上的冰箱食物图片
multimodal_message = HumanMessage(
    content=[
        {"type": "image",
         "url": "https://img.freepik.com/free-photo/arrangement-different-foods-organized-fridge_23-2149099882.jpg"},
        {"type": "text", "text": "帮我看看这些食材能做些什么？"}
    ])

config = {"configurable": {"thread_id": "6"}}

response = agent.invoke({"messages": [multimodal_message]}, config)

# 友好打印
for message in response['messages']:
    message.pretty_print()

print("\n\n\n")

response = agent.invoke(
    {"messages": [HumanMessage(content="我喜欢第1道菜，可以说的更详细点吗？")]},
    config
)

# 友好打印
response['messages'][-1].pretty_print()