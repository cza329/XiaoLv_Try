import os
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark

load_dotenv()
api_key = os.getenv("ARK_API_KEY")
base_url = os.getenv("ARK_BASE_URL")

client = Ark(
    base_url=base_url,
    api_key=api_key,
)

tools = [{
    "type": "web_search",
    "max_keyword": 2,
}]

# 创建一个对话请求
response = client.responses.create(
    model="doubao-seed-1-6-250615",
    input=[{"role": "user", "content": "今天有什么热点新闻？"}],
    tools=tools,
)

print(response)