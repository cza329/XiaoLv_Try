import env
from volcenginesdkarkruntime import Ark

# 从环境变量中获取您的API KEY，配置方法见：https://www.volcengine.com/docs/82379/1399008
api_key = env.ARK_API_KEY

client = Ark(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
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