"""
通义千问联网搜索工具
封装为LangChain Tool，供主程序在需要联网搜索时调用
"""
import json
import os
from typing import Type, Dict, Any
from dotenv import load_dotenv
from langchain.tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel, Field, PrivateAttr


class QwenWebSearchInput(BaseModel):
    """联网搜索工具的输入参数"""
    query: str = Field(description="需要搜索的关键词或问题")


class QwenWebSearchTool(BaseTool):
    """通义千问联网搜索工具"""

    name: str = "qwen_web_search"
    description: str = """使用通义千问的联网搜索功能获取最新信息。
    当需要获取实时数据、最新新闻、当前天气、股票信息等需要联网查询的信息时使用此工具。"""
    args_schema: Type[BaseModel] = QwenWebSearchInput

    # 使用私有属性存储配置，避免pydantic模型字段冲突
    _api_key: str = PrivateAttr(default="")
    _base_url: str = PrivateAttr(default="")
    _model: str = PrivateAttr(default="qwen-plus")
    _default_params: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _client: OpenAI = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 加载环境变量
        self._load_environment()
        # 加载配置参数
        self._load_config()
        # 初始化OpenAI客户端
        self._init_client()

    def _load_environment(self):
        """加载环境配置文件"""
        try:
            # 尝试从项目根目录的settings/.env文件加载环境变量
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            env_path = os.path.join(project_root, "settings", ".env")

            print(f"try_dir:{project_root}")
            print(f"env_path:{env_path}")

            if os.path.exists(env_path):
                load_dotenv(env_path)
                print(f"已加载.env文件: {env_path}")
            else:
                # 尝试从当前目录的上级查找
                load_dotenv()
                print("尝试从默认位置加载环境变量")

            # 获取环境变量
            self._api_key = os.getenv("QWEN_API_KEY")
            self._base_url = os.getenv("QWEN_BASE_URL")
            model_env = os.getenv("QWEN_MODEL")
            if model_env:
                self._model = model_env

            # 验证必要环境变量
            if not self._api_key:
                raise ValueError("QWEN_API_KEY 环境变量未设置")
            if not self._base_url:
                raise ValueError("QWEN_BASE_URL 环境变量未设置")

            print(f"成功加载环境变量: API_KEY={self._api_key[:10]}..., BASE_URL={self._base_url}, MODEL={self._model}")

        except Exception as e:
            print(f"加载环境变量时出错: {e}")
            raise

    def _load_config(self):
        """加载参数配置文件"""
        try:
            # 尝试从项目根目录的configs/qwen_config.json文件加载配置
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            config_path = os.path.join(project_root, "configs", "qwen_config.json")

            # 默认参数配置
            self._default_params = {
                "temperature": 0.7,
                "top_p": 0.8,
                "presence_penalty": 0,
                "repetition_penalty": 1.05,
                "top_k": 1
            }

            if os.path.exists(config_path):
                print(f"发现配置文件: {config_path}")
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # 获取参数配置中的use值
                if "parameters" in config:
                    for param_name, param_config in config["parameters"].items():
                        if "use" in param_config and isinstance(param_config["use"], list):
                            # 使用第一个可用的值
                            if param_config["use"]:
                                self._default_params[param_name] = param_config["use"][0]
                                print(f"从配置加载参数 {param_name}: {param_config['use'][0]}")
                        elif "default" in param_config:
                            self._default_params[param_name] = param_config["default"]
                            print(f"从配置加载参数 {param_name}: {param_config['default']}")
            else:
                print(f"未找到配置文件: {config_path}，使用默认参数")

        except Exception as e:
            print(f"读取配置文件失败，使用默认参数: {e}")

    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url
            )
            print("OpenAI客户端初始化成功")
        except Exception as e:
            print(f"初始化OpenAI客户端失败: {e}")
            raise

    def _run(self, query: str) -> str:
        """执行搜索操作"""
        try:
            # 构建消息
            messages = [
                {"role": "system", "content": "你是一个有用的助手，请使用联网搜索功能回答用户的问题。"},
                {"role": "user", "content": query}
            ]

            # 构建请求参数
            completion_params = {
                "model": self._model,
                "messages": messages,
                "extra_body": {"enable_search": True}
            }

            # 添加可选参数
            if "temperature" in self._default_params:
                completion_params["temperature"] = self._default_params["temperature"]
            if "top_p" in self._default_params:
                completion_params["top_p"] = self._default_params["top_p"]
            if "presence_penalty" in self._default_params:
                completion_params["presence_penalty"] = self._default_params["presence_penalty"]
            if "repetition_penalty" in self._default_params:
                # 注意：OpenAI API中通常使用frequency_penalty，但通义千问可能使用repetition_penalty
                completion_params["extra_body"]["repetition_penalty"] = self._default_params["repetition_penalty"]
            if "top_k" in self._default_params:
                completion_params["extra_body"]["top_k"] = self._default_params["top_k"]

            print(f"发送搜索请求: {query}")
            print(f"使用模型: {self._model}")

            # 发起请求
            completion = self._client.chat.completions.create(**completion_params)

            # 返回结果
            result = completion.choices[0].message.content
            print(f"搜索请求成功完成")
            return result

        except Exception as e:
            error_msg = f"搜索失败: {str(e)}"
            print(error_msg)
            return error_msg

    async def _arun(self, query: str) -> str:
        """异步执行搜索操作"""
        # 这里实现异步版本，当前使用同步版本
        return self._run(query)

    @property
    def model(self) -> str:
        """获取模型名称的只读属性"""
        return self._model

    @property
    def default_params(self) -> Dict[str, Any]:
        """获取默认参数的只读属性"""
        return self._default_params.copy()


# 创建工具的工厂函数
def create_qwen_websearch_tool() -> QwenWebSearchTool:
    """
    创建并返回一个配置好的通义千问联网搜索工具

    Returns:
        QwenWebSearchTool: 配置好的搜索工具实例
    """
    return QwenWebSearchTool()


# 测试代码
if __name__ == "__main__":
    # 测试工具功能
    print("测试通义千问联网搜索工具...")

    try:
        # 创建工具实例
        tool = create_qwen_websearch_tool()
        print("工具创建成功!")
        print(f"工具名称: {tool.name}")
        print(f"工具描述: {tool.description}")
        print(f"使用模型: {tool.model}")  # 通过属性访问
        print(f"参数配置: {tool.default_params}")  # 通过属性访问

        # 测试搜索功能
        print("\n开始测试搜索功能...")

        # 测试用例1: 查询天气
        test_query1 = "杭州明天天气如何"
        print(f"\n测试查询1: {test_query1}")
        result1 = tool.run(test_query1)
        print(f"搜索结果1:\n{result1}")

        # 测试用例2: 查询新闻
        test_query2 = "今天有哪些重要的科技新闻"
        print(f"\n测试查询2: {test_query2}")
        result2 = tool.run(test_query2)
        print(f"搜索结果2:\n{result2}")

        # 测试用例3: 查询具体事实
        test_query3 = "2026年春节是几月几号"
        print(f"\n测试查询3: {test_query3}")
        result3 = tool.run(test_query3)
        print(f"搜索结果3:\n{result3}")

        print("\n所有测试完成!")

    except ValueError as e:
        print(f"配置错误: {e}")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")

'''
prompts_元宝：
同doubao_websearch.py，把doubao换成qwen
'''