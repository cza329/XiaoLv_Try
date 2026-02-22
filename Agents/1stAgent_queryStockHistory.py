"""
股票查询工具
使用无websearch功能的豆包/qwen模型，需要搜索时调用独立的doubao_websearch工具/qwen_websearch工具
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 获取当前脚本所在目录的父目录（try目录）
try_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(try_dir, "settings", ".env")
# 加载环境变量
load_dotenv(dotenv_path=env_path)

# 检查必要的环境变量
required_env_vars = [
    'STOCK_API_KEY',
    'ARK_API_KEY', 'ARK_BASE_URL', 'ARK_MODEL',
    'QWEN_API_KEY', 'QWEN_MODEL', 'QWEN_BASE_URL'
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"警告：以下环境变量未设置：{missing_vars}")
    print("请确保.env文件包含这些变量")

# 添加LLM_Tools目录到Python路径
llm_tools_dir = os.path.join(try_dir, "LLM_Tools")
if os.path.exists(llm_tools_dir):
    sys.path.insert(0, llm_tools_dir)
    print(f"✓ 已添加LLM_Tools目录到Python路径：{llm_tools_dir}")
else:
    print(f"⚠️ 未找到LLM_Tools目录：{llm_tools_dir}")
    print("请确保query_stock_history.py在LLM_Tools子目录下")
    sys.exit(1)

# 导入自定义股票查询工具
try:
    from query_stock_history import query_stock_history
    print("✓ 成功导入自定义股票查询工具")
except ImportError as e:
    print(f"✗ 导入自定义股票查询工具失败：{e}")
    print("请确保query_stock_history.py文件在LLM_Tools目录下")
    sys.exit(1)

# 导入豆包WebSearch工具
try:
    from doubao_websearch import create_doubao_websearch_tool
    print("✓ 成功导入豆包WebSearch工具")
except ImportError as e:
    print(f"✗ 导入豆包WebSearch工具失败：{e}")
    print("请确保doubao_websearch.py文件在LLM_Tools目录下")

# 导入LangChain相关模块
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import Tool, StructuredTool
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    print("✓ 成功导入LangChain模块")
except ImportError as e:
    print(f"✗ 导入LangChain模块失败：{e}")
    print("请安装必要的包：pip install langchain langchain-openai python-dotenv")
    sys.exit(1)

# 定义Pydantic模型用于结构化输出
class StockQueryParams(BaseModel):
    """股票查询参数的结构化输出模型"""
    symbol: str = Field(
        description="股票代码和市场后缀，例如 '000001.SZ'、'AAPL.US'"
    )
    start_date: Optional[str] = Field(
        None,
        description="开始日期，格式为 YYYYMMDD 或 YYYYMMDDhhmmss，例如 '20240601'。默认为None，表示查询全部历史"
    )
    end_date: Optional[str] = Field(
        None,
        description="结束日期，格式同 start_date。默认为None，表示查询到最新数据"
    )
    interval: str = Field(
        "d",
        description="分时级别。支持 '5', '15', '30', '60', 'd', 'w', 'm', 'y'，分别对应5分钟、15分钟、30分钟、60分钟、日线、周线、月线、年线。默认为 'd'（日线）"
    )
    adjust: str = Field(
        "n",
        description="除权方式。支持 'n' (不复权), 'f' (前复权), 'b' (后复权), 'fr' (等比前复权), 'br' (等比后复权)。分钟线只支持 'n'。默认为 'n'"
    )
    fields: Optional[List[str]] = Field(
        None,
        description="需要返回的字段列表。如果为None，则返回API所有字段。常用字段: ['t', 'o', 'h', 'l', 'c', 'v', 'a']"
    )

# 定义搜索请求的格式化输出模型
class SearchRequest(BaseModel):
    """搜索请求的结构化输出模型"""
    search_query: str = Field(description="需要搜索的问题关键词")

class ModelTester:
    """模型测试器，用于测试不同模型对股票查询工具的调用"""

    def __init__(self, model_type: int):
        """
        初始化模型测试器

        Args:
            model_type: 模型类型，1=豆包，2=千问
        """
        self.model_type = model_type
        self.model_config = self._get_model_config(model_type)
        self.llm = None
        self.llm_with_tools = None
        self.websearch_tool = None
        self.response_history = []

    def _get_model_config(self, model_type: int) -> Dict[str, Any]:
        """获取模型配置"""
        configs = {
            1: {
                "name": "豆包",
                "api_key": os.getenv("ARK_API_KEY"),
                "base_url": os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
                "model_name": os.getenv("ARK_MODEL", "doubao-seed-1-6-250615"),
                "config_key": "豆包"
            },
            2: {
                "name": "千问",
                "api_key": os.getenv("QWEN_API_KEY"),
                "base_url": os.getenv("QWEN_BASE_URL",
                                      "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                "model_name": os.getenv("QWEN_MODEL", "qwen-flash"),
                "config_key": "千问"
            }
        }

        if model_type not in configs:
            raise ValueError(f"不支持的模型类型：{model_type}，请使用1（豆包）或2（千问）")

        config = configs[model_type]

        if not config["api_key"]:
            raise ValueError(f"{config['name']} API Key未设置")

        return config

    def create_stock_tool(self) -> Tool:
        """创建股票查询工具的LangChain包装器"""

        def stock_tool_wrapper(
            symbol: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            interval: str = "d",
            adjust: str = "n",
            fields: Optional[List[str]] = None
        ) -> str:
            """包装函数，调用自定义股票查询工具"""
            try:
                # 记录工具调用开始
                tool_call_start = {
                    "timestamp": datetime.now().isoformat(),
                    "action": "tool_call_start",
                    "tool_name": "query_stock_history",
                    "parameters": {
                        "symbol": symbol,
                        "start_date": start_date,
                        "end_date": end_date,
                        "interval": interval,
                        "adjust": adjust,
                        "fields": fields
                    }
                }
                self.response_history.append(tool_call_start)

                # 调用导入的自定义函数
                result = query_stock_history(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    adjust=adjust,
                    fields=fields
                )

                # 记录工具调用结果
                tool_call_result = {
                    "timestamp": datetime.now().isoformat(),
                    "action": "tool_call_result",
                    "tool_name": "query_stock_history",
                    "result_preview": str(result)[:500] + "..." if len(str(result)) > 500 else str(result),
                    "result_length": len(str(result))
                }
                self.response_history.append(tool_call_result)

                return result
            except Exception as e:
                error_result = f"股票查询工具执行出错：{str(e)}"
                tool_error = {
                    "timestamp": datetime.now().isoformat(),
                    "action": "tool_call_error",
                    "tool_name": "query_stock_history",
                    "error": str(e)
                }
                self.response_history.append(tool_error)
                return error_result

        # 从JSON文件加载工具描述
        tool_description = "查询股票的历史交易数据，包括价格、成交量等信息。"

        # 尝试从LLM_Tools子目录下的JSON文件加载更详细的描述
        json_path = os.path.join(llm_tools_dir, "query_stock_history.json")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                if isinstance(json_data, dict) and 'function' in json_data:
                    tool_description = json_data['function'].get('description', tool_description)
        except FileNotFoundError:
            print(f"⚠️ 未找到query_stock_history.json文件在：{json_path}，使用默认工具描述")
        except json.JSONDecodeError as e:
            print(f"⚠️ query_stock_history.json文件格式错误：{e}，使用默认工具描述")
        except Exception as e:
            print(f"⚠️ 读取JSON文件时出错：{e}，使用默认工具描述")

        # 使用StructuredTool替代普通Tool，解决多参数传递问题
        return StructuredTool.from_function(
            name="query_stock_history",
            description=tool_description,
            func=stock_tool_wrapper,
            args_schema=StockQueryParams
        )

    def create_doubao_websearch_tool(self) -> Tool:
        """创建豆包WebSearch工具的LangChain包装器"""
        try:
            websearch_tool = create_doubao_websearch_tool()
            print("✓ 成功创建豆包WebSearch工具")
            return websearch_tool
        except Exception as e:
            print(f"✗ 创建豆包WebSearch工具失败：{e}")
            # 创建回退工具
            def fallback_websearch_wrapper(keyword: str, max_results: int = 2, max_keyword: int = 1) -> str:
                return f"豆包WebSearch工具不可用：{e}"

            return StructuredTool.from_function(
                name="doubao_websearch",
                description="使用豆包原生web_search工具进行网络搜索，获取最新的网络信息。",
                func=fallback_websearch_wrapper,
                args_schema=None
            )

    def create_qwen_websearch_tool(self) -> Tool:
        """创建千问WebSearch工具的LangChain包装器"""
        try:
            # 尝试导入qwen_websearch模块
            from qwen_websearch import create_qwen_websearch_tool as qwen_tool_creator
            websearch_tool = qwen_tool_creator()
            print("✓ 成功创建千问WebSearch工具")
            return websearch_tool
        except ImportError as e:
            print(f"✗ 导入千问WebSearch工具失败：{e}")
            print("请确保qwen_websearch.py文件在LLM_Tools目录下")
            # 创建回退工具
            def fallback_websearch_wrapper(query: str) -> str:
                return f"千问WebSearch工具不可用：{e}"

            return StructuredTool.from_function(
                name="qwen_websearch",
                description="使用千问原生web_search工具进行网络搜索，获取最新的网络信息。",
                func=fallback_websearch_wrapper,
                args_schema=SearchRequest
            )
        except Exception as e:
            print(f"✗ 创建千问WebSearch工具失败：{e}")
            # 创建回退工具
            def fallback_websearch_wrapper(query: str) -> str:
                return f"千问WebSearch工具不可用：{e}"

            return StructuredTool.from_function(
                name="qwen_websearch",
                description="使用千问原生web_search工具进行网络搜索，获取最新的网络信息。",
                func=fallback_websearch_wrapper,
                args_schema=SearchRequest
            )

    def _load_qwen_config_params(self) -> Dict[str, Any]:
        """加载千问配置文件中的参数"""
        config_dir = os.path.join(try_dir, "configs")
        config_path = os.path.join(config_dir, "qwen_config.json")

        default_params = {
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 0,
            "repetition_penalty": 1.05,
            "top_k": 1
        }

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                if "parameters" in config_data:
                    for param_name, param_config in config_data["parameters"].items():
                        if "use" in param_config and isinstance(param_config["use"], list) and param_config["use"]:
                            default_params[param_name] = param_config["use"][0]
                        elif "default" in param_config:
                            default_params[param_name] = param_config["default"]

                print(f"✓ 从配置文件加载千问参数: {default_params}")
            else:
                print(f"⚠️ 未找到千问配置文件: {config_path}，使用默认参数")

        except Exception as e:
            print(f"⚠️ 读取千问配置文件失败: {e}，使用默认参数")

        return default_params

    def initialize_llm(self):
        """初始化语言模型"""
        config = self.model_config

        # 记录模型初始化
        init_record = {
            "timestamp": datetime.now().isoformat(),
            "action": "model_initialization",
            "model_name": config["name"],
            "model_type": self.model_type,
            "config": {
                "base_url": config["base_url"],
                "model_name": config["model_name"]
            }
        }
        self.response_history.append(init_record)

        # 根据模型类型配置不同的参数
        if config["name"] == "豆包":
            # 豆包模型配置 - 使用无websearch功能的配置
            self.llm_params = {
                "model": config["model_name"],
                "api_key": config["api_key"],
                "base_url": config["base_url"],
                "temperature": 0.1,
                "timeout": 60
            }

            # 创建无websearch功能的豆包模型
            self.llm = ChatOpenAI(**self.llm_params)

            # 只绑定股票查询工具，不绑定websearch工具
            stock_tool = self.create_stock_tool()
            self.llm_with_tools = self.llm.bind_tools([stock_tool])

            # 单独创建websearch工具实例，但不绑定到模型
            self.websearch_tool = self.create_doubao_websearch_tool()

            print(f"✓ {config['name']}模型初始化成功")
            print(f"  - 注意：使用无websearch功能的豆包模型")
            print(f"  - 股票查询工具已通过LangChain绑定")
            print(f"  - WebSearch工具已单独创建，但不绑定到模型")

        else:  # 千问模型
            # 加载千问配置参数
            qwen_params = self._load_qwen_config_params()

            # 千问模型配置 - 使用无websearch功能的配置
            self.llm_params = {
                "model": config["model_name"],
                "api_key": config["api_key"],
                "base_url": config["base_url"],
                "temperature": qwen_params.get("temperature", 0.1),
                "top_p": qwen_params.get("top_p", 0.8),
                "presence_penalty": qwen_params.get("presence_penalty", 0),
                "timeout": 60
            }

            # 创建无websearch功能的千问模型
            # 注意：通过model_kwargs传递enable_search=False
            self.llm = ChatOpenAI(
                **self.llm_params,
                model_kwargs={
                    "enable_search": False,
                    "repetition_penalty": qwen_params.get("repetition_penalty", 1.05),
                    "top_k": qwen_params.get("top_k", 1)
                }
            )

            # 绑定股票查询工具（不绑定websearch工具）
            stock_tool = self.create_stock_tool()
            self.llm_with_tools = self.llm.bind_tools([stock_tool])

            # 单独创建千问websearch工具实例
            self.websearch_tool = self.create_qwen_websearch_tool()

            print(f"✓ {config['name']}模型初始化成功")
            print(f"  - 注意：使用无websearch功能的千问模型 (enable_search=False)")
            print(f"  - 股票查询工具已通过LangChain绑定")
            print(f"  - WebSearch工具已单独创建，但不绑定到模型")

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        # 尝试从配置文件加载系统提示词
        config_files = {
            1: "doubao_config.json",
            2: "qwen_config.json"
        }

        config_file = config_files.get(self.model_type)
        if config_file:
            # 构建配置文件的完整路径
            config_dir = os.path.join(try_dir, "configs")
            config_path = os.path.join(config_dir, config_file)

            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                        if "system_prompt" in config_data:
                            system_prompt = config_data["system_prompt"]
                            print(f"✓ 从 {config_path} 成功加载系统提示词")
                            return system_prompt
                        else:
                            print(f"⚠️ 配置文件 {config_path} 中没有找到 system_prompt 字段")
                except json.JSONDecodeError as e:
                    print(f"⚠️ 配置文件 {config_path} 格式错误：{e}")
                except Exception as e:
                    print(f"⚠️ 读取 {config_path} 失败：{e}")
            else:
                print(f"⚠️ 配置文件不存在：{config_path}")

        # 根据模型类型返回不同的默认系统提示词
        if self.model_type == 2:  # 千问模型
            print("⚠️ 使用千问模型默认系统提示词")
            return """你是一个专业的金融助手，可以调用工具来获取股票数据和最新信息。

当用户询问股票相关问题时，请遵循以下原则：
1. 首先分析得到用户所需查询的时间范围，然后，若你知道用户问的公司对应的股票代码、所属市场，则【不需要】使用web_search工具，若你不清楚则使用web_search工具获取（sh表示上证，sz表示深证）
2. 使用query_stock_history工具获取准确的股票交易数据
3. 无论用户是否明确要求，都务必使用web_search工具获取2条该公司的最新新闻并做简报，【禁止使用已有知识生成新闻】
4. 基于获取的数据与新闻提供专业、准确的简析

重要：当你判断需要搜索时，请生成格式化的搜索请求，使用SearchRequest模型输出搜索关键词。

请以清晰、专业的方式回答用户问题，确保数据、新闻准确无误。"""
        else:  # 豆包模型
            print("⚠️ 使用豆包模型默认系统提示词")
            return """你是一个专业的金融助手，可以调用工具来获取股票数据和最新信息。

当用户询问股票相关问题时，请遵循以下原则：
1. 首先分析得到用户所需查询的时间范围，然后，若你知道用户问的公司对应的股票代码、所属市场，则【不需要】使用web_search工具，若你不清楚则使用web_search工具获取（sh表示上证，sz表示深证）
2. 使用query_stock_history工具获取准确的股票交易数据
3. 无论用户是否明确要求，都务必使用web_search工具获取2条该公司的最新新闻并做简报，【禁止使用已有知识生成新闻】
4. 基于获取的数据与新闻提供专业、准确的简析

请以清晰、专业的方式回答用户问题，确保数据、新闻准确无误。"""

    def parse_search_request(self, model_response: str) -> Optional[str]:
        """
        解析模型响应，提取需要搜索的问题

        Args:
            model_response: 模型的响应文本

        Returns:
            需要搜索的问题关键词，如果不需要搜索则返回None
        """
        # 查找格式化输出中的搜索请求部分
        search_markers = [
            "【需要搜索】",
            "[需要搜索]",
            "需要搜索：",
            "搜索关键词：",
            "请搜索：",
            "搜索问题："
        ]

        for marker in search_markers:
            if marker in model_response:
                # 提取标记后的内容
                start_idx = model_response.find(marker) + len(marker)
                # 找到行尾或下一个标记
                end_idx = len(model_response)
                for end_marker in ["\n", "。", "；", "，", " ", "【", "["]:
                    if end_marker in model_response[start_idx:]:
                        temp_end = model_response.find(end_marker, start_idx)
                        if temp_end != -1 and temp_end < end_idx:
                            end_idx = temp_end

                search_query = model_response[start_idx:end_idx].strip()
                if search_query:
                    return search_query

        # 如果没有找到明确标记，检查是否有需要搜索的暗示
        need_search_phrases = [
            "我需要搜索",
            "请帮我搜索",
            "查找相关信息",
            "获取最新新闻",
            "搜索新闻"
        ]

        for phrase in need_search_phrases:
            if phrase in model_response:
                # 尝试提取公司名称或关键词
                lines = model_response.split('\n')
                for line in lines:
                    if any(keyword in line.lower() for keyword in ["公司", "股票", "证券", "股份"]):
                        # 提取公司名称
                        import re
                        # 匹配中文公司名模式
                        company_pattern = r'[（(]?(.*?公司|.*?集团|.*?银行|.*?证券)[）)]?'
                        match = re.search(company_pattern, line)
                        if match:
                            return match.group(1) + " 最新新闻"

        return None

    def generate_formatted_search_request(self, search_query: str) -> str:
        """
        生成格式化的搜索请求

        Args:
            search_query: 需要搜索的问题

        Returns:
            格式化的搜索请求文本
        """
        model_name = self.model_config["name"]

        if model_name == "千问":
            formatted_request = f"""【需要搜索的问题】
{search_query}

【搜索要求】
1. 请使用千问原生web_search工具进行搜索
2. 搜索关键词应简洁明了，包含公司名称和"最新新闻"等关键词
3. 获取2条最新新闻并做简报
4. 确保新闻来源可靠、信息准确

【格式化输出】
请将搜索结果格式化为：
1. 新闻标题
2. 新闻摘要（不超过100字）
3. 新闻来源
4. 发布时间（如果可用）

【注意事项】
- 禁止使用已有知识生成新闻
- 确保新闻的时效性（尽量获取24小时内的新闻）
- 如果找不到最新新闻，请如实说明"""
        else:  # 豆包
            formatted_request = f"""【需要搜索的问题】
{search_query}

【搜索要求】
1. 请使用豆包原生web_search工具进行搜索
2. 搜索关键词应简洁明了，包含公司名称和"最新新闻"等关键词
3. 获取2条最新新闻并做简报
4. 确保新闻来源可靠、信息准确

【格式化输出】
请将搜索结果格式化为：
1. 新闻标题
2. 新闻摘要（不超过100字）
3. 新闻来源
4. 发布时间（如果可用）

【注意事项】
- 禁止使用已有知识生成新闻
- 确保新闻的时效性（尽量获取24小时内的新闻）
- 如果找不到最新新闻，请如实说明"""

        return formatted_request

    def run_structured_test_query(self, query: str) -> Dict[str, Any]:
        """运行结构化测试查询并返回完整结果"""
        config = self.model_config

        print(f"\n{'=' * 60}")
        print(f"开始结构化测试 {config['name']} 模型")
        print(f"测试查询：{query}")
        print(f"{'=' * 60}")

        # 记录查询开始
        query_start = {
            "timestamp": datetime.now().isoformat(),
            "action": "structured_query_start",
            "model": config["name"],
            "query": query
        }
        self.response_history.append(query_start)

        try:
            # 获取系统提示词
            system_prompt = self.get_system_prompt()

            # 创建初始消息列表
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]

            print(f"\n[步骤1] 使用{config['name']}模型处理查询...")

            # 记录初始消息
            initial_messages_record = {
                "timestamp": datetime.now().isoformat(),
                "action": "initial_messages",
                "system_prompt": system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt,
                "user_query": query
            }
            self.response_history.append(initial_messages_record)

            # 工具调用循环处理
            max_iterations = 5
            final_output = None
            total_iterations = 0
            last_response = None
            search_performed = False

            for iteration in range(max_iterations):
                total_iterations = iteration + 1
                print(f"\n[迭代 {iteration + 1}/{max_iterations}]")

                # 记录迭代开始
                iteration_start = {
                    "timestamp": datetime.now().isoformat(),
                    "action": "iteration_start",
                    "iteration": iteration + 1
                }
                self.response_history.append(iteration_start)

                # 调用模型
                print("调用模型中...")
                start_time = time.time()

                try:
                    # 对于千问模型，使用支持结构化输出的方式
                    if config["name"] == "千问" and iteration == 0:
                        # 第一次迭代，绑定SearchRequest模型用于结构化输出
                        llm_with_structured_output = self.llm.bind_tools(
                            [self.create_stock_tool()],
                            response_format=SearchRequest
                        )
                        response = llm_with_structured_output.invoke(messages)
                    else:
                        response = self.llm_with_tools.invoke(messages)

                    last_response = response
                    response_time = time.time() - start_time

                    # 记录模型响应
                    model_response_record = {
                        "timestamp": datetime.now().isoformat(),
                        "action": "model_response",
                        "iteration": iteration + 1,
                        "response_time": response_time,
                        "response_type": type(response).__name__,
                        "has_tool_calls": hasattr(response, 'tool_calls') and response.tool_calls,
                        "response_preview": str(response.content)[:200] + "..." if hasattr(response, 'content') and len(str(response.content)) > 200 else str(response.content)[:200] if hasattr(response, 'content') else str(response)[:200],
                        "response_length": len(str(response.content)) if hasattr(response, 'content') else len(str(response))
                    }

                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        model_response_record["tool_calls"] = [
                            {
                                "name": tc.get("name", "未知"),
                                "args": tc.get("args", {})
                            }
                            for tc in response.tool_calls
                        ]

                    self.response_history.append(model_response_record)

                    print(f"模型响应时间：{response_time:.2f}秒")

                except Exception as e:
                    error_record = {
                        "timestamp": datetime.now().isoformat(),
                        "action": "model_error",
                        "iteration": iteration + 1,
                        "error": str(e)
                    }
                    self.response_history.append(error_record)
                    print(f"模型调用出错：{e}")
                    break

                # 检查是否有工具调用
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    print(f"检测到工具调用：{len(response.tool_calls)}个")

                    # 将模型的响应添加到消息历史中
                    messages.append(response)

                    # 处理每个工具调用
                    for i, tool_call in enumerate(response.tool_calls):
                        tool_name = tool_call["name"]
                        tool_args = tool_call.get("args", {})

                        print(f"  处理工具调用 {i + 1}: {tool_name}")

                        # 记录工具调用详情
                        tool_call_record = {
                            "timestamp": datetime.now().isoformat(),
                            "action": "tool_call_detected",
                            "iteration": iteration + 1,
                            "tool_index": i + 1,
                            "tool_name": tool_name,
                            "tool_args": tool_args
                        }
                        self.response_history.append(tool_call_record)

                        # 根据工具名称执行相应的工具
                        if tool_name == "query_stock_history":
                            print(f"    执行股票查询工具，参数：{tool_args}")

                            # 创建股票查询工具实例
                            stock_tool = self.create_stock_tool()

                            # 执行工具调用
                            tool_start_time = time.time()
                            try:
                                tool_result = stock_tool.invoke(tool_args)
                                tool_time = time.time() - tool_start_time

                                # 记录工具执行结果
                                tool_result_record = {
                                    "timestamp": datetime.now().isoformat(),
                                    "action": "tool_call_result",
                                    "iteration": iteration + 1,
                                    "tool_index": i + 1,
                                    "tool_name": tool_name,
                                    "execution_time": tool_time,
                                    "result_preview": str(tool_result)[:200] + "..." if len(
                                        str(tool_result)) > 200 else str(tool_result),
                                    "result_length": len(str(tool_result))
                                }
                                self.response_history.append(tool_result_record)

                                print(
                                    f"    股票查询完成，执行时间：{tool_time:.2f}秒，结果长度：{len(str(tool_result))}字符")

                                # 将工具结果作为ToolMessage添加到消息列表中
                                messages.append(
                                    ToolMessage(
                                        content=str(tool_result),
                                        tool_call_id=tool_call.get("id", f"call_{iteration}_{i}")
                                    )
                                )

                            except Exception as e:
                                error_result = f"股票查询工具执行出错：{str(e)}"
                                tool_error_record = {
                                    "timestamp": datetime.now().isoformat(),
                                    "action": "tool_call_error",
                                    "iteration": iteration + 1,
                                    "tool_index": i + 1,
                                    "tool_name": tool_name,
                                    "error": str(e)
                                }
                                self.response_history.append(tool_error_record)

                                messages.append(
                                    ToolMessage(
                                        content=error_result,
                                        tool_call_id=tool_call.get("id", f"call_{iteration}_{i}")
                                    )
                                )
                                print(f"    股票查询失败：{e}")

                else:
                    # 没有工具调用，检查是否需要搜索
                    if hasattr(response, 'content'):
                        response_content = response.content
                    else:
                        response_content = str(response)

                    # 检查是否需要搜索（对于千问和豆包模型）
                    if not search_performed:
                        search_query = None

                        # 对于千问模型，尝试解析结构化输出
                        if config["name"] == "千问" and hasattr(response, 'parsed'):
                            try:
                                parsed_data = response.parsed
                                if isinstance(parsed_data, SearchRequest):
                                    search_query = parsed_data.search_query
                            except Exception as e:
                                print(f"解析结构化输出失败: {e}")

                        # 如果结构化输出没有结果，尝试文本解析
                        if not search_query:
                            search_query = self.parse_search_request(response_content)

                        if search_query:
                            print(f"\n[步骤2] 检测到需要搜索的问题：{search_query}")

                            # 记录搜索请求
                            search_request_record = {
                                "timestamp": datetime.now().isoformat(),
                                "action": "search_request_detected",
                                "search_query": search_query
                            }
                            self.response_history.append(search_request_record)

                            # 生成格式化搜索请求
                            formatted_search_request = self.generate_formatted_search_request(search_query)
                            print(f"生成的格式化搜索请求：\n{formatted_search_request[:200]}...")

                            # 调用独立的WebSearch工具
                            print(f"\n[步骤3] 调用独立{config['name']} WebSearch工具进行搜索...")
                            search_start_time = time.time()

                            try:
                                # 根据模型类型调用不同的工具
                                if config["name"] == "千问":
                                    # 千问websearch工具使用SearchRequest模型
                                    search_result = self.websearch_tool.invoke({
                                        "query": search_query
                                    })
                                else:  # 豆包
                                    search_result = self.websearch_tool.invoke({
                                        "keyword": search_query,
                                        "max_results": 2,
                                        "max_keyword": 1
                                    })

                                search_time = time.time() - search_start_time

                                # 记录搜索结果
                                search_result_record = {
                                    "timestamp": datetime.now().isoformat(),
                                    "action": "websearch_executed",
                                    "search_query": search_query,
                                    "execution_time": search_time,
                                    "result_length": len(search_result),
                                    "result_preview": search_result[:200] + "..." if len(search_result) > 200 else search_result
                                }
                                self.response_history.append(search_result_record)

                                print(f"WebSearch工具执行完成，耗时：{search_time:.2f}秒")
                                print(f"搜索结果长度：{len(search_result)}字符")

                                # 将搜索结果添加到消息历史中
                                messages.append(
                                    HumanMessage(content=f"请基于以下搜索结果完善回答：\n\n{search_result}")
                                )

                                search_performed = True
                                print("搜索结果已添加到消息历史，继续迭代处理...")
                                continue  # 继续下一轮迭代，让模型基于搜索结果完善回答

                            except Exception as e:
                                error_msg = f"WebSearch工具执行失败：{str(e)}"
                                print(f"✗ {error_msg}")

                                # 记录错误
                                search_error_record = {
                                    "timestamp": datetime.now().isoformat(),
                                    "action": "websearch_error",
                                    "search_query": search_query,
                                    "error": str(e)
                                }
                                self.response_history.append(search_error_record)

                                # 将错误信息添加到消息历史
                                messages.append(
                                    HumanMessage(content=f"WebSearch工具执行失败：{str(e)}，请基于已有信息继续回答。")
                                )

                                search_performed = True
                                continue  # 继续下一轮迭代

                    # 如果没有需要搜索或搜索已完成，则作为最终答案
                    if hasattr(response, 'content'):
                        final_output = response.content
                    else:
                        final_output = str(response)

                    print(f"获得最终答案，长度：{len(final_output)}字符")

                    # 记录最终输出
                    final_output_record = {
                        "timestamp": datetime.now().isoformat(),
                        "action": "final_output",
                        "iteration": iteration + 1,
                        "output_preview": final_output[:200] + "..." if len(final_output) > 200 else final_output,
                        "output_length": len(final_output)
                }
                    self.response_history.append(final_output_record)

                    break

            if final_output is None and last_response is not None:
                # 尝试从最后一次响应中提取内容
                if hasattr(last_response, 'content') and last_response.content:
                    final_output = last_response.content
                elif last_response and hasattr(last_response, '__str__'):
                    final_output = str(last_response)

            if final_output is None:
                final_output = f"达到最大迭代次数（{max_iterations}次），未获得完整答案。"
                print(f"⚠️ 达到最大迭代次数，使用部分结果")

            # 记录查询完成
            query_complete = {
                "timestamp": datetime.now().isoformat(),
                "action": "structured_query_complete",
                "model": config["name"],
                "total_iterations": total_iterations,
                "final_output_length": len(final_output),
                "history_entries": len(self.response_history),
                "search_performed": search_performed
            }
            self.response_history.append(query_complete)

            print(f"\n[步骤4] 生成最终输出完成")
            print(f"总迭代次数：{total_iterations}")
            print(f"最终输出长度：{len(final_output)}字符")
            if search_performed:
                print(f"搜索已执行：是")

            # 返回测试结果
            test_result = {
                "success": True,
                "model": config["name"],
                "model_type": self.model_type,
                "query": query,
                "final_output": final_output,
                "final_output_length": len(final_output),
                "total_iterations": total_iterations,
                "search_performed": search_performed,
                "history_entries": len(self.response_history),
                "response_history": self.response_history
            }

            return test_result

        except Exception as e:
            error_msg = f"结构化测试过程中出错：{str(e)}"
            print(f"✗ {error_msg}")

            # 记录错误
            error_record = {
                "timestamp": datetime.now().isoformat(),
                "action": "test_error",
                "model": config["name"],
                "error": str(e),
                "error_type": type(e).__name__
            }
            self.response_history.append(error_record)

            return {
                "success": False,
                "model": config["name"],
                "model_type": self.model_type,
                "query": query,
                "final_output": error_msg,
                "error": str(e),
                "history_entries": len(self.response_history),
                "response_history": self.response_history
            }

    def save_results(self, results: Dict[str, Any], output_dir: str = "test_results") -> str:
        """保存测试结果到文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_config["name"]
        filename = f"{model_name}_test_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)

            return filepath
        except Exception as e:
            print(f"保存结果失败：{e}")
            return ""

def display_menu() -> None:
    """显示菜单"""
    print("\n" + "=" * 60)
    print("股票查询工具结构化测试系统")
    print("=" * 60)
    print("请选择要测试的模型：")
    print("1. 豆包模型 (无websearch功能，使用独立WebSearch工具)")
    print("2. 千问模型 (无websearch功能，使用独立WebSearch工具)")
    print("3. 扩展预留位置 (待实现)")
    print("4. 扩展预留位置 (待实现)")
    print("5. 扩展预留位置 (待实现)")
    print("0. 退出")
    print("=" * 60)

def get_user_query(default_query: str) -> str:
    """获取用户查询"""
    print(f"\n默认查询：{default_query}")
    choice = input("是否使用默认查询？(y/n): ").strip().lower()

    if choice == 'y' or choice == '':
        return default_query
    else:
        custom_query = input("请输入您的查询：").strip()
        return custom_query if custom_query else default_query

def display_test_summary(result: Dict[str, Any]) -> None:
    """显示测试结果摘要"""
    print(f"\n{'=' * 60}")
    print("结构化测试结果摘要")
    print(f"{'=' * 60}")
    print(f"模型：{result['model']}")
    print(f"查询：{result['query']}")
    print(f"测试成功：{'是' if result['success'] else '否'}")

    if result['success']:
        print(f"总迭代次数：{result.get('total_iterations', 0)}")
        print(f"最终输出长度：{result['final_output_length']}字符")
        print(f"历史记录条目数：{result['history_entries']}")
        if 'search_performed' in result:
            print(f"搜索已执行：{'是' if result['search_performed'] else '否'}")

        print(f"\n最终输出预览：")
        print("-" * 40)
        output_preview = result['final_output'][:500] + "..." if len(result['final_output']) > 500 else result['final_output']
        print(output_preview)
        print("-" * 40)
    else:
        print(f"错误：{result.get('error', '未知错误')}")
        print(f"历史记录条目数：{result['history_entries']}")

def save_results_interactive(tester, result: Dict[str, Any]) -> None:
    """交互式保存结果"""
    save_choice = input("\n是否保存完整测试结果？(y/n): ").strip().lower()
    if save_choice == 'y':
        filepath = tester.save_results(result)
        if filepath:
            print(f"✓ 测试结果已保存到：{filepath}")

            view_choice = input("是否查看保存的文件路径？(y/n): ").strip().lower()
            if view_choice == 'y':
                print(f"\n文件保存路径：{os.path.abspath(filepath)}")

def display_history_summary(result: Dict[str, Any]) -> None:
    """显示历史记录摘要"""
    history_choice = input("\n是否显示历史记录摘要？(y/n): ").strip().lower()
    if history_choice == 'y':
        print(f"\n历史记录摘要（共{result['history_entries']}条）：")
        for i, record in enumerate(result.get('response_history', [])[:10], 1):
            action = record.get('action', '未知')
            timestamp = record.get('timestamp', '未知时间')
            time_str = timestamp[11:19] if isinstance(timestamp, str) and len(timestamp) > 19 else str(timestamp)
            print(f"{i:2d}. [{time_str}] {action}")

        if result['history_entries'] > 10:
            print(f"... 还有{result['history_entries'] - 10}条记录")

def run_model_test(model_type: int) -> None:
    """运行指定模型的测试"""
    try:
        # 获取用户查询
        default_query = "帮我查一下平安银行2026.2.12到2.14日的日收盘价和成交量"
        query = get_user_query(default_query)

        # 创建模型测试器
        tester = ModelTester(model_type)

        # 初始化模型
        tester.initialize_llm()

        # 运行测试
        print(f"\n开始执行结构化测试查询...")
        result = tester.run_structured_test_query(query)

        # 显示测试结果摘要
        display_test_summary(result)

        # 交互式保存结果
        save_results_interactive(tester, result)

        # 显示历史记录摘要
        display_history_summary(result)

        print(f"\n{result['model']}模型测试完成！")
        input("\n按Enter键继续...")

    except Exception as e:
        print(f"测试过程中出错：{e}")
        import traceback
        traceback.print_exc()
        input("\n按Enter键继续...")

def main() -> None:
    """主函数"""
    print("股票查询工具结构化测试脚本")
    print("版本：10.0.0（重构千问模型，使用无websearch功能配置和独立WebSearch工具）")
    print("功能：测试股票查询工具对豆包和千问模型的结构化输出支持")
    print("=" * 60)

    while True:
        display_menu()

        try:
            choice = int(input("\n请输入选择 (0-5): ").strip())

            if choice == 0:
                print("退出测试系统。")
                break
            elif choice == 1:
                print("\n" + "=" * 60)
                print("测试豆包模型 (无websearch功能，使用独立WebSearch工具)")
                print("=" * 60)
                run_model_test(1)
            elif choice == 2:
                print("\n" + "=" * 60)
                print("测试千问模型 (无websearch功能，使用独立WebSearch工具)")
                print("=" * 60)
                run_model_test(2)
            elif choice in [3, 4, 5]:
                print(f"\n预留位置 {choice}，当前版本暂未实现")
                print("未来可扩展支持更多模型")
            else:
                print("无效选择，请重新输入")

        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n用户中断操作")
            break
        except Exception as e:
            print(f"发生错误：{e}")

if __name__ == "__main__":
    main()
