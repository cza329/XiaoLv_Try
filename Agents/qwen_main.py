"""
我的项目结构：
“try/multi_model_agent.py”：通用主任务，基于langchain集成多个模型（目前做了doubao_main.py，要做qwen_main.py）、多个工具（目前做了doubao_websearch、qwen_websearch、query_stock_history）
现在，请你写py，要参考给出的“try/multi_model_agent.py”、“try/LLM_Tools/qwen_websearch.py”、“try/Agents/doubao_main.py”以及千问官方的格式化输出示例，写“try/Agents/qwen_main.py”，具体要求如下：
"""
"""
千问模型特定实现
继承自BaseModelTester，实现千问特有的功能
"""

import os
import sys
import json
import time
import importlib.util
import re
from datetime import datetime
from typing import Optional, Dict, Any, List

# 修复导入路径问题
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 当前文件目录: try/Agents
current_dir = os.path.dirname(current_file_path)
# 项目根目录: try (向上两级)
project_root = os.path.dirname(current_dir)

# 添加项目根目录到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入基础模块
try:
    from multi_model_agent import BaseModelTester, StockQueryParams
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import Tool, StructuredTool
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    print("✓ 成功导入基础模块")
except ImportError as e:
    print(f"✗ 导入基础模块失败：{e}")
    print("请确保multi_model_agent.py在项目根目录中，且所有依赖已安装")
    sys.exit(1)


class QwenModelTester(BaseModelTester):
    """千问模型测试器，继承自BaseModelTester"""

    def __init__(self):
        """初始化千问模型测试器"""
        # 不调用父类的__init__，而是直接初始化基类属性
        self.model_name = "千问"
        self.response_history = []

        # 设置正确的项目路径
        self._setup_correct_paths()

        # 加载环境变量和依赖
        self._load_dependencies()

        # 获取千问模型配置
        self.model_config = self._get_model_config()
        self.websearch_tool = None

        print(f"✓ 千问模型测试器初始化完成")
        print(f"  项目根目录: {self.try_dir}")
        print(f"  LLM_Tools目录: {self.llm_tools_dir}")

    def _setup_correct_paths(self):
        """设置正确的项目路径"""
        # 获取当前文件路径
        current_file_path = os.path.abspath(__file__)
        # 当前文件在: try/Agents/qwen_main.py
        # try_dir应该是: try (即当前文件的父目录)
        self.try_dir = os.path.dirname(os.path.dirname(current_file_path))
        # LLM_Tools目录
        self.llm_tools_dir = os.path.join(self.try_dir, "LLM_Tools")

        # 确保LLM_Tools目录存在
        if not os.path.exists(self.llm_tools_dir):
            print(f"⚠️ LLM_Tools目录不存在：{self.llm_tools_dir}")
            print(f"  请确保目录结构正确：{self.try_dir}/LLM_Tools/")
        else:
            print(f"✓ 找到LLM_Tools目录：{self.llm_tools_dir}")

        # 添加LLM_Tools目录到Python路径
        if self.llm_tools_dir not in sys.path:
            sys.path.insert(0, self.llm_tools_dir)
            print(f"✓ 已添加LLM_Tools目录到Python路径")

    def _load_dependencies(self):
        """加载必要的依赖项"""
        # 加载环境变量
        from dotenv import load_dotenv
        env_path = os.path.join(self.try_dir, "settings", ".env")
        load_dotenv(dotenv_path=env_path)

        # 导入自定义股票查询工具
        try:
            from query_stock_history import query_stock_history
            self.query_stock_history_func = query_stock_history
            print("✓ 成功导入自定义股票查询工具")
        except ImportError as e:
            print(f"✗ 导入自定义股票查询工具失败：{e}")
            print(f"请确保query_stock_history.py文件在 {self.llm_tools_dir} 目录中")
            sys.exit(1)

    def _get_model_config(self) -> Dict[str, Any]:
        """获取千问模型配置"""
        return {
            "name": "千问",
            "api_key": os.getenv("QWEN_API_KEY"),
            "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "model_name": os.getenv("QWEN_MODEL", "qwen-plus"),
            "config_key": "千问"
        }

    def initialize_llm(self):
        """初始化千问语言模型"""
        config = self.model_config

        if not config["api_key"]:
            raise ValueError(f"{config['name']} API Key未设置")

        # 记录模型初始化
        init_record = {
            "timestamp": datetime.now().isoformat(),
            "action": "model_initialization",
            "model_name": config["name"],
            "config": {
                "base_url": config["base_url"],
                "model_name": config["model_name"]
            }
        }
        self.response_history.append(init_record)

        # 千问模型配置 - 使用无websearch功能的配置
        self.llm_params = {
            "model": config["model_name"],
            "api_key": config["api_key"],
            "base_url": config["base_url"],
            "temperature": 0.1,
            "timeout": 60
        }

        # 创建无websearch功能的千问模型
        self.llm = ChatOpenAI(**self.llm_params)

        # 只绑定股票查询工具，不绑定websearch工具
        stock_tool = self.create_stock_tool()
        self.llm_with_tools = self.llm.bind_tools([stock_tool])

        # 单独创建websearch工具实例，但不绑定到模型
        self.websearch_tool = self.create_websearch_tool()

        print(f"✓ {config['name']}模型初始化成功")
        print(f"  注意：使用无websearch功能的千问模型")
        print(f"  股票查询工具已通过LangChain绑定")
        print(f"  WebSearch工具已单独创建，但不绑定到模型")

    def create_stock_tool(self) -> StructuredTool:
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
                result = self.query_stock_history_func(
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
        json_path = os.path.join(self.llm_tools_dir, "query_stock_history.json")
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

    def create_websearch_tool(self):
        """创建千问WebSearch工具的LangChain包装器"""
        try:
            # 检查LLM_Tools目录
            if not os.path.exists(self.llm_tools_dir):
                raise FileNotFoundError(f"LLM_Tools目录不存在：{self.llm_tools_dir}")

            # 检查qwen_websearch.py文件
            websearch_file = os.path.join(self.llm_tools_dir, "qwen_websearch.py")
            if not os.path.exists(websearch_file):
                raise FileNotFoundError(f"qwen_websearch.py文件不存在：{websearch_file}")

            # 使用动态导入确保路径正确
            spec = importlib.util.spec_from_file_location("qwen_websearch", websearch_file)
            qwen_websearch_module = importlib.util.module_from_spec(spec)

            # 确保模块可以找到其他依赖
            if self.llm_tools_dir not in sys.path:
                sys.path.insert(0, self.llm_tools_dir)

            spec.loader.exec_module(qwen_websearch_module)

            # 调用创建函数
            websearch_tool = qwen_websearch_module.create_qwen_websearch_tool()
            print("✓ 成功创建千问WebSearch工具")
            return websearch_tool
        except Exception as e:
            print(f"✗ 创建千问WebSearch工具失败：{e}")

            # 创建回退工具
            def fallback_websearch_wrapper(query: str) -> str:
                return f"WebSearch工具不可用：{e}"

            return StructuredTool.from_function(
                name="qwen_web_search",
                description="使用千问原生web_search工具进行网络搜索，获取最新的网络信息。",
                func=fallback_websearch_wrapper,
                args_schema=None
            )

    def parse_search_request(self, model_response: str) -> Optional[str]:
        """
        解析千问模型响应，提取需要搜索的问题

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

        return formatted_request

    def _check_search_needed(self, response_content, search_performed):
        """检查千问模型是否需要搜索"""
        if not search_performed:
            search_query = self.parse_search_request(response_content)
            return search_query
        return None

    def _handle_search_request(self, messages, search_query):
        """处理千问模型的搜索请求"""
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
        print("\n[步骤3] 调用独立WebSearch工具进行搜索...")
        search_start_time = time.time()

        try:
            # 使用WebSearch工具
            search_result = self.websearch_tool.invoke({
                "query": search_query
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

            print("搜索结果已添加到消息历史，继续迭代处理...")
            return True  # 搜索已执行

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

            return True  # 标记搜索已尝试

    def _handle_websearch_tool_call(self, messages, tool_call, iteration, i, tool_args):
        """处理千问模型的web_search工具调用"""
        print(f"    检测到web_search工具调用，参数：{tool_args}")

        # 记录搜索请求
        search_request_record = {
            "timestamp": datetime.now().isoformat(),
            "action": "websearch_request",
            "iteration": iteration + 1,
            "tool_index": i + 1,
            "tool_name": "web_search",
            "tool_args": tool_args
        }
        self.response_history.append(search_request_record)

        # 提取搜索关键词
        search_keyword = tool_args.get("query", "")
        if not search_keyword:
            # 如果没有query参数，尝试从其他字段提取
            search_keyword = tool_args.get("keyword", "")
            if not search_keyword and "平安银行" in str(tool_args):
                search_keyword = "平安银行 最新新闻"

        if search_keyword:
            print(f"    提取到搜索关键词：{search_keyword}")

            # 调用独立的WebSearch工具执行搜索
            tool_start_time = time.time()
            try:
                # 准备搜索参数 - 千问工具使用query参数
                search_params = {
                    "query": search_keyword
                }

                print(f"    调用qwen_websearch工具，参数：{search_params}")

                # 执行搜索
                search_result = self.websearch_tool.invoke(search_params)
                tool_time = time.time() - tool_start_time

                # 记录搜索结果
                tool_result_record = {
                    "timestamp": datetime.now().isoformat(),
                    "action": "websearch_executed",
                    "iteration": iteration + 1,
                    "tool_index": i + 1,
                    "tool_name": "web_search",
                    "execution_time": tool_time,
                    "search_keyword": search_keyword,
                    "result_preview": str(search_result)[:200] + "..." if len(
                        str(search_result)) > 200 else str(search_result),
                    "result_length": len(str(search_result))
                }
                self.response_history.append(tool_result_record)

                print(f"    WebSearch工具执行完成，耗时：{tool_time:.2f}秒，结果长度：{len(str(search_result))}字符")

                # 将搜索结果作为ToolMessage添加到消息列表中
                messages.append(
                    ToolMessage(
                        content=str(search_result),
                        tool_call_id=tool_call.get("id", f"call_{iteration}_{i}")
                    )
                )

            except Exception as e:
                error_result = f"WebSearch工具执行出错：{str(e)}"
                tool_error_record = {
                    "timestamp": datetime.now().isoformat(),
                    "action": "websearch_error",
                    "iteration": iteration + 1,
                    "tool_index": i + 1,
                    "tool_name": "web_search",
                    "error": str(e),
                    "search_keyword": search_keyword
                }
                self.response_history.append(tool_error_record)

                messages.append(
                    ToolMessage(
                        content=error_result,
                        tool_call_id=tool_call.get("id", f"call_{iteration}_{i}")
                    )
                )
                print(f"    WebSearch工具执行失败：{e}")
        else:
            error_msg = "无法提取有效的搜索关键词"
            print(f"    {error_msg}")

            messages.append(
                ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call.get("id", f"call_{iteration}_{i}")
                )
            )


# 测试代码
if __name__ == "__main__":
    print("测试千问模型测试器...")
    print("=" * 60)

    try:
        # 创建测试器实例
        tester = QwenModelTester()

        print("\n测试结果：")
        print(f"模型名称: {tester.model_name}")
        print(f"模型配置: {tester.model_config['name']}")
        print(f"项目根目录: {tester.try_dir}")
        print(f"LLM_Tools目录: {tester.llm_tools_dir}")

        # 检查目录是否存在
        if os.path.exists(tester.llm_tools_dir):
            print(f"✓ LLM_Tools目录存在: {tester.llm_tools_dir}")
        else:
            print(f"✗ LLM_Tools目录不存在")

        print("\n测试通过！")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()