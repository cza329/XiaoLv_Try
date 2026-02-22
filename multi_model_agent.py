"""
多模型智能代理通用框架
包含所有模型通用的测试逻辑和基础类
"""

import os
import sys
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool, StructuredTool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


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


class BaseModelTester(ABC):
    """模型测试器基类，包含所有模型通用的方法和属性"""

    def __init__(self, model_name: str):
        """
        初始化模型测试器基类

        Args:
            model_name: 模型名称，如'豆包'、'千问'
        """
        self.model_name = model_name
        self.llm = None
        self.llm_with_tools = None
        self.response_history = []
        self.query_stock_history_func = None  # 添加这个属性

        # 获取项目根目录 - 基于当前文件位置
        self.try_dir = os.path.dirname(os.path.abspath(__file__))
        self.llm_tools_dir = os.path.join(self.try_dir, "LLM_Tools")

        # 添加Agents目录到Python路径，以便可以导入子模块
        self.agents_dir = os.path.join(self.try_dir, "Agents")
        if os.path.exists(self.agents_dir) and self.agents_dir not in sys.path:
            sys.path.insert(0, self.agents_dir)
            print(f"✓ 已添加Agents目录到Python路径：{self.agents_dir}")

        # 加载环境变量
        self._load_environment()

    def _load_environment(self):
        """加载环境变量并检查必要的依赖"""
        env_path = os.path.join(self.try_dir, "settings", ".env")
        load_dotenv(dotenv_path=env_path)

        # 添加LLM_Tools目录到Python路径
        if os.path.exists(self.llm_tools_dir):
            sys.path.insert(0, self.llm_tools_dir)
            print(f"✓ 已添加LLM_Tools目录到Python路径：{self.llm_tools_dir}")
        else:
            print(f"⚠️ 未找到LLM_Tools目录：{self.llm_tools_dir}")
            sys.exit(1)

        # 导入自定义股票查询工具
        try:
            from query_stock_history import query_stock_history
            self.query_stock_history_func = query_stock_history
            print("✓ 成功导入自定义股票查询工具")
        except ImportError as e:
            print(f"✗ 导入自定义股票查询工具失败：{e}")
            sys.exit(1)

    def _get_config_filename(self, model_name: str) -> str:
        """
        根据模型中文名获取对应的配置文件名

        Args:
            model_name: 模型的中文名称

        Returns:
            对应的配置文件名
        """
        # 中文模型名到配置文件的映射
        model_config_map = {
            "豆包": "doubao_config.json",
            "千问": "qwen_config.json",
            "元宝": "yuanbao_config.json"
        }

        # 如果模型名在映射中，使用映射的配置文件名，否则使用模型名小写
        config_file = model_config_map.get(model_name, f"{model_name.lower()}_config.json")
        print(f"模型 '{model_name}' 对应的配置文件名: {config_file}")
        return config_file

    def get_system_prompt(self) -> str:
        """获取系统提示词（通用实现）"""
        # 获取配置文件路径
        config_dir = os.path.join(self.try_dir, "configs")
        config_file = self._get_config_filename(self.model_name)
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

        # 默认系统提示词
        print("⚠️ 使用默认系统提示词")
        return """你是一个专业的金融助手，可以调用工具来获取股票数据和最新信息。

当用户询问股票相关问题时，请遵循以下原则：
1. 首先分析得到用户所需查询的时间范围，然后，若你知道用户问的公司对应的股票代码、所属市场，则【不需要】使用web_search工具，若你不清楚则使用web_search工具获取（sh表示上证，sz表示深证）
2. 使用query_stock_history工具获取准确的股票交易数据
3. 无论用户是否明确要求，都务必使用web_search工具获取2条该公司的最新新闻并做简报，【禁止使用已有知识生成新闻】
4. 基于获取的数据与新闻提供专业、准确的简析

请以清晰、专业的方式回答用户问题，确保数据、新闻准确无误。"""

    def create_stock_tool(self) -> StructuredTool:
        """创建股票查询工具的LangChain包装器（通用实现）"""
        if self.query_stock_history_func is None:
            raise ValueError("股票查询工具函数未初始化")

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

    @abstractmethod
    def initialize_llm(self):
        """初始化语言模型（抽象方法，由子类实现）"""
        pass

    @abstractmethod
    def create_websearch_tool(self) -> Tool:
        """创建WebSearch工具（抽象方法，由子类实现）"""
        pass

    def run_structured_test_query(self, query: str) -> Dict[str, Any]:
        """运行结构化测试查询并返回完整结果（通用主逻辑）"""
        print(f"\n{'=' * 60}")
        print(f"开始结构化测试 {self.model_name} 模型")
        print(f"测试查询：{query}")
        print(f"{'=' * 60}")

        # 记录查询开始
        query_start = {
            "timestamp": datetime.now().isoformat(),
            "action": "structured_query_start",
            "model": self.model_name,
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

            print(f"\n[步骤1] 使用{self.model_name}模型处理查询...")

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
                        "response_preview": str(response.content)[:200] + "..." if hasattr(response, 'content') and len(
                            str(response.content)) > 200 else str(response.content)[:200] if hasattr(response,
                                                                                                     'content') else str(
                            response)[:200],
                        "response_length": len(str(response.content)) if hasattr(response, 'content') else len(
                            str(response))
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

                        elif tool_name == "web_search":
                            print(f"    检测到web_search工具调用，参数：{tool_args}")

                            # 记录搜索请求
                            search_request_record = {
                                "timestamp": datetime.now().isoformat(),
                                "action": "websearch_request",
                                "iteration": iteration + 1,
                                "tool_index": i + 1,
                                "tool_name": tool_name,
                                "tool_args": tool_args
                            }
                            self.response_history.append(search_request_record)

                            # 处理web_search工具调用（由子类实现）
                            self._handle_websearch_tool_call(messages, tool_call, iteration, i, tool_args)

                        else:
                            # 其他未知工具的处理
                            print(f"    检测到未知工具调用：{tool_name}")
                            tool_result = f"未知工具 '{tool_name}'，无法执行。"

                            messages.append(
                                ToolMessage(
                                    content=tool_result,
                                    tool_call_id=tool_call.get("id", f"call_{iteration}_{i}")
                                )
                            )

                else:
                    # 没有工具调用，检查是否需要搜索
                    if hasattr(response, 'content'):
                        response_content = response.content
                    else:
                        response_content = str(response)

                    # 检查是否需要搜索（由子类实现）
                    search_query = self._check_search_needed(response_content, search_performed)

                    if search_query:
                        search_performed = self._handle_search_request(messages, search_query)
                        if search_performed:
                            continue  # 继续下一轮迭代，让模型基于搜索结果完善回答

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
                "model": self.model_name,
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
                "model": self.model_name,
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
                "model": self.model_name,
                "error": str(e),
                "error_type": type(e).__name__
            }
            self.response_history.append(error_record)

            return {
                "success": False,
                "model": self.model_name,
                "query": query,
                "final_output": error_msg,
                "error": str(e),
                "history_entries": len(self.response_history),
                "response_history": self.response_history
            }

    def _handle_websearch_tool_call(self, messages, tool_call, iteration, i, tool_args):
        """处理web_search工具调用（由子类实现）"""
        pass

    def _check_search_needed(self, response_content, search_performed):
        """检查是否需要搜索（由子类实现）"""
        return None

    def _handle_search_request(self, messages, search_query):
        """处理搜索请求（由子类实现）"""
        return False

    def save_results(self, results: Dict[str, Any], output_dir: str = "test_results") -> str:
        """保存测试结果到文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_name
        filename = f"{model_name}_test_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)

            return filepath
        except Exception as e:
            print(f"保存结果失败：{e}")
            return ""


# 通用的UI交互函数
def display_menu() -> None:
    """显示菜单"""
    print("\n" + "=" * 60)
    print("股票查询工具结构化测试系统")
    print("=" * 60)
    print("请选择要测试的模型：")
    print("1. 豆包模型 (无websearch功能，使用独立WebSearch工具)")
    print("2. 千问模型 (支持结构化输出和原生web_search)")
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
        output_preview = result['final_output'][:500] + "..." if len(result['final_output']) > 500 else result[
            'final_output']
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


def run_model_test(tester) -> None:
    """运行模型测试（通用函数）"""
    try:
        # 获取用户查询
        default_query = "帮我查一下思特奇2026.2.12到2.14日的日收盘价和最高价，还要思特奇2026年2月的至少2条最新新闻及新闻发布时间。"
        query = get_user_query(default_query)

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

        print(f"\n{tester.model_name}模型测试完成！")
        input("\n按Enter键继续...")

    except Exception as e:
        print(f"测试过程中出错：{e}")
        import traceback
        traceback.print_exc()
        input("\n按Enter键继续...")


def main() -> None:
    """主函数（通用入口）"""
    print("股票查询工具结构化测试脚本")
    print("版本：9.0.0（重构豆包模型，使用独立WebSearch工具）")
    print("功能：测试股票查询工具对豆包和千问模型的结构化输出支持")
    print("=" * 60)

    # 添加Agents目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    agents_dir = os.path.join(current_dir, "Agents")
    if os.path.exists(agents_dir) and agents_dir not in sys.path:
        sys.path.insert(0, agents_dir)
        print(f"✓ 已添加Agents目录到Python路径：{agents_dir}")

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
                try:
                    # 尝试从Agents目录导入豆包模型测试器
                    from doubao_main import DouBaoModelTester
                    tester = DouBaoModelTester()
                    run_model_test(tester)
                except ImportError as e:
                    print(f"导入豆包模型测试器失败：{e}")
                    print(f"请确保doubao_main.py文件在 {agents_dir} 目录中")
                    input("\n按Enter键返回主菜单...")
            elif choice == 2:
                print("\n" + "=" * 60)
                print("测试千问模型 (支持结构化输出和原生web_search)")
                print("=" * 60)
                try:
                    # 尝试从Agents目录导入千问模型测试器
                    from qwen_main import QwenModelTester
                    tester = QwenModelTester()
                    run_model_test(tester)
                except ImportError as e:
                    print(f"导入千问模型测试器失败：{e}")
                    print(f"请确保qwen_main.py文件在 {agents_dir} 目录中")
                    input("\n按Enter键返回主菜单...")
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
            import traceback
            traceback.print_exc()
            input("\n按Enter键继续...")


if __name__ == "__main__":
    main()
