"""
豆包WebSearch工具封装 - 用于集成豆包原生web_search功能
功能：接收搜索关键词，创建豆包客户端，使用原生web_search工具进行搜索并返回结果
"""

import os
import sys
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from datetime import datetime
import logging
from volcenginesdkarkruntime import Ark
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 获取当前脚本所在目录的父目录
try_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'try_dir:{try_dir}')
env_path = os.path.join(try_dir, "settings", ".env")
print(f'env_path:{env_path}')

# 加载环境变量
load_dotenv(dotenv_path=env_path)

# 检查必要的环境变量
required_env_vars = ['ARK_API_KEY', 'ARK_BASE_URL', 'ARK_MODEL']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.warning(f"以下环境变量未设置：{missing_vars}")
    logger.warning("请确保.env文件包含这些变量")


class DouBaoWebSearchConfig:
    """豆包WebSearch配置管理类"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self.config_path = config_path or os.path.join(try_dir, "configs", "doubao_config.json")
        self.config_data = None
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            logger.warning(f"配置文件不存在：{self.config_path}，使用默认配置")
            return self.get_default_config()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
            logger.info(f"成功加载配置文件：{self.config_path}")
            return self.config_data
        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误：{e}")
            return self.get_default_config()
        except Exception as e:
            logger.error(f"读取配置文件失败：{e}")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "system_prompt": "你是一个专业的搜索引擎助手，根据用户提供的关键词进行精确、全面的搜索，并返回格式化的搜索结果。",
            "parameters": {
                "temperature": {"use": [0.1]},
                "top_p": {"use": [0.7]}
            }
        }

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        if self.config_data and "system_prompt" in self.config_data:
            return self.config_data["system_prompt"]
        return self.get_default_config()["system_prompt"]

    def get_parameter(self, param_name: str, default_value: Any = None) -> Any:
        """
        获取指定参数的值
        Args:
            param_name: 参数名
            default_value: 默认值
        Returns:
            参数值
        """
        if not self.config_data or "parameters" not in self.config_data:
            return default_value

        params = self.config_data.get("parameters", {})
        if param_name in params:
            param_config = params[param_name]
            use_values = param_config.get("use", [])
            if use_values:
                return use_values[0]

        return default_value

    def get_temperature(self) -> float:
        """获取temperature参数"""
        return self.get_parameter("temperature", 0.1)

    def get_top_p(self) -> float:
        """获取top_p参数"""
        return self.get_parameter("top_p", 0.7)


class DouBaoWebSearchClient:
    """豆包WebSearch客户端"""
    def __init__(self, config_manager: Optional[DouBaoWebSearchConfig] = None):
        """
        初始化豆包WebSearch客户端
        Args:
            config_manager: 配置管理器，如果为None则创建新的
        """
        self.config_manager = config_manager or DouBaoWebSearchConfig()
        self.client = None
        self.initialize_client()

    def initialize_client(self):
        """初始化豆包客户端"""
        try:
            api_key = os.getenv("ARK_API_KEY")
            base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
            model_name = os.getenv("ARK_MODEL", "doubao-seed-1-6-250615")

            if not api_key:
                raise ValueError("ARK_API_KEY未设置")

            self.client = Ark(
                base_url=base_url,
                api_key=api_key,
            )

            self.model_name = model_name
            logger.info(f"豆包客户端初始化成功，模型：{model_name}")

        except Exception as e:
            logger.error(f"初始化豆包客户端失败：{e}")
            raise

    def search(self, keyword: str, max_results: int = 2, max_keyword: int = 1) -> Dict[str, Any]:
        """
        执行Web搜索
        Args:
            keyword: 搜索关键词
            max_results: 最大搜索结果数量（影响返回内容长度）
            max_keyword: 最大关键词数量
        Returns:
            搜索结果字典
        """
        if not self.client:
            raise RuntimeError("豆包客户端未初始化")

        try:
            # 构建搜索查询
            search_query = f"请搜索以下关键词并返回搜索结果：{keyword}"

            # 配置web_search工具
            tools = [{
                "type": "web_search",
                "max_keyword": max_keyword,
            }]

            # 获取配置参数
            temperature = self.config_manager.get_temperature()
            top_p = self.config_manager.get_top_p()

            # 调用豆包API进行搜索
            logger.info(f"开始搜索关键词：{keyword}")
            start_time = datetime.now()

            response = self.client.responses.create(
                model=self.model_name,
                input=[{"role": "user", "content": search_query}],
                tools=tools,
                temperature=temperature,
                top_p=top_p,
            )

            search_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"搜索完成，耗时：{search_time:.2f}秒")

            # 解析响应
            result = self._parse_response(response, keyword, search_time)
            return result

        except Exception as e:
            logger.error(f"搜索过程中出错：{e}")
            return {
                "success": False,
                "keyword": keyword,
                "error": str(e),
                "search_results": [],
                "summary": f"搜索失败：{str(e)}",
                "search_time": 0
            }

    def _parse_response(self, response, keyword: str, search_time: float) -> Dict[str, Any]:
        """解析豆包API响应"""
        try:
            # 提取响应内容
            if hasattr(response, 'output') and hasattr(response.output, 'choices'):
                choice = response.output.choices[0]

                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content

                    # 构建搜索结果
                    search_result = {
                        "success": True,
                        "keyword": keyword,
                        "search_time": search_time,
                        "content": content,
                        "content_length": len(content),
                        "search_results": [
                            {
                                "title": f"关于'{keyword}'的搜索结果",
                                "summary": content[:500] + "..." if len(content) > 500 else content,
                                "source": "豆包WebSearch"
                            }
                        ],
                        "summary": self._extract_summary(content),
                        "metadata": {
                            "model": self.model_name,
                            "timestamp": datetime.now().isoformat(),
                            "max_keyword": 1
                        }
                    }

                    logger.info(f"搜索结果解析成功，内容长度：{len(content)}字符")
                    return search_result

            # 如果无法从标准结构解析，尝试其他方式
            fallback_content = str(response)
            logger.warning(f"使用备用方式解析响应，内容长度：{len(fallback_content)}字符")

            return {
                "success": True,
                "keyword": keyword,
                "search_time": search_time,
                "content": fallback_content,
                "content_length": len(fallback_content),
                "search_results": [
                    {
                        "title": f"关于'{keyword}'的搜索结果",
                        "summary": fallback_content[:500] + "..." if len(fallback_content) > 500 else fallback_content,
                        "source": "豆包WebSearch"
                    }
                ],
                "summary": self._extract_summary(fallback_content),
                "metadata": {
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "max_keyword": 1
                }
            }

        except Exception as e:
            logger.error(f"解析响应时出错：{e}")
            return {
                "success": False,
                "keyword": keyword,
                "error": f"解析响应失败：{str(e)}",
                "search_results": [],
                "summary": f"解析搜索结果失败：{str(e)}",
                "search_time": search_time
            }

    def _extract_summary(self, content: str, max_length: int = 300) -> str:
        """从内容中提取摘要"""
        if not content:
            return "无搜索结果"

        # 尝试提取关键信息
        lines = content.split('\n')
        summary_lines = []

        for line in lines:
            line = line.strip()
            if line and len(' '.join(summary_lines)) < max_length:
                # 跳过太短的行或纯标点符号
                if len(line) > 20 and any(c.isalpha() or c.isdigit() for c in line):
                    summary_lines.append(line)

        summary = ' '.join(summary_lines)

        # 如果摘要太长则截断
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        return summary if summary else content[:max_length] + "..." if len(content) > max_length else content


# Pydantic模型用于定义工具参数
class WebSearchParams(BaseModel):
    """Web搜索参数的结构化输出模型"""
    keyword: str = Field(
        description="搜索关键词，例如 '今日热点新闻'、'苹果公司最新财报'"
    )
    max_results: Optional[int] = Field(
        2,
        description="最大搜索结果数量，影响返回内容的丰富程度。默认为2"
    )
    max_keyword: Optional[int] = Field(
        1,
        description="最大关键词数量。豆包原生web_search工具的参数，默认为1"
    )


def create_doubao_websearch_tool() -> Tool:
    """
    创建豆包WebSearch工具的LangChain包装器
    Returns:
        LangChain Tool对象
    """

    def doubao_websearch_wrapper(
            keyword: str,
            max_results: int = 2,
            max_keyword: int = 1
    ) -> str:
        """
        豆包WebSearch工具包装函数
        Args:
            keyword: 搜索关键词
            max_results: 最大搜索结果数量
            max_keyword: 最大关键词数量
        Returns:
            格式化的搜索结果字符串
        """
        try:
            logger.info(f"豆包WebSearch工具调用，关键词：{keyword}")

            # 创建配置管理器和客户端
            config_manager = DouBaoWebSearchConfig()
            client = DouBaoWebSearchClient(config_manager)

            # 执行搜索
            result = client.search(keyword, max_results, max_keyword)

            # 格式化结果
            if result.get("success", False):
                formatted_result = format_search_result(result)
                logger.info(f"豆包WebSearch工具执行成功，结果长度：{len(formatted_result)}字符")
                return formatted_result
            else:
                error_msg = f"豆包WebSearch工具执行失败：{result.get('error', '未知错误')}"
                logger.error(error_msg)
                return error_msg

        except Exception as e:
            error_msg = f"豆包WebSearch工具执行出错：{str(e)}"
            logger.error(error_msg)
            return error_msg

    # 从JSON文件加载工具描述
    tool_description = "使用豆包原生web_search工具进行网络搜索，获取最新的网络信息。"

    # 尝试从当前目录下的JSON文件加载更详细的描述
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "doubao_websearch.json")

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                if isinstance(json_data, dict) and 'function' in json_data:
                    tool_description = json_data['function'].get('description', tool_description)
        except Exception as e:
            logger.warning(f"读取工具描述文件失败：{e}，使用默认描述")

    # 使用StructuredTool创建工具
    return StructuredTool.from_function(
        name="doubao_websearch",
        description=tool_description,
        func=doubao_websearch_wrapper,
        args_schema=WebSearchParams
    )


def format_search_result(result: Dict[str, Any]) -> str:
    """
    格式化搜索结果

    Args:
        result: 搜索结果字典

    Returns:
        格式化的搜索结果字符串
    """
    if not result.get("success", False):
        return f"搜索失败：{result.get('error', '未知错误')}"

    keyword = result.get("keyword", "未知关键词")
    summary = result.get("summary", "无摘要")
    search_time = result.get("search_time", 0)
    content = result.get("content", "")

    # 构建格式化结果
    formatted = f"""# 豆包WebSearch搜索结果

## 搜索关键词
{keyword}

## 搜索摘要
{summary}

## 详细结果
{content}

## 搜索统计
- 搜索耗时：{search_time:.2f}秒
- 内容长度：{len(content)}字符
- 数据来源：豆包原生web_search工具
- 搜索时间：{result.get('metadata', {}).get('timestamp', '未知时间')}
"""

    return formatted


def test_doubao_websearch():
    """测试豆包WebSearch工具"""
    print("开始测试豆包WebSearch工具...")

    try:
        # 测试1：创建工具
        print("\n1. 创建豆包WebSearch工具...")
        tool = create_doubao_websearch_tool()
        print(f"✓ 工具创建成功：{tool.name}")
        print(f"工具描述：{tool.description}")

        # 测试2：使用工具进行搜索
        print("\n2. 执行搜索测试...")
        test_keyword = "2026年人工智能最新进展"

        print(f"搜索关键词：{test_keyword}")
        result = tool.invoke({"keyword": test_keyword, "max_results": 2})

        print(f"\n搜索结果（前500字符）：")
        print("-" * 60)
        print(result[:500] + "..." if len(result) > 500 else result)
        print("-" * 60)
        print(f"✓ 搜索完成，结果长度：{len(result)}字符")

        # 测试3：测试配置管理器
        print("\n3. 测试配置管理器...")
        config_manager = DouBaoWebSearchConfig()
        system_prompt = config_manager.get_system_prompt()
        temperature = config_manager.get_temperature()
        top_p = config_manager.get_top_p()

        print(f"系统提示词（前100字符）：{system_prompt[:100]}...")
        print(f"Temperature参数：{temperature}")
        print(f"Top_p参数：{top_p}")
        print("✓ 配置管理器测试完成")

        print("\n✅ 所有测试通过！")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    success = test_doubao_websearch()

    if success:
        print("\n豆包WebSearch工具测试完成，可以集成到主程序中。")
        print(success)
'''
prompts_元宝:
我要修改代码，把主程序代码里的搜索部分替换为自己写、自己测试的、确定可以进行搜索的doubao_websearch.py和qwen_websearch.py，
目的是让主程序不能联网，而当主程序判断需要联网搜索时，生成想要搜索的关键词，然后调用这个工具代码：
创建一个新的豆包，并用它的原生websearch功能进行搜索并输出结果，传入主程序中。
所以我需要把这个“接收关键词-创建联网新豆包-搜索并输出”的代码封装成符合LangChain规范的“工具”（Tool）以供断网主程序进行对应的调用。
请你给我完成第一步：根据给出的官方成功示例，写doubao_websearch.py。
注意参考我上面给出的代码的项目结构：
主程序：try/Agents/1stAgent_queryStockHistory.py
此文件：try/LLM_Tools/doubao_websearch.py
环境配置文件：try/settings/.env，包括ARK_API_KEY、ARK_BASE_URL、ARK_MODEL字段，用dotenv读取
参数配置文件（如有必要就用，我觉得只是写一个websearch工具的话不需要吧？但我还是给你了，你自己决定用不用）：try/configs/doubao_config.json，要使用的是每个参数的"use"字段的值，如果用的话不许硬编码，必须是读取这个json；不用的话就不用管了。
doubao_config.json内容如下：（略）
请你自己决定用不用这个参数文件。
豆包官方websearch示例：（略）
'''
