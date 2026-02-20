import os
import requests
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
load_dotenv()
def query_stock_history(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "d",
    adjust: str = "n",
    fields: Optional[List[str]] = None
) -> str:
    """
    根据股票代码、日期范围、分时级别等参数查询历史交易数据。

    Args:
        symbol (str): 股票代码和市场后缀，例如 ‘000001.SZ'、'AAPL.US'。
        start_date (Optional[str]): 开始日期，格式为 YYYYMMDD 或 YYYYMMDDhhmmss，例如 ‘20240601’。默认为None，表示查询全部历史。
        end_date (Optional[str]): 结束日期，格式同 start_date。默认为None，表示查询到最新数据。
        interval (str): 分时级别。支持 ‘5‘, ‘15‘, ‘30‘, ‘60‘, ’d‘, ’w‘, ’m‘, ’y‘，分别对应5分钟、15分钟、30分钟、60分钟、日线、周线、月线、年线。默认为 ‘d‘（日线）。
        adjust (str): 除权方式。支持 ‘n‘ (不复权)， ‘f‘ (前复权)， ‘b‘ (后复权)， ‘fr‘ (等比前复权)， ‘br‘ (等比后复权)。分钟线只支持 ‘n‘。默认为 ‘n‘。
        fields (Optional[List[str]]): 需要返回的字段列表。如果为None，则返回API所有字段。常用字段: [’t‘, ’o‘, ’h‘, ’l‘, ’c‘, ’v‘, ’a‘]。

    Returns:
        str: 查询到的历史交易数据JSON字符串，或错误信息。
    """
    # 1. 构造API请求URL
    base_url = "https://api.zhituapi.com/hs/history"
    # 从环境变量获取API Token
    api_token = os.getenv("STOCK_API_KEY")
    if not api_token:
        return "错误：未在环境变量中找到STOCK_API_KEY。"

    # 构建请求路径和参数
    url = f"{base_url}/{symbol}/{interval}/{adjust}"
    params = {"token": api_token}

    if start_date:
        params["st"] = start_date
    if end_date:
        params["et"] = end_date

    # 2. 发起API请求
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # 检查HTTP错误
        data = response.json()

        # 3. 处理API返回结果 (假设成功返回JSON列表)
        if isinstance(data, list):
            # 如果指定了fields，则进行字段过滤
            if fields:
                filtered_data = []
                for item in data:
                    filtered_item = {k: v for k, v in item.items() if k in fields}
                    filtered_data.append(filtered_item)
                result_data = filtered_data
            else:
                result_data = data
            # 将结果转换为JSON字符串返回，供LLM读取
            return str(result_data)
        else:
            # API可能返回错误信息
            return f"API返回了非列表数据: {data}"

    except requests.exceptions.RequestException as e:
        return f"网络请求失败: {str(e)}"
    except ValueError as e:  # JSON解析错误
        return f"解析API响应失败: {str(e)}"
    except Exception as e:
        return f"查询过程中发生未知错误: {str(e)}"


# 可选：测试函数
if __name__ == "__main__":
    test_result = query_stock_history(
        symbol="000001.SZ",
        start_date="20240601",
        end_date="20240605",
        interval="d",
        adjust="n",
        fields=['t', 'o', 'h', 'l', 'c', 'v']
    )
    print("测试输出（前200字符）:", test_result[:200])