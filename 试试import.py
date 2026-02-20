# 已知ok的语句：
# from langchain_openai import ChatOpenAI
# from langchain.agents import create_agent
# from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
# 要探索这个：
import langchain.agents
import importlib  # 用来探索的工具包


def explore_module(module, depth=0, max_depth=2):
    """递归探索模块"""
    indent = "  " * depth

    # 打印模块基本信息
    print(f"{indent}模块: {module.__name__}")

    # 只探索到最大深度
    if depth >= max_depth:
        return

    # 尝试获取模块的所有属性
    try:
        for attr_name in dir(module):
            # 跳过私有属性
            if attr_name.startswith('_'):
                continue

            try:
                attr = getattr(module, attr_name)

                # 判断属性类型
                if hasattr(attr, '__module__'):
                    module_name = attr.__module__
                    # 如果是来自当前模块或子模块
                    if module_name and module_name.startswith('langchain.agents'):
                        print(f"{indent}  {attr_name} (类型: {type(attr).__name__})")

                        # 如果是模块，递归探索
                        if isinstance(attr, type(importlib.import_module('sys'))):
                            explore_module(attr, depth + 1, max_depth)
            except Exception as e:
                print(f"{indent}  {attr_name}: 无法获取 - {e}")
    except Exception as e:
        print(f"{indent}无法探索: {e}")


# 开始探索
explore_module(langchain.agents)
