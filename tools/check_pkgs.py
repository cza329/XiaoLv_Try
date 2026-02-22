# test_install.py（更新为v0.1.0+版本）
import langchain

print(f"LangChain版本: {langchain.__version__}")
print("✅ LangChain安装成功！")

# 测试基本功能 - 使用新版导入方式
try:
    # 新版导入方式
    from langchain_core.prompts import PromptTemplate

    template = "你好，{name}！今天天气如何？"
    prompt = PromptTemplate.from_template(template)
    print(f"Prompt示例: {prompt.format(name='小明')}")
    print("✅ 新版导入成功！")

except ImportError:
    # 如果新版不可用，尝试旧版
    try:
        from langchain.prompts import PromptTemplate

        template = "你好，{name}！今天天气如何？"
        prompt = PromptTemplate.from_template(template)
        print(f"Prompt示例: {prompt.format(name='小明')}")
        print("✅ 旧版导入成功！")
    except ImportError as e:
        print(f"导入失败: {e}")
        print("请安装 langchain-core: pip install langchain-core")
