# 多模型智能代理股票查询系统

## 项目概述
这是一个基于LangChain的多模型智能代理框架，专门设计用于股票历史数据查询和最新新闻搜索。系统支持豆包(Doubao)和通义千问(Qwen)两种大语言模型，通过结构化工具调用实现专业的金融信息查询功能。

## 核心功能
### 1. 股票历史数据查询
- 支持查询指定股票代码在特定时间范围内的交易数据
- 可自定义查询参数：时间范围、分时级别(K线)、除权方式、返回字段
- 集成第三方股票数据API（智图API）

### 2. 智能联网新闻搜索
- 豆包模型：使用原生web_search工具获取最新信息
- 千问模型：使用联网搜索功能获取实时数据
- 自动识别用户查询中的搜索需求，获取公司最新新闻

### 3. 多模型支持
- 豆包模型：基于Volcengine ARK Runtime
- 千问模型：基于DashScope API
- 统一的工具调用接口和交互流程

### 4. 结构化输出与控制
- 基于LangChain的工具调用框架
- 自动迭代处理：模型思考 → 工具调用 → 结果整合 → 最终输出
- 完整的执行历史记录和错误处理

## 项目结构
    try/
    ├── multi_model_agent.py          # 主程序入口，多模型代理通用框架
    ├── settings/
    │   └── .env                      # 环境变量配置文件
    ├── configs/                      # 各模型配置文件
    │   ├── doubao_config.json
    │   └── qwen_config.json
    ├── test_results/                 # 测试结果保存目录
    ├── LLM_Tools/                    # 工具模块
    │   ├── doubao_websearch.py       # 豆包WebSearch工具封装
    │   ├── qwen_websearch.py         # 千问联网搜索工具
    │   ├── query_stock_history.py    # 股票历史数据查询工具
    │   └── query_stock_history.json  # 股票查询工具描述文件
    └── Agents/                       # 各模型特定实现
        ├── doubao_main.py            # 豆包模型代理实现
        └── qwen_main.py              # 千问模型代理实现


## 核心组件详解
### 1. 多模型代理框架 (multi_model_agent.py)
- BaseModelTester：抽象基类，定义所有模型的通用接口和行为
- StockQueryParams：股票查询参数的结构化Pydantic模型
- 主要功能：
  - 统一的工具调用循环处理逻辑
  - 消息历史管理
  - 错误处理和结果记录
  - 交互式用户界面

### 2. 工具模块 (LLM_Tools/)
#### 豆包WebSearch工具 (doubao_websearch.py)
主要特性
- 基于豆包原生web_search功能
- 支持多种搜索参数配置
- 工具化的LangChain接口
- 结果解析和格式化输出
- 完整的错误处理机制

#### 千问联网搜索工具 (qwen_websearch.py)
主要特性
- 基于千问原生web_search功能
- 支持多种搜索参数配置
- 工具化的LangChain接口
- 结果解析和格式化输出
- 完整的错误处理机制

#### 股票历史数据查询工具 (query_stock_history.py)
主要特性
- 支持多种时间范围和K线级别
- 多种除权方式选择
- 字段筛选功能
- 错误处理和超时控制

### 3. 模型代理实现 (Agents/)
#### 豆包模型代理 (doubao_main.py)
继承自BaseModelTester
- 特定功能：
  - 豆包客户端初始化
  - WebSearch工具独立调用
  - 搜索请求解析和格式化
  - 豆包特定的工具调用处理

#### 千问模型代理 (qwen_main.py)
继承自BaseModelTester
- 特定功能：
  - 千问客户端初始化
  - 联网搜索功能集成
  - 搜索请求解析和格式化
  - 千问特定的工具调用处理

## 环境配置
### 1. 环境变量设置
在 try/settings/.env 文件中配置：

    # 豆包模型配置
    ARK_API_KEY=your_ark_api_key_here
    ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
    ARK_MODEL=doubao-seed-1-6-250615
    
    # 千问模型配置
    QWEN_API_KEY=your_qwen_api_key_here
    QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    QWEN_MODEL=qwen-plus
    
    # 股票数据API配置
    STOCK_API_KEY=your_stock_api_key_here


### 2. 模型配置文件
每个模型在 try/configs/ 目录下有对应的JSON配置文件：
豆包配置 (doubao_config.json)：

    {
        "system_prompt": "专业的金融助手提示词...",
        "parameters": {
            "temperature": {"use": [0.1]},
            "top_p": {"use": [0.7]}
        }
    }

千问配置 (qwen_config.json)：

    {
        "system_prompt": "专业的金融助手提示词...",
        "parameters": {
            "temperature": {"use": [0.7]},
            "top_p": {"use": [0.8]},
            "presence_penalty": {"use": [0]},
            "repetition_penalty": {"use": [1.05]},
            "top_k": {"use": [1]}
        }
    }


## 安装与运行
### 1. 依赖安装
    # 进入项目目录
    cd try
    # 安装核心依赖
    pip install langchain langchain-openai python-dotenv requests pydantic
    # 豆包SDK
    pip install volcenginesdkarkruntime
    # 千问SDK（通过OpenAI兼容接口）
    pip install openai


### 2. 运行主程序
    # 直接运行主程序
    python multi_model_agent.py

### 3. 命令行界面
程序启动后显示交互式菜单：

    ============================================================
    股票查询工具结构化测试系统
    ============================================================
    请选择要测试的模型：
       1. 豆包模型 (无websearch功能，使用独立WebSearch工具)
       2. 千问模型 (支持结构化输出和原生web_search)
       3. 扩展预留位置 (待实现)
       4. 扩展预留位置 (待实现)
       5. 扩展预留位置 (待实现)
       0. 退出
    ============================================================

### 使用示例
默认查询：

    帮我查一下思特奇2026.2.12到2.14日的日收盘价和最高价，还要思特奇2026年2月的至少2条最新新闻及新闻发布时间。

## 执行流程
1. 模型初始化：加载选择的模型和工具
2. 查询解析：分析用户查询，确定需要的信息
3. 工具调用（按需）：
   - 调用股票查询工具获取历史价格数据
   - 调用WebSearch工具获取最新新闻

4. 结果整合：模型基于工具返回的数据生成最终回答
5. 输出展示：格式化显示查询结果

### 输出结果
系统会生成包含以下内容的回答：
- 股票代码识别和市场后缀
- 指定时间范围内的历史交易数据
- 公司最新新闻简报（标题、摘要、来源、时间）
- 数据分析和总结

## 扩展与自定义
1. 添加新模型

        1. 在 Agents/ 目录下创建新的模型代理文件（如 new_model_main.py）
        2. 继承 BaseModelTester 类
        3. 实现必要的抽象方法：
          • initialize_llm()：初始化语言模型
          • create_websearch_tool()：创建WebSearch工具
          • _handle_websearch_tool_call()：处理工具调用
        4. 在 configs/ 目录下添加对应的配置文件
        5. 在主菜单中添加新的选项

2. 添加新工具

        1. 在 LLM_Tools/ 目录下创建新的工具文件
        2. 实现工具功能，确保有清晰的函数接口
        3. 如果需要，创建对应的JSON描述文件
        4. 在模型代理中集成新工具

3. 修改系统提示词

        编辑对应模型的配置文件中的 system_prompt 字段，定制模型行为。

## 错误处理与日志
### 1. 错误类型
- 环境变量错误：API密钥未设置或错误
- 网络错误：API调用失败或超时
- 工具调用错误：股票查询或搜索失败
- 模型错误：模型返回异常或格式错误

### 2. 日志记录
系统记录完整的执行历史，包括：
- 模型初始化信息
- 每次工具调用的参数和结果
- 搜索请求和响应
- 最终输出和错误信息

### 3. 结果保存
测试结果可保存为JSON文件，包含：
- 完整的执行历史
- 最终输出内容
- 性能统计信息
- 错误记录（如果有）

## 性能优化
### 1. 缓存机制
- 可考虑添加股票数据缓存，减少重复API调用

- 新闻搜索结果缓存，提高响应速度

### 2. 并发处理

- 支持并行执行多个工具调用

- 异步处理长时间运行的操作

### 3. 资源管理

- 连接池管理API客户端

- 内存使用监控和优化

## 注意事项

### 1. API限制
- 股票数据API有调用频率限制（每日200次）
- 大语言模型API有token限制和费用
- 搜索工具可能受网络状况影响

### 2. 数据准确性
- 股票数据来自第三方API，需验证数据源可靠性
- 新闻搜索结果依赖搜索引擎的实时性和准确性

### 3. 安全考虑
- API密钥存储在环境变量中，不要提交到版本控制

## 常见问题
### 1. 导入错误
   ImportError: No module named 'XXX'
   
   解决：运行 pip install XXX

### 2. API密钥错误

   ValueError: XXX_API_KEY未设置
   
   解决：检查 .env 文件配置，确保环境变量正确设置

### 3. 工具调用失败

   股票查询工具执行出错：网络请求失败
   
   解决：检查网络连接，验证股票API密钥，确认股票代码格式

### 4. 搜索结果为空

   搜索不到相关信息
   
   解决：调整搜索关键词，检查搜索工具配置，验证模型联网功能

## 未来扩展计划

### 短期计划
1. 添加网格调参功能
2. 添加更多大语言模型支持（如元宝、deepseek等）
3. 添加数据可视化输出

### 长期计划
1. 支持实时股票数据推送
2. 添加投资建议和风险评估
3. 集成更多金融数据源
4. 开发Web界面和移动端应用

# 重要提示：本系统仅供学习和研究使用。股票投资有风险，系统提供的信息不应作为投资决策的唯一依据。请结合专业投资顾问的意见做出决策。
