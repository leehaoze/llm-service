# llm-service

统一的大语言模型服务接口，支持多种 LLM 提供商和 function calling 能力。

## 特性

- **统一接口**：为不同的 LLM 提供商提供统一的调用接口
- **Function Call 支持**：原生支持 function calling，包括 prompt 模式模拟
- **流式输出**：支持流式响应
- **类型安全**：完整的 Python 类型注解

## 安装

```bash
pip install -e .
```

## 快速开始

### 基本使用

```python
from llm_service import OpenAIWrapper, types

# 创建 LLM 实例
llm = OpenAIWrapper(
    model="gpt-4o-mini",
    api_key="your-api-key"
)

# 发送消息
messages = [
    types.Message(role="user", content="你好")
]

response = llm.complete(messages)
print(response.message.content)
```

### 使用 Function Call

```python
from llm_service import OpenAIWrapper, types

llm = OpenAIWrapper(model="gpt-4o-mini", api_key="your-api-key")

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    }
]

messages = [
    types.Message(role="user", content="北京今天天气怎么样？")
]

response = llm.complete(messages, tools=tools)

if response.message.tool_calls:
    for tool_call in response.message.tool_calls:
        print(f"调用工具: {tool_call['function']['name']}")
        print(f"参数: {tool_call['function']['arguments']}")
```

### 使用 FunctionCallDecorator（Prompt 模式）

对于不支持原生 function call 的模型，或者想要对比效果，可以使用装饰器将 FC 转换为 prompt 模式：

```python
from llm_service import OpenAIWrapper, FunctionCallDecorator

# 创建原始 LLM
base_llm = OpenAIWrapper(model="gpt-4o-mini", api_key="your-api-key")

# 用装饰器包装，强制走 prompt 模式
llm = FunctionCallDecorator(inner=base_llm)

# 使用方式与原生 FC 完全相同
response = llm.complete(messages, tools=tools)
```

详细使用说明请参考：[FunctionCallDecorator 文档](docs/FunctionCallDecorator.md)

## 项目结构

```
llm-service/
├── src/llm_service/
│   ├── types.py              # 类型定义
│   ├── llm.py                # LLM Protocol 定义
│   ├── providers/            # LLM 提供商实现
│   │   └── common.py         # OpenAI 兼容实现
│   └── decorators/           # 装饰器
│       └── function_call_decorator.py  # FC Prompt 模式装饰器
├── tests/                    # 测试
├── examples/                 # 示例代码
└── docs/                     # 文档

```

## 测试

```bash
# 运行测试
python tests/test_decorator_simple.py
```

## 文档

- [FunctionCallDecorator 使用指南](docs/FunctionCallDecorator.md)
- [实现总结](IMPLEMENTATION_SUMMARY.md)

## 开发

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```
