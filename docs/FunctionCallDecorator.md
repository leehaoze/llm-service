# FunctionCallDecorator 使用指南

## 概述

`FunctionCallDecorator` 是一个装饰器，用于将 LLM 的 function call 能力转换为基于 prompt 的模式。

### 核心特性

- **纯粹的转换器**：无论底层 LLM 是否原生支持 function call，都会将其转换为 prompt 模式
- **统一接口**：装饰后的 LLM 仍然遵循相同的 `LLM` Protocol
- **配置驱动**：是否使用装饰器完全由外部配置决定

## 使用场景

### 场景 1：为不支持 FC 的模型添加能力

某些模型（如某些 Llama 模型）不支持原生 function call，但可以通过 prompt engineering 来模拟这个能力。

```python
from llm_service import FunctionCallDecorator
from your_llm_provider import LlamaLLM

# 创建不支持 FC 的 LLM
base_llm = LlamaLLM(model="llama-3-70b")

# 用装饰器包装，让它"支持" FC
llm = FunctionCallDecorator(inner=base_llm)

# 现在可以像使用支持 FC 的模型一样使用它
response = llm.complete(messages, tools=tools)
```

### 场景 2：对比原生 FC 和 Prompt FC

即使模型原生支持 function call（如 OpenAI），也可以强制使用 prompt 模式来对比效果。

```python
from llm_service import OpenAILLM, FunctionCallDecorator

# 原生 FC 模式
native_llm = OpenAILLM(model="gpt-4")
response1 = native_llm.complete(messages, tools=tools)

# Prompt FC 模式
prompt_llm = FunctionCallDecorator(inner=OpenAILLM(model="gpt-4"))
response2 = prompt_llm.complete(messages, tools=tools)

# 对比两种模式的效果
```

## 工作原理

### 1. Prompt 注入

当调用 `complete()` 或 `stream()` 时，装饰器会：

1. 将 tools 定义转换为可读的描述
2. 构造一个 system prompt，引导模型输出 JSON 格式的工具调用
3. 将这个 system prompt 插入到消息列表开头

示例 prompt：

```
You are an AI assistant that can call functions to help users.

Available functions:
- get_weather: 获取指定城市的天气信息
  Parameters: {"type": "object", "properties": {"city": {"type": "string"}}}

When you need to call a function, respond with a JSON object in the following format:
{
  "name": "function_name",
  "arguments": {
    "param1": "value1"
  }
}

IMPORTANT:
- Only respond with the JSON object when you need to call a function
- Do NOT wrap the JSON in code blocks or markdown
- If you don't need to call a function, respond normally with natural language
```

### 2. 调用底层 LLM

装饰器调用底层 LLM 的 `complete()` 或 `stream()` 方法，**不传递 tools 参数**（因为已经通过 prompt 注入了）。

### 3. 解析响应

装饰器会：

1. 从 LLM 的文本响应中提取 JSON
2. 支持多种格式：
   - 纯 JSON：`{"name": "get_weather", "arguments": {...}}`
   - 数组：`[{"name": "func1", ...}, {"name": "func2", ...}]`
   - Markdown 代码块：`` ```json\n{...}\n``` ``
   - 混合文本：`这是一段文字 {"name": "get_weather", ...} 其他文字`
3. 将提取的 JSON 转换为标准的 `ToolCall` 结构
4. 返回标准的 `ModelResponse`

## API 参考

### FunctionCallDecorator

```python
@dataclass(slots=True)
class FunctionCallDecorator:
    """将 function call 转换为 prompt 模式的装饰器。"""

    inner: LLM  # 被装饰的 LLM 实例

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
    ) -> ModelResponse:
        """执行一次标准的大模型推理。"""
        ...

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
    ) -> Iterable[StreamChunk]:
        """执行一次流式推理，逐个返回 chunk。"""
        ...
```

### 行为说明

#### 当 `tools` 为 `None` 或空列表

直接透传给底层 LLM，不做任何处理。

#### 当 `tools` 有值

1. 注入 system prompt
2. 调用底层 LLM（不传 tools）
3. 解析响应，提取工具调用
4. 返回标准格式

#### 流式模式

当前实现是**先收集完整响应，再解析**，最后一次性产出 tool_call chunks。

未来可以优化为**真正的流式 JSON 解析**，边读边产出 chunk。

## 完整示例

```python
from llm_service import OpenAILLM, FunctionCallDecorator, types

# 1. 创建原始 LLM
base_llm = OpenAILLM(
    model="gpt-4o-mini",
    api_key="your-api-key"
)

# 2. 用装饰器包装
llm = FunctionCallDecorator(inner=base_llm)

# 3. 定义工具
tools: list[types.Tool] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 4. 调用（Complete 模式）
messages = [
    types.Message(role="user", content="北京今天天气怎么样？")
]

response = llm.complete(messages, tools=tools)

# 5. 处理响应
if response.message.tool_calls:
    for tool_call in response.message.tool_calls:
        print(f"调用工具: {tool_call['function']['name']}")
        print(f"参数: {tool_call['function']['arguments']}")
else:
    print(f"普通回复: {response.message.content}")

# 6. 流式模式
for chunk in llm.stream(messages, tools=tools):
    if chunk.type == "tool_call" and chunk.tool_call:
        print(f"工具调用: {chunk.tool_call}")
    elif chunk.type == "content" and chunk.content:
        print(f"内容: {chunk.content}")

    if chunk.finish_reason:
        print(f"结束原因: {chunk.finish_reason}")
```

## 构建时组装

在实际应用中，应该根据配置在构建 LLM 时决定是否使用装饰器：

```python
def build_llm(config: dict) -> LLM:
    """根据配置构建 LLM 实例"""

    # 1. 创建原始 LLM
    if config["provider"] == "openai":
        base_llm = OpenAILLM(model=config["model"])
    elif config["provider"] == "llama":
        base_llm = LlamaLLM(model=config["model"])
    else:
        raise ValueError(f"Unknown provider: {config['provider']}")

    # 2. 根据配置决定是否装饰
    if config.get("use_prompt_fc", False):
        return FunctionCallDecorator(inner=base_llm)

    return base_llm

# 使用示例
config = {
    "provider": "openai",
    "model": "gpt-4",
    "use_prompt_fc": True  # 强制使用 prompt 模式
}

llm = build_llm(config)
```

## 测试

运行测试：

```bash
python tests/test_decorator_simple.py
```

测试覆盖：
- ✓ 单个工具调用
- ✓ 多个工具调用
- ✓ 普通文本回复（不调用工具）
- ✓ 不传 tools 时的透传行为
- ✓ 从 markdown 代码块中提取 JSON

## 未来优化方向

### 1. 真正的流式 JSON 解析

当前流式模式会先收集完整响应再解析。可以优化为：

- 边读取 chunk 边解析 JSON
- 实时产出 tool_call chunk
- 支持增量式 JSON parsing

### 2. 更智能的 Prompt

- 支持多语言（根据用户消息语言调整 prompt）
- Few-shot examples（提供示例来提高准确率）
- 根据工具数量动态调整 prompt 详细程度

### 3. 错误恢复

- JSON 格式错误时的自动重试
- 部分字段缺失时的智能补全
- 多轮对话中的上下文保持

### 4. 性能监控

- 记录 JSON 解析成功率
- 对比原生 FC 和 Prompt FC 的效果差异
- 延迟和 token 使用量统计
