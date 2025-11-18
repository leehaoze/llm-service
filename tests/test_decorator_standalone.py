"""独立测试 FunctionCallDecorator（不依赖 OpenAI）"""

import sys
import json
from pathlib import Path

# 添加 src 到 path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from typing import Sequence, Iterable

# 直接导入模块，避免从 __init__ 导入
from llm_service import types
from llm_service.decorators.function_call_decorator import FunctionCallDecorator

Message = types.Message
ModelResponse = types.ModelResponse
StreamChunk = types.StreamChunk
Tool = types.Tool
ToolCall = types.ToolCall


class MockLLM:
    """模拟一个不支持原生 FC 的 LLM"""

    def __init__(self, mock_response: str):
        self.mock_response = mock_response

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
    ) -> ModelResponse:
        # 模拟返回 JSON 格式的工具调用
        return ModelResponse(
            message=Message(
                role="assistant",
                content=self.mock_response,
            )
        )

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
    ) -> Iterable[StreamChunk]:
        # 模拟流式返回
        for char in self.mock_response:
            yield StreamChunk(
                type="content",
                content=char,
                tool_call=None,
                finish_reason=None,
            )


def test_single_tool_call():
    """测试单个工具调用"""
    print("测试：单个工具调用...")

    mock_response = json.dumps({
        "name": "get_weather",
        "arguments": {"city": "北京"}
    }, ensure_ascii=False)

    mock_llm = MockLLM(mock_response)
    decorator = FunctionCallDecorator(inner=mock_llm)

    tools: list[Tool] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    messages = [
        Message(role="user", content="北京天气怎么样？")
    ]

    response = decorator.complete(messages, tools=tools)

    assert response.message.role == "assistant"
    assert response.message.tool_calls is not None
    assert len(response.message.tool_calls) == 1
    assert response.message.tool_calls[0]["function"]["name"] == "get_weather"

    args = json.loads(response.message.tool_calls[0]["function"]["arguments"])
    assert args["city"] == "北京"

    print("✓ 测试通过")


def test_multiple_tool_calls():
    """测试多个工具调用"""
    print("测试：多个工具调用...")

    mock_response = json.dumps([
        {"name": "get_weather", "arguments": {"city": "北京"}},
        {"name": "get_weather", "arguments": {"city": "上海"}},
    ], ensure_ascii=False)

    mock_llm = MockLLM(mock_response)
    decorator = FunctionCallDecorator(inner=mock_llm)

    tools: list[Tool] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    messages = [
        Message(role="user", content="北京和上海的天气怎么样？")
    ]

    response = decorator.complete(messages, tools=tools)

    assert response.message.tool_calls is not None
    assert len(response.message.tool_calls) == 2
    assert response.message.tool_calls[0]["function"]["name"] == "get_weather"
    assert response.message.tool_calls[1]["function"]["name"] == "get_weather"

    print("✓ 测试通过")


def test_natural_language_response():
    """测试普通文本回复（不调用工具）"""
    print("测试：普通文本回复...")

    mock_response = "我是一个 AI 助手，很高兴为您服务！"

    mock_llm = MockLLM(mock_response)
    decorator = FunctionCallDecorator(inner=mock_llm)

    tools: list[Tool] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

    messages = [
        Message(role="user", content="你好")
    ]

    response = decorator.complete(messages, tools=tools)

    assert response.message.role == "assistant"
    assert response.message.content == mock_response
    assert response.message.tool_calls is None

    print("✓ 测试通过")


def test_no_tools():
    """测试不传 tools 时直接透传"""
    print("测试：不传 tools...")

    mock_response = "这是一个普通回复"

    mock_llm = MockLLM(mock_response)
    decorator = FunctionCallDecorator(inner=mock_llm)

    messages = [
        Message(role="user", content="你好")
    ]

    response = decorator.complete(messages, tools=None)

    assert response.message.content == mock_response
    assert response.message.tool_calls is None

    print("✓ 测试通过")


def test_stream_with_tool_calls():
    """测试流式模式下的工具调用"""
    print("测试：流式模式下的工具调用...")

    mock_response = json.dumps({
        "name": "get_weather",
        "arguments": {"city": "北京"}
    }, ensure_ascii=False)

    mock_llm = MockLLM(mock_response)
    decorator = FunctionCallDecorator(inner=mock_llm)

    tools: list[Tool] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    messages = [
        Message(role="user", content="北京天气怎么样？")
    ]

    chunks = list(decorator.stream(messages, tools=tools))

    # 应该至少有一个 tool_call chunk
    tool_call_chunks = [c for c in chunks if c.type == "tool_call" and c.tool_call]
    assert len(tool_call_chunks) > 0
    assert tool_call_chunks[0].tool_call["function"]["name"] == "get_weather"

    # 应该有 finish_reason
    finish_chunks = [c for c in chunks if c.finish_reason]
    assert len(finish_chunks) > 0
    assert finish_chunks[0].finish_reason == "tool_use"

    print("✓ 测试通过")


def test_json_in_markdown():
    """测试从 markdown 代码块中提取 JSON"""
    print("测试：从 markdown 代码块中提取 JSON...")

    mock_response = """好的，我来帮你查询天气：

```json
{
  "name": "get_weather",
  "arguments": {"city": "北京"}
}
```
"""

    mock_llm = MockLLM(mock_response)
    decorator = FunctionCallDecorator(inner=mock_llm)

    tools: list[Tool] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

    messages = [
        Message(role="user", content="北京天气怎么样？")
    ]

    response = decorator.complete(messages, tools=tools)

    assert response.message.tool_calls is not None
    assert len(response.message.tool_calls) == 1
    assert response.message.tool_calls[0]["function"]["name"] == "get_weather"

    print("✓ 测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("运行 FunctionCallDecorator 测试")
    print("=" * 60)

    test_single_tool_call()
    test_multiple_tool_calls()
    test_natural_language_response()
    test_no_tools()
    test_stream_with_tool_calls()
    test_json_in_markdown()

    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)
