"""简单测试 FunctionCallDecorator（完全独立）"""

import sys
import json
from pathlib import Path

# 添加 src 到 path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# 手动加载模块，避免触发 openai 导入
import importlib.util

def load_module(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# 加载 types
types = load_module("llm_service.types", src_path / "llm_service" / "types.py")

# 加载 llm (Protocol定义)
llm_mod = load_module("llm_service.llm", src_path / "llm_service" / "llm.py")

# 加载 decorator
decorator_mod = load_module(
    "llm_service.decorators.function_call_decorator",
    src_path / "llm_service" / "decorators" / "function_call_decorator.py"
)

FunctionCallDecorator = decorator_mod.FunctionCallDecorator
Message = types.Message
ModelResponse = types.ModelResponse
StreamChunk = types.StreamChunk
Tool = types.Tool


class MockLLM:
    """模拟一个不支持原生 FC 的 LLM"""

    def __init__(self, mock_response: str):
        self.mock_response = mock_response

    def complete(self, messages, tools=None):
        return ModelResponse(
            message=Message(
                role="assistant",
                content=self.mock_response,
            )
        )

    def stream(self, messages, tools=None):
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

    tools = [
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

    messages = [Message(role="user", content="北京天气怎么样？")]

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

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }
    ]

    messages = [Message(role="user", content="北京和上海的天气怎么样？")]
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

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

    messages = [Message(role="user", content="你好")]
    response = decorator.complete(messages, tools=tools)

    assert response.message.role == "assistant"
    assert response.message.content == mock_response
    assert response.message.tool_calls is None or len(response.message.tool_calls) == 0

    print("✓ 测试通过")


def test_no_tools():
    """测试不传 tools 时直接透传"""
    print("测试：不传 tools...")

    mock_response = "这是一个普通回复"

    mock_llm = MockLLM(mock_response)
    decorator = FunctionCallDecorator(inner=mock_llm)

    messages = [Message(role="user", content="你好")]
    response = decorator.complete(messages, tools=None)

    assert response.message.content == mock_response
    assert response.message.tool_calls is None

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

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

    messages = [Message(role="user", content="北京天气怎么样？")]
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
    test_json_in_markdown()

    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)
