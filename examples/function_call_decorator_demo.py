"""演示 FunctionCallDecorator 的使用"""

from llm_service.providers import OpenAIWrapper
from llm_service import FunctionCallDecorator, types


def demo_prompt_fc_mode():
    """演示使用装饰器将原生 FC 转换为 Prompt 模式"""

    # 1. 创建原始 LLM（OpenAI 原生支持 FC）
    base_llm = OpenAIWrapper()

    # 2. 用装饰器包装，强制走 Prompt 模式
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

    # 4. 调用 LLM
    messages = [
        types.Message(
            role="user",
            content="北京今天天气怎么样？"
        )
    ]

    print("=== Complete 模式 ===")
    response = llm.complete(messages, tools=tools)
    print(f"Response: {response.message.content}")
    if response.message.tool_calls:
        print(f"Tool calls: {response.message.tool_calls}")

    print("\n=== Stream 模式 ===")
    for chunk in llm.stream(messages, tools=tools):
        if chunk.type == "tool_call" and chunk.tool_call:
            print(f"Tool call chunk: {chunk.tool_call}")
        elif chunk.type == "content" and chunk.content:
            print(f"Content chunk: {chunk.content}")
        if chunk.finish_reason:
            print(f"Finish reason: {chunk.finish_reason}")


def demo_without_decorator():
    """演示不使用装饰器，直接使用原生 FC"""

    llm = OpenAIWrapper()

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
                            "description": "城市名称"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    messages = [
        types.Message(
            role="user",
            content="北京今天天气怎么样？"
        )
    ]

    print("=== 原生 FC 模式 ===")
    response = llm.complete(messages, tools=tools)
    print(f"Response: {response.message.content}")
    if response.message.tool_calls:
        print(f"Tool calls: {response.message.tool_calls}")


if __name__ == "__main__":
    # 对比两种模式
    print("【方式一：使用装饰器 - Prompt FC 模式】")
    demo_prompt_fc_mode()

    print("\n" + "="*60 + "\n")

    print("【方式二：不使用装饰器 - 原生 FC 模式】")
    demo_without_decorator()
