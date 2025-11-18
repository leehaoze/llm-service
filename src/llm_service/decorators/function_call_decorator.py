"""将 function call 转换为 prompt 模式的装饰器。

无论底层 LLM 是否原生支持 FC，都会：
1. 将 tools 注入到 prompt
2. 引导模型输出 JSON 格式的工具调用
3. 解析 JSON 并转换成标准的 ToolCall 结构
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, cast

from ..llm import LLM
from ..types import (
    Message,
    MessageContent,
    ModelResponse,
    StreamChunk,
    Tool,
    ToolCall,
)


@dataclass(slots=True)
class FunctionCallDecorator:
    """将 function call 转换为 prompt 模式的装饰器。"""

    inner: LLM

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
    ) -> ModelResponse:
        """执行一次标准的大模型推理。"""
        # 没有 tools，直接透传
        if not tools:
            return self.inner.complete(messages, tools=None)

        # 有 tools，走 prompt 模式
        return self._complete_via_prompt_fc(messages, tools)

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
    ) -> Iterable[StreamChunk]:
        """执行一次流式推理，逐个返回 chunk。"""
        # 没有 tools，直接透传
        if not tools:
            yield from self.inner.stream(messages, tools=None)
            return

        # 有 tools，走 prompt 模式
        yield from self._stream_via_prompt_fc(messages, tools)

    def _complete_via_prompt_fc(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
    ) -> ModelResponse:
        """通过 prompt 模式实现 function call"""
        # 1. 构造注入了工具说明的消息
        prompt_messages = _build_prompt_fc_messages(messages, tools)

        # 2. 调用底层 LLM（不传 tools 参数）
        base_resp = self.inner.complete(prompt_messages, tools=None)

        # 3. 解析 JSON → ToolCall 列表
        tool_calls, final_content = _parse_prompt_fc_output(
            base_resp.message.content
        )

        # 4. 返回标准的 ModelResponse
        return ModelResponse(
            message=Message(
                role="assistant",
                content=final_content or "",
                tool_calls=tool_calls if tool_calls else None,
            )
        )

    def _stream_via_prompt_fc(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
    ) -> Iterable[StreamChunk]:
        """通过 prompt 模式实现流式 function call"""
        prompt_messages = _build_prompt_fc_messages(messages, tools)
        raw_stream = self.inner.stream(prompt_messages, tools=None)

        # 收集所有 chunk
        buffer: list[str] = []
        for chunk in raw_stream:
            if chunk.content:
                buffer.append(chunk.content)

        # 解析完整文本
        text = "".join(buffer)
        tool_calls, final_content = _parse_prompt_fc_output(text)

        # 先发 tool_call chunk
        if tool_calls:
            for tc in tool_calls:
                yield StreamChunk(
                    type="tool_call",
                    content=None,
                    tool_call=tc,
                    finish_reason=None,
                )
            # 如果有 tool_calls，finish_reason 应该是 tool_use
            yield StreamChunk(
                type="tool_call",
                content=None,
                tool_call=None,
                finish_reason="tool_use",
            )

        # 再发 content（如果有）
        if final_content:
            yield StreamChunk(
                type="content",
                content=final_content,
                tool_call=None,
                finish_reason="stop",
            )


def _build_prompt_fc_messages(
    messages: Sequence[Message],
    tools: Sequence[Tool],
) -> list[Message]:
    """构造包含工具说明的消息列表。

    将工具定义注入到 system prompt 中，引导模型输出 JSON 格式的工具调用。
    """
    # 构造工具定义的 JSON 描述
    tools_desc = _format_tools_description(tools)

    # 构造 system prompt
    system_content = f"""You are an AI assistant that can call functions to help users.

Available functions:
{tools_desc}

When you need to call a function, respond with a JSON object in the following format:
{{
  "name": "function_name",
  "arguments": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}

If you need to call multiple functions, respond with a JSON array:
[
  {{"name": "func1", "arguments": {{}}}},
  {{"name": "func2", "arguments": {{}}}}
]

IMPORTANT:
- Only respond with the JSON object/array when you need to call a function
- Do NOT wrap the JSON in code blocks or markdown
- Do NOT add any text before or after the JSON
- If you don't need to call a function, respond normally with natural language
"""

    # 插入到消息列表开头
    system_message = Message(role="system", content=system_content)

    return [system_message, *messages]


def _format_tools_description(tools: Sequence[Tool]) -> str:
    """格式化工具定义为可读的描述"""
    lines: list[str] = []

    for tool in tools:
        func = tool["function"]
        lines.append(f"- {func['name']}: {func['description']}")
        lines.append(f"  Parameters: {json.dumps(func['parameters'], ensure_ascii=False, indent=2)}")

    return "\n".join(lines)


def _parse_prompt_fc_output(
    content: MessageContent,
) -> tuple[list[ToolCall], str | None]:
    """从模型输出中解析工具调用。

    Returns:
        (tool_calls, final_content):
            - tool_calls: 解析出的工具调用列表
            - final_content: 如果有非 JSON 的自然语言内容，返回它；否则返回 None
    """
    # 如果 content 不是字符串，转换为字符串
    if isinstance(content, str):
        text = content
    else:
        # 如果是 Sequence[ContentPart]，需要提取文本
        text_parts: list[str] = []
        for part in content:
            if part.get("type") == "text":
                # TypedDict 的 get 返回值需要类型检查
                text_value = part.get("text")
                if text_value is not None and isinstance(text_value, str):
                    text_parts.append(text_value)
        text = "".join(text_parts)

    # 尝试提取 JSON
    json_data = _extract_json(text)

    if json_data is None:
        # 没有找到 JSON，认为是普通文本回复
        return [], text

    # 解析为 ToolCall 列表
    tool_calls = _convert_to_tool_calls(json_data)

    # 如果整个内容就是 JSON，则 final_content 为 None
    # 否则，保留非 JSON 部分（简化处理：如果有 tool_calls，就不保留 content）
    final_content = None if tool_calls else text

    return tool_calls, final_content


def _extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    """从文本中提取 JSON 对象或数组。

    支持：
    - 纯 JSON
    - 包含在 markdown 代码块中的 JSON
    - 混合在文本中的 JSON
    """
    # 去除首尾空白
    text = text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 markdown 代码块中的 JSON
    code_block_pattern = r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```"
    match = re.search(code_block_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取第一个 JSON 对象或数组
    # 对象
    obj_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    match = re.search(obj_pattern, text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 数组
    arr_pattern = r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]"
    match = re.search(arr_pattern, text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 没有找到有效的 JSON
    return None


def _convert_to_tool_calls(data: dict[str, Any] | list[Any]) -> list[ToolCall]:
    """将 JSON 数据转换为 ToolCall 列表"""
    # 如果是单个对象，转换为列表
    items: list[Any]
    if isinstance(data, dict):
        items = [data]
    else:
        items = list(data)  # 确保是 list 类型

    tool_calls: list[ToolCall] = []

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        # 类型缩窄：现在 item 是 dict[str, Any]
        item_dict = cast(dict[str, Any], item)

        # 检查必需字段
        if "name" not in item_dict:
            continue

        # 提取字段，确保类型安全
        call_id = item_dict.get("id")
        func_name = item_dict.get("name")
        func_args = item_dict.get("arguments", {})

        # 构造 ToolCall
        tool_call = ToolCall(
            id=str(call_id) if call_id is not None else f"call_{i}",
            type="function",
            function={
                "name": str(func_name),
                "arguments": json.dumps(
                    func_args,
                    ensure_ascii=False
                ),
            },
        )

        tool_calls.append(tool_call)

    return tool_calls
