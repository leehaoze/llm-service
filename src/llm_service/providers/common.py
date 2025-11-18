"""适合所有兼容 openai 的大模型请求"""
from openai import OpenAI

from typing import Sequence, Iterable, Any, Literal, cast

from ..llm import LLM
from ..types import Role, Message, MessageContent, ModelResponse, StreamChunk, Tool

from openai.types.chat import(
    ChatCompletionMessageParam,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionToolParam
)

class OpenAIWrapper(LLM):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None
    ) -> None:
        self._model = model
        self._client = OpenAI(
            api_key=api_key,
        )
    
    
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None
    ) -> ModelResponse:
        """执行一次标准的大模型推理。"""
        if tools:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=_serialize_messages(messages),
                tools=_serialize_tools(tools)
            )
        else:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=_serialize_messages(messages)
            )

        return _deserialize_response(response)

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None
    ) -> Iterable[StreamChunk]:
        """执行一次流式推理，逐个返回 chunk。"""
        if tools:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=_serialize_messages(messages),
                tools=_serialize_tools(tools),
                stream=True
            )
        else:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=_serialize_messages(messages),
                stream=True
            )

        for chunk in stream:
            if parsed_chunk := _deserialize_stream_chunk(chunk):
                yield parsed_chunk
    
    

def _serialize_tools(tools: Sequence[Tool]) -> list[ChatCompletionToolParam]:
    """将统一的 Tool 格式转换为 OpenAI 的格式"""
    return [cast(ChatCompletionToolParam, tool) for tool in tools]


def _serialize_messages(messages: Sequence[Message]) -> list[ChatCompletionMessageParam]:
    serialized: list[ChatCompletionMessageParam] = []

    for message in messages:
        msg_dict: dict[str, Any] = {
            "role": _transfrom_role(message.role),
            "content": _serialize_content(message.content)
        }

        # 处理 assistant 消息的 tool_calls
        if message.role == "assistant" and message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                }
                for tc in message.tool_calls
            ]

        # 处理 tool 消息的 tool_call_id
        if message.role == "tool" and message.tool_call_id:
            msg_dict["tool_call_id"] = message.tool_call_id

        serialized.append(cast(ChatCompletionMessageParam, msg_dict))

    return serialized

def _transfrom_role(role: Role) -> Literal["user", "assistant", "system", "tool"]:
    match role:
        case "user":
            return "user"
        case "assistant":
            return "assistant"
        case "system":
            return "system"
        case "tool":
            return "tool"
        case _ :
            raise ValueError(f"Unknow role type: {role}")
    

def _serialize_content(content: MessageContent) -> Any:
    if isinstance(content, str):
        return content

    serialized_parts: list[dict[str, Any]] = []
    for part in content:
        match part:
            case {"type": "text", "text": text}:
                serialized_parts.append({"type": "text", "text": text})
            case {"type": "image_url", "url": url}:
                serialized_parts.append({"type": "image_url", "url": url})
            case {"type": "video_url", "url": url}:
                serialized_parts.append({"type": "video_url", "url": url})
            case _:
                raise ValueError(f"Unknow part type: {part}")

    return serialized_parts


def _deserialize_response(response: ChatCompletion) -> ModelResponse:
    """将 OpenAI 响应转换为统一的 ModelResponse 格式"""
    choice = response.choices[0]
    message = choice.message

    # 提取消息内容
    content: MessageContent = message.content or ""

    # 提取工具调用（如果有）
    tool_calls = None
    if message.tool_calls:
        from ..types import ToolCall
        tool_calls = [
            ToolCall(
                id=tc.id,
                type="function",
                function={
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            )
            for tc in message.tool_calls
            if tc.type == "function"
        ]

    return ModelResponse(
        message=Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls
        )
    )


def _deserialize_stream_chunk(chunk: ChatCompletionChunk) -> StreamChunk | None:
    """将 OpenAI 流式 chunk 转换为统一的 StreamChunk 格式"""
    if not chunk.choices:
        return None

    choice = chunk.choices[0]
    delta = choice.delta

    # 提取内容
    content = delta.content

    # 提取工具调用
    tool_call = None
    if delta.tool_calls:
        from ..types import ToolCall
        tc = delta.tool_calls[0]
        if tc.function:
            tool_call = ToolCall(
                id=tc.id or "",
                type="function",
                function={
                    "name": tc.function.name or "",
                    "arguments": tc.function.arguments or ""
                }
            )

    # 确定 chunk 类型
    chunk_type: Literal["content", "thinkg_content", "tool_call"] = "content"
    if tool_call:
        chunk_type = "tool_call"

    # 提取结束原因
    finish_reason = None
    if choice.finish_reason:
        match choice.finish_reason:
            case "stop":
                finish_reason = "stop"
            case "tool_calls":
                finish_reason = "tool_use"
            case "length":
                finish_reason = "max_tokens"
            case _:
                finish_reason = "error"

    return StreamChunk(
        type=chunk_type,
        content=content,
        tool_call=tool_call,
        finish_reason=finish_reason
    )