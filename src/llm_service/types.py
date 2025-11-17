"""对外暴露的消息、内容片段、chunk 类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, TypeAlias, TypedDict


Role: TypeAlias = Literal["system", "user", "assistant", "tool"]
"""标准的消息角色"""

ContentPartType: TypeAlias = Literal["text", "image_url", "video_url"]
"""消息类型，适用于输入与输出"""


class TextPart(TypedDict, total=True):
    """纯文本消息"""

    type: Literal["text"]
    text: str


class ImagePart(TypedDict, total=True):
    """图片类型消息"""

    type: Literal["image_url"]
    url: str


class VideoPart(TypedDict, total=True):
    """视频类型消息"""

    type: Literal["video_url"]
    url: str


# 统一的内容片段类型
ContentPart: TypeAlias = TextPart | ImagePart | VideoPart

# 消息内容：可以是纯字符串，也可以是多个Part
MessageContent: TypeAlias = str | Sequence[ContentPart]


class ToolCallFunction(TypedDict):
    """模型要调用的函数名称"""

    name: str
    arguments: str


class ToolCall(TypedDict):
    """模型回复的工具调用信息"""

    id: str
    type: Literal["function"]
    function: ToolCallFunction


@dataclass(slots=True)
class Message:
    """输入的消息格式"""

    role: Role
    content: MessageContent
    tool_call_id: str | None = None
    tool_calls: Sequence[ToolCall] | None = None


@dataclass(slots=True)
class ModelResponse:
    """大模型完整返回时，使用的数据结构"""

    message: Message


@dataclass(slots=True)
class StreamChunk:
    """模块返回的流式 chunk"""

    type: Literal["content", "thinkg_content", "tool_call"]
    content: str | None = None
    tool_call: ToolCall | None = None
    finish_reason: Literal["stop", "tool_use", "max_tokens", "error"] | None = None


__all__ = [
    "Role",
    "ContentPartType",
    "TextPart",
    "ImagePart",
    "VideoPart",
    "ContentPart",
    "MessageContent",
    "ToolCallFunction",
    "ToolCall",
    "Message",
    "StreamChunk",
]
