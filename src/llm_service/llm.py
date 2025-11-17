"""定义 llm-service 在内部使用的 LLM 协议。"""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence

from .types import Message, ModelResponse, StreamChunk


class LLM(Protocol):
    """描述一个可以被 llm-service 调用的大模型接口。"""

    def complete(self, messages: Sequence[Message]) -> ModelResponse:
        """执行一次标准的大模型推理。"""
        ...

    def stream(self, messages: Sequence[Message]) -> Iterable[StreamChunk]:
        """执行一次流式推理，逐个返回 chunk。"""
        ...


__all__ = ["LLM"]
