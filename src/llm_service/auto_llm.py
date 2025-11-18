"""AutoLLM - 智能大模型，根据需求自动选择最合适的模型"""
from __future__ import annotations

import os
from typing import Iterable, Literal, Sequence

from dotenv import load_dotenv

from .llm import LLM
from .model_registry import MODEL_REGISTRY, ModelCapability, get_model_capability
from .providers.common import OpenAIWrapper
from .decorators.function_call_decorator import FunctionCallDecorator
from .types import Message, ModelResponse, StreamChunk, Tool


PreferenceType = Literal["speed", "quality"]
FCModeType = Literal["native", "prompt"]


class AutoLLM(LLM):
    """
    AutoLLM - 智能大模型

    根据用户偏好自动选择最合适的模型，或支持手动指定模型。

    使用方式：
        # 自动选择：速度优先
        llm = AutoLLM(prefer="speed")

        # 自动选择：质量优先 + 多模态
        llm = AutoLLM(prefer="quality", multimodal=True)

        # 手动指定模型
        llm = AutoLLM(model="qwen-max")
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        prefer: PreferenceType = "quality",
        multimodal: bool = False,
        fc_mode: FCModeType = "native"
    ) -> None:
        """
        初始化 AutoLLM

        Args:
            model: 手动指定模型名称，如果指定则忽略 prefer 和 multimodal
            prefer: 偏好类型，"speed" 或 "quality"
            multimodal: 是否需要多模态支持
            fc_mode: function call 模式，"native"（原生）或 "prompt"（prompt 模拟）
        """
        # 加载 .env 文件
        load_dotenv(override=False)

        # 选择模型
        if model:
            # 手动指定模型
            selected_model = model
            capability = get_model_capability(model)
        else:
            # 自动选择模型
            capability = self._select_model(prefer=prefer, multimodal=multimodal)
            selected_model = capability.model_name

        # 检查环境变量配置
        api_key = os.getenv(f"{capability.env_key_prefix}_API_KEY")
        base_url = os.getenv(f"{capability.env_key_prefix}_BASE_URL")

        if not api_key:
            raise ValueError(
                f"Missing API key for {capability.provider}. "
                f"Please set {capability.env_key_prefix}_API_KEY in .env file"
            )

        if not base_url:
            raise ValueError(
                f"Missing base URL for {capability.provider}. "
                f"Please set {capability.env_key_prefix}_BASE_URL in .env file"
            )

        # 创建底层 LLM wrapper
        base_llm = OpenAIWrapper(
            model=selected_model,
            api_key=api_key,
            base_url=base_url
        )

        # 根据 fc_mode 决定是否使用装饰器
        if fc_mode == "prompt":
            self._llm = FunctionCallDecorator(inner=base_llm)
        else:
            self._llm = base_llm

        # 保存选择信息
        self._selected_model = selected_model
        self._capability = capability
        self._fc_mode = fc_mode

    def _select_model(
        self,
        prefer: PreferenceType,
        multimodal: bool
    ) -> ModelCapability:
        """
        根据偏好自动选择最合适的模型

        Args:
            prefer: 偏好类型
            multimodal: 是否需要多模态

        Returns:
            选中的模型能力信息
        """
        # 筛选符合条件的模型
        candidates = [
            cap for cap in MODEL_REGISTRY.values()
            if not multimodal or cap.multimodal
        ]

        if not candidates:
            raise ValueError("No model matches the requirements")

        # 根据偏好排序
        if prefer == "speed":
            # 速度优先：按 speed_score 降序，speed 相同时按 quality 降序
            candidates.sort(
                key=lambda x: (x.speed_score, x.quality_score),
                reverse=True
            )
        else:  # quality
            # 质量优先：按 quality_score 降序，quality 相同时按 speed 降序
            candidates.sort(
                key=lambda x: (x.quality_score, x.speed_score),
                reverse=True
            )

        # 返回最佳选择
        return candidates[0]

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None
    ) -> ModelResponse:
        """执行一次标准的大模型推理"""
        return self._llm.complete(messages, tools)

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None
    ) -> Iterable[StreamChunk]:
        """执行一次流式推理，逐个返回 chunk"""
        return self._llm.stream(messages, tools)

    @property
    def selected_model(self) -> str:
        """返回当前选中的模型名称"""
        return self._selected_model

    @property
    def capability(self) -> ModelCapability:
        """返回当前模型的能力信息"""
        return self._capability


__all__ = ["AutoLLM", "PreferenceType", "FCModeType"]
