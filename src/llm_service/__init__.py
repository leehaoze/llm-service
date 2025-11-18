"""Core package for the llm-service project."""

from __future__ import annotations

from . import types
from .providers import OpenAIWrapper
from .decorators import FunctionCallDecorator
from .auto_llm import AutoLLM, PreferenceType, FCModeType
from .model_registry import list_available_models, get_model_capability

# 为了保持兼容性，提供别名
OpenAILLM = OpenAIWrapper

__all__ = [
    "types",
    "AutoLLM",
    "PreferenceType",
    "FCModeType",
    "list_available_models",
    "get_model_capability",
    # 底层实现（一般不需要直接使用）
    "OpenAIWrapper",
    "OpenAILLM",
    "FunctionCallDecorator",
]
