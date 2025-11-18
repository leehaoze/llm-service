"""Core package for the llm-service project."""

from __future__ import annotations

from . import types
from .providers import OpenAIWrapper
from .decorators import FunctionCallDecorator

# 为了保持兼容性，提供别名
OpenAILLM = OpenAIWrapper

__all__ = ["types", "OpenAIWrapper", "OpenAILLM", "FunctionCallDecorator"]
