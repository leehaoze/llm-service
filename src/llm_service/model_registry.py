"""模型注册表，定义所有支持的模型及其能力"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ModelProvider = Literal["qwen", "deepseek", "doubao"]


@dataclass(frozen=True)
class ModelCapability:
    """模型能力定义"""

    provider: ModelProvider
    model_name: str
    speed_score: int  # 1-10，越高越快
    quality_score: int  # 1-10，越高质量越好
    multimodal: bool  # 是否支持多模态
    env_key_prefix: str  # 环境变量前缀，如 "QWEN"


# 模型注册表
MODEL_REGISTRY: dict[str, ModelCapability] = {
    # 通义千问系列
    "qwen-turbo": ModelCapability(
        provider="qwen",
        model_name="qwen-turbo",
        speed_score=9,
        quality_score=6,
        multimodal=False,
        env_key_prefix="QWEN"
    ),
    "qwen-plus": ModelCapability(
        provider="qwen",
        model_name="qwen-plus",
        speed_score=7,
        quality_score=8,
        multimodal=False,
        env_key_prefix="QWEN"
    ),
    "qwen-max": ModelCapability(
        provider="qwen",
        model_name="qwen-max",
        speed_score=5,
        quality_score=9,
        multimodal=False,
        env_key_prefix="QWEN"
    ),
    "qwen-vl-plus": ModelCapability(
        provider="qwen",
        model_name="qwen-vl-plus",
        speed_score=7,
        quality_score=8,
        multimodal=True,
        env_key_prefix="QWEN"
    ),
    "qwen-vl-max": ModelCapability(
        provider="qwen",
        model_name="qwen-vl-max",
        speed_score=5,
        quality_score=9,
        multimodal=True,
        env_key_prefix="QWEN"
    ),

    # DeepSeek 系列
    "deepseek-chat": ModelCapability(
        provider="deepseek",
        model_name="deepseek-v3-1-terminus",
        speed_score=8,
        quality_score=8,
        multimodal=False,
        env_key_prefix="DEEPSEEK"
    ),

    # 豆包系列
    "Doubao-lite-4k": ModelCapability(
        provider="doubao",
        model_name="doubao-seed-1.6-flash",
        speed_score=9,
        quality_score=6,
        multimodal=False,
        env_key_prefix="DOUBAO"
    ),
    # "Doubao-pro-4k": ModelCapability(
    #     provider="doubao",
    #     model_name="Doubao-pro-4k",
    #     speed_score=7,
    #     quality_score=8,
    #     multimodal=False,
    #     env_key_prefix="DOUBAO"
    # ),
    "Doubao-pro-32k": ModelCapability(
        provider="doubao",
        model_name="doubao-seed-1-6-250615",
        speed_score=6,
        quality_score=8,
        multimodal=False,
        env_key_prefix="DOUBAO"
    ),
}


def get_model_capability(model_name: str) -> ModelCapability:
    """获取指定模型的能力信息"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {', '.join(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def list_available_models() -> list[str]:
    """列出所有可用的模型"""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    "ModelProvider",
    "ModelCapability",
    "MODEL_REGISTRY",
    "get_model_capability",
    "list_available_models",
]
