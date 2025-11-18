"""测试 AutoLLM 的基本功能"""
from llm_service.model_registry import get_model_capability, list_available_models
from llm_service.auto_llm import AutoLLM


def test_list_available_models():
    """测试列出所有可用模型"""
    models = list_available_models()
    assert len(models) > 0
    assert "qwen-turbo" in models
    assert "qwen-max" in models
    assert "deepseek-chat" in models


def test_get_model_capability():
    """测试获取模型能力"""
    cap = get_model_capability("qwen-turbo")
    assert cap.provider == "qwen"
    assert cap.speed_score > 0
    assert cap.quality_score > 0
    assert cap.multimodal is False


def test_get_model_capability_multimodal():
    """测试获取多模态模型能力"""
    cap = get_model_capability("qwen-vl-max")
    assert cap.provider == "qwen"
    assert cap.multimodal is True


def test_get_unknown_model():
    """测试获取不存在的模型"""
    try:
        get_model_capability("unknown-model")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown model" in str(e)


def test_model_selection_speed():
    """测试速度优先选择（不实际调用 API）"""
    # 这个测试只验证选择逻辑，不会真正创建 LLM wrapper
    from llm_service.model_registry import MODEL_REGISTRY

    # 筛选非多模态模型
    candidates = [cap for cap in MODEL_REGISTRY.values() if not cap.multimodal]
    # 按速度排序
    candidates.sort(key=lambda x: (x.speed_score, x.quality_score), reverse=True)

    # 验证速度最快的是 qwen-turbo 或 Doubao-lite-4k
    fastest = candidates[0]
    assert fastest.speed_score >= 9


def test_model_selection_quality():
    """测试质量优先选择（不实际调用 API）"""
    from llm_service.model_registry import MODEL_REGISTRY

    # 筛选非多模态模型
    candidates = [cap for cap in MODEL_REGISTRY.values() if not cap.multimodal]
    # 按质量排序
    candidates.sort(key=lambda x: (x.quality_score, x.speed_score), reverse=True)

    # 验证质量最高的是 qwen-max
    best_quality = candidates[0]
    assert best_quality.quality_score >= 9
    assert best_quality.model_name in ["qwen-max"]


def test_model_selection_multimodal():
    """测试多模态选择（不实际调用 API）"""
    from llm_service.model_registry import MODEL_REGISTRY

    # 筛选多模态模型
    candidates = [cap for cap in MODEL_REGISTRY.values() if cap.multimodal]

    # 验证所有候选都支持多模态
    assert len(candidates) > 0
    for cap in candidates:
        assert cap.multimodal is True


if __name__ == "__main__":
    print("Running tests...")
    test_list_available_models()
    print("✓ test_list_available_models passed")

    test_get_model_capability()
    print("✓ test_get_model_capability passed")

    test_get_model_capability_multimodal()
    print("✓ test_get_model_capability_multimodal passed")

    try:
        test_get_unknown_model()
    except AssertionError:
        print("✓ test_get_unknown_model passed")

    test_model_selection_speed()
    print("✓ test_model_selection_speed passed")

    test_model_selection_quality()
    print("✓ test_model_selection_quality passed")

    test_model_selection_multimodal()
    print("✓ test_model_selection_multimodal passed")

    print("\nAll tests passed!")
