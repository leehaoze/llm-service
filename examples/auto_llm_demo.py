"""AutoLLM 使用示例"""
from llm_service import AutoLLM, types, list_available_models


def example_auto_select_speed():
    """示例1: 自动选择 - 速度优先"""
    print("=" * 60)
    print("示例1: 速度优先")
    print("=" * 60)

    # 创建路由器：速度优先
    llm = AutoLLM(prefer="speed")
    print(f"选中的模型: {llm.selected_model}")
    print(f"模型能力: speed={llm.capability.speed_score}, "
          f"quality={llm.capability.quality_score}, "
          f"multimodal={llm.capability.multimodal}")

    # 发送消息
    messages = [types.Message(role="user", content="你好，介绍一下你自己")]
    response = llm.complete(messages)
    print(f"\n回复: {response.message.content}\n")


def example_auto_select_quality():
    """示例2: 自动选择 - 质量优先"""
    print("=" * 60)
    print("示例2: 质量优先")
    print("=" * 60)

    # 创建路由器：质量优先
    llm = AutoLLM(prefer="quality")
    print(f"选中的模型: {llm.selected_model}")
    print(f"模型能力: speed={llm.capability.speed_score}, "
          f"quality={llm.capability.quality_score}, "
          f"multimodal={llm.capability.multimodal}")

    # 发送消息
    messages = [types.Message(role="user", content="写一首关于春天的诗")]
    response = llm.complete(messages)
    print(f"\n回复: {response.message.content}\n")


def example_multimodal():
    """示例3: 多模态支持"""
    print("=" * 60)
    print("示例3: 多模态 + 质量优先")
    print("=" * 60)

    # 创建路由器：需要多模态支持，质量优先
    llm = AutoLLM(prefer="quality", multimodal=True)
    print(f"选中的模型: {llm.selected_model}")
    print(f"模型能力: speed={llm.capability.speed_score}, "
          f"quality={llm.capability.quality_score}, "
          f"multimodal={llm.capability.multimodal}")

    # 发送多模态消息（图片）
    messages = [
        types.Message(
            role="user",
            content=[
                {"type": "text", "text": "这张图片里有什么？"},
                {"type": "image_url", "url": "https://example.com/image.jpg"}
            ]
        )
    ]
    response = llm.complete(messages)
    print(f"\n回复: {response.message.content}\n")


def example_manual_select():
    """示例4: 手动指定模型"""
    print("=" * 60)
    print("示例4: 手动指定模型")
    print("=" * 60)

    # 手动指定使用 deepseek-chat
    llm = AutoLLM(model="deepseek-chat")
    print(f"选中的模型: {llm.selected_model}")
    print(f"模型能力: speed={llm.capability.speed_score}, "
          f"quality={llm.capability.quality_score}, "
          f"multimodal={llm.capability.multimodal}")

    # 发送消息
    messages = [types.Message(role="user", content="解释一下什么是递归")]
    response = llm.complete(messages)
    print(f"\n回复: {response.message.content}\n")


def example_list_models():
    """示例5: 列出所有可用模型"""
    print("=" * 60)
    print("示例5: 列出所有可用模型")
    print("=" * 60)

    models = list_available_models()
    print(f"共有 {len(models)} 个模型:")
    for model in models:
        print(f"  - {model}")
    print()


def example_streaming():
    """示例6: 流式输出"""
    print("=" * 60)
    print("示例6: 流式输出 (速度优先)")
    print("=" * 60)

    llm = AutoLLM(prefer="speed")
    print(f"使用模型: {llm.selected_model}\n")

    messages = [types.Message(role="user", content="讲一个笑话")]

    print("流式输出: ", end="", flush=True)
    for chunk in llm.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    # 列出所有可用模型
    example_list_models()

    # 运行各个示例
    # 注意：需要先在 .env 文件中配置对应的 API Key 和 Base URL

    try:
        example_auto_select_speed()
    except Exception as e:
        print(f"示例1 执行失败: {e}\n")

    try:
        example_auto_select_quality()
    except Exception as e:
        print(f"示例2 执行失败: {e}\n")

    try:
        example_multimodal()
    except Exception as e:
        print(f"示例3 执行失败: {e}\n")

    try:
        example_manual_select()
    except Exception as e:
        print(f"示例4 执行失败: {e}\n")

    try:
        example_streaming()
    except Exception as e:
        print(f"示例6 执行失败: {e}\n")
