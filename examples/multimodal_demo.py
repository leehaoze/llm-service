"""演示多模态能力的测试脚本

支持测试:
- 图片理解
- 音频理解
- 视频理解
"""

import argparse
import base64
import sys
from pathlib import Path

from llm_service.providers import OpenAIWrapper
from llm_service import types


def encode_file_to_base64(file_path: str) -> str:
    """将文件编码为 base64 字符串"""
    with open(file_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_audio_format(file_path: str) -> str:
    """根据文件扩展名获取音频格式"""
    suffix = Path(file_path).suffix.lower()
    format_map = {
        ".wav": "wav",
        ".mp3": "mp3",
        ".flac": "flac",
        ".ogg": "ogg",
        ".m4a": "m4a",
    }
    return format_map.get(suffix, "wav")


def test_image(llm: OpenAIWrapper, image_source: str, prompt: str = "请描述这张图片的内容"):
    """测试图片理解能力

    Args:
        llm: LLM 实例
        image_source: 图片 URL 或本地文件路径
        prompt: 提示词
    """
    print("=" * 60)
    print("【图片理解测试】")
    print(f"图片: {image_source}")
    print(f"提示: {prompt}")
    print("=" * 60)

    # 判断是 URL 还是本地文件
    if image_source.startswith(("http://", "https://")):
        image_url = image_source
    else:
        # 本地文件，转换为 data URL
        file_path = Path(image_source)
        if not file_path.exists():
            print(f"错误: 文件不存在 - {image_source}")
            return

        # 获取 MIME 类型
        suffix = file_path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

        base64_data = encode_file_to_base64(image_source)
        image_url = f"data:{mime_type};base64,{base64_data}"

    messages = [
        types.Message(
            role="user",
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "url": image_url}
            ]
        )
    ]

    print("\n--- 响应 ---")
    response = llm.complete(messages)
    print(response.message.content)
    print()


def test_audio(llm: OpenAIWrapper, audio_source: str, prompt: str = "请描述这段音频的内容"):
    """测试音频理解能力

    Args:
        llm: LLM 实例
        audio_source: 音频文件路径（需要本地文件）
        prompt: 提示词
    """
    print("=" * 60)
    print("【音频理解测试】")
    print(f"音频: {audio_source}")
    print(f"提示: {prompt}")
    print("=" * 60)

    file_path = Path(audio_source)
    if not file_path.exists():
        print(f"错误: 文件不存在 - {audio_source}")
        return

    # 编码音频文件
    base64_data = encode_file_to_base64(audio_source)
    audio_format = get_audio_format(audio_source)

    messages = [
        types.Message(
            role="user",
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64_data,
                        "format": audio_format
                    }
                }
            ]
        )
    ]

    print("\n--- 响应 ---")
    response = llm.complete(messages)
    print(response.message.content)
    print()


def test_video(llm: OpenAIWrapper, video_source: str, prompt: str = "请描述这段视频的内容"):
    """测试视频理解能力

    Args:
        llm: LLM 实例
        video_source: 视频 URL
        prompt: 提示词
    """
    print("=" * 60)
    print("【视频理解测试】")
    print(f"视频: {video_source}")
    print(f"提示: {prompt}")
    print("=" * 60)

    # 视频目前只支持 URL
    if not video_source.startswith(("http://", "https://")):
        print("错误: 视频目前只支持 URL 格式")
        return

    messages = [
        types.Message(
            role="user",
            content=[
                {"type": "text", "text": prompt},
                {"type": "video_url", "url": video_source}
            ]
        )
    ]

    print("\n--- 响应 ---")
    response = llm.complete(messages)
    print(response.message.content)
    print()


def test_image_stream(llm: OpenAIWrapper, image_source: str, prompt: str = "请描述这张图片的内容"):
    """测试图片理解能力（流式输出）"""
    print("=" * 60)
    print("【图片理解测试 - 流式】")
    print(f"图片: {image_source}")
    print(f"提示: {prompt}")
    print("=" * 60)

    # 判断是 URL 还是本地文件
    if image_source.startswith(("http://", "https://")):
        image_url = image_source
    else:
        file_path = Path(image_source)
        if not file_path.exists():
            print(f"错误: 文件不存在 - {image_source}")
            return

        suffix = file_path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

        base64_data = encode_file_to_base64(image_source)
        image_url = f"data:{mime_type};base64,{base64_data}"

    messages = [
        types.Message(
            role="user",
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "url": image_url}
            ]
        )
    ]

    print("\n--- 响应 (流式) ---")
    for chunk in llm.stream(messages):
        if chunk.type == "content" and chunk.content:
            print(chunk.content, end="", flush=True)
        if chunk.finish_reason:
            print(f"\n[完成: {chunk.finish_reason}]")
    print()


def demo_all():
    """演示所有多模态能力"""
    llm = OpenAIWrapper()

    # 示例图片 URL
    sample_image = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"

    # 示例视频 URL
    sample_video = "https://yanzhipai-video.ks3-cn-beijing.ksyuncs.com/10s%E6%89%8B%E5%8A%BF.mp4"

    print("\n" + "=" * 60)
    print("多模态能力演示")
    print("=" * 60 + "\n")

    # 测试图片
    test_image(llm, sample_image, "这张图片里有什么动物？请详细描述。")

    # 测试视频
    test_video(llm, sample_video, "请描述视频中的人在做什么动作。")


def main():
    parser = argparse.ArgumentParser(
        description="多模态能力测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 测试图片（URL）
  python multimodal_demo.py image --source "https://example.com/image.jpg"

  # 测试图片（本地文件）
  python multimodal_demo.py image --source "./test.jpg" --prompt "图片中有什么？"

  # 测试图片（流式输出）
  python multimodal_demo.py image --source "./test.jpg" --stream

  # 测试音频（本地文件）
  python multimodal_demo.py audio --source "./test.wav" --prompt "这段音频说了什么？"

  # 测试视频（URL）
  python multimodal_demo.py video --source "https://example.com/video.mp4"

  # 运行完整演示
  python multimodal_demo.py demo
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="测试类型")

    # 图片测试
    image_parser = subparsers.add_parser("image", help="测试图片理解")
    image_parser.add_argument("--source", "-s", required=True, help="图片 URL 或本地文件路径")
    image_parser.add_argument("--prompt", "-p", default="请描述这张图片的内容", help="提示词")
    image_parser.add_argument("--stream", action="store_true", help="使用流式输出")

    # 音频测试
    audio_parser = subparsers.add_parser("audio", help="测试音频理解")
    audio_parser.add_argument("--source", "-s", required=True, help="音频文件路径")
    audio_parser.add_argument("--prompt", "-p", default="请描述这段音频的内容", help="提示词")

    # 视频测试
    video_parser = subparsers.add_parser("video", help="测试视频理解")
    video_parser.add_argument("--source", "-s", required=True, help="视频 URL")
    video_parser.add_argument("--prompt", "-p", default="请描述这段视频的内容", help="提示词")

    # 完整演示
    subparsers.add_parser("demo", help="运行完整演示")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 创建 LLM 实例
    llm = OpenAIWrapper()

    if args.command == "image":
        if args.stream:
            test_image_stream(llm, args.source, args.prompt)
        else:
            test_image(llm, args.source, args.prompt)
    elif args.command == "audio":
        test_audio(llm, args.source, args.prompt)
    elif args.command == "video":
        test_video(llm, args.source, args.prompt)
    elif args.command == "demo":
        demo_all()


if __name__ == "__main__":
    main()
