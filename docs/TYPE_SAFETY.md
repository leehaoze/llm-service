# 类型安全说明

## 概述

`FunctionCallDecorator` 使用了完整的 Python 类型注解，确保类型安全和良好的 IDE 支持。

## 关键类型定义

### MessageContent

```python
# 来自 types.py
ContentPart: TypeAlias = TextPart | ImagePart | VideoPart
MessageContent: TypeAlias = str | Sequence[ContentPart]
```

`MessageContent` 可以是：
- 纯字符串：`"hello"`
- 多个内容片段：`[{"type": "text", "text": "hello"}, {"type": "image_url", "url": "..."}]`

### 类型处理挑战

#### 1. ContentPart 的处理

问题：`ContentPart` 是 `TextPart | ImagePart | VideoPart` 的联合类型，每个都是 `TypedDict`。

解决方案：
```python
def _parse_prompt_fc_output(content: MessageContent) -> tuple[list[ToolCall], str | None]:
    if isinstance(content, str):
        text = content
    else:
        # content 是 Sequence[ContentPart]
        text_parts: list[str] = []
        for part in content:
            if part.get("type") == "text":
                text_value = part.get("text")
                if text_value is not None and isinstance(text_value, str):
                    text_parts.append(text_value)
        text = "".join(text_parts)
```

#### 2. JSON 解析结果的类型

问题：`json.loads()` 返回 `Any` 类型，需要明确类型约束。

解决方案：
```python
def _extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    """返回类型明确为 dict、list 或 None"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # ... 其他解析逻辑
```

#### 3. dict 字段访问的类型安全

问题：`dict.get()` 返回 `Unknown` 类型，需要类型检查。

解决方案：
```python
def _convert_to_tool_calls(data: dict[str, Any] | list[Any]) -> list[ToolCall]:
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        # 使用 cast 进行类型缩窄
        item_dict = cast(dict[str, Any], item)

        # 提取字段并检查类型
        call_id = item_dict.get("id")
        func_name = item_dict.get("name")

        # 使用条件表达式确保类型安全
        tool_call = ToolCall(
            id=str(call_id) if call_id is not None else f"call_{i}",
            type="function",
            function={
                "name": str(func_name),
                "arguments": json.dumps(func_args, ensure_ascii=False),
            },
        )
```

## 类型导入

完整的类型导入确保了类型检查器能够正确推断：

```python
from typing import Any, Iterable, Sequence, cast

from ..llm import LLM
from ..types import (
    ContentPart,
    Message,
    MessageContent,
    ModelResponse,
    StreamChunk,
    Tool,
    ToolCall,
)
```

## 类型检查工具支持

### Pylance

- ✓ 所有参数类型明确
- ✓ 返回值类型明确
- ✓ 使用 `cast` 进行类型缩窄
- ✓ 运行时类型检查（`isinstance`）

### mypy

可以通过以下命令进行类型检查（如果安装了 mypy）：

```bash
mypy src/llm_service/decorators/function_call_decorator.py
```

## 常见类型问题及解决方案

### 问题 1: TypedDict 的 get() 返回 Unknown

**错误**：
```python
part.get("text")  # 类型为 Unknown
```

**解决**：
```python
text_value = part.get("text")
if text_value is not None and isinstance(text_value, str):
    text_parts.append(text_value)
```

### 问题 2: Any 类型的 dict 访问

**错误**：
```python
item["name"]  # 如果 item 是 Any，返回 Unknown
```

**解决**：
```python
item_dict = cast(dict[str, Any], item)
func_name = item_dict.get("name")
tool_call = ToolCall(
    function={"name": str(func_name), ...}
)
```

### 问题 3: Sequence vs List

**错误**：
```python
items = data  # data 可能是 list[Any]，但 Sequence 是协变的
```

**解决**：
```python
items = list(data)  # 显式转换为 list
```

## 类型安全的好处

1. **IDE 自动补全**：准确的类型提示帮助 IDE 提供更好的代码补全
2. **错误提前发现**：类型检查器在运行前就能发现潜在问题
3. **代码可维护性**：类型注解作为文档，帮助理解代码
4. **重构安全**：修改代码时，类型检查器会提示相关影响

## 测试验证

所有类型修复都经过测试验证：

```bash
$ python tests/test_decorator_simple.py
============================================================
运行 FunctionCallDecorator 测试
============================================================
测试：单个工具调用...
✓ 测试通过
测试：多个工具调用...
✓ 测试通过
测试：普通文本回复...
✓ 测试通过
测试：不传 tools...
✓ 测试通过
测试：从 markdown 代码块中提取 JSON...
✓ 测试通过

============================================================
所有测试通过！✓
============================================================
```

## 总结

通过以下技术确保了类型安全：

- ✓ 明确的类型注解（`MessageContent`, `dict[str, Any]`, `list[Any]`）
- ✓ 运行时类型检查（`isinstance`）
- ✓ 类型缩窄（`cast`）
- ✓ 条件表达式处理 `None` 值
- ✓ 完整的类型导入

代码既保证了类型安全，又保持了运行时的灵活性。
