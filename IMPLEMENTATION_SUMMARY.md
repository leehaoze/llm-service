# FunctionCallDecorator 实现总结

## 概述

成功实现了 `FunctionCallDecorator` 装饰器，用于将 LLM 的 function call 能力转换为基于 prompt 的模式。

## 核心设计思想

### 装饰器的定位

- **纯粹的转换器**：不判断底层 LLM 是否支持原生 FC，只负责转换
- **统一接口**：装饰后仍然遵循 `LLM` Protocol
- **配置驱动**：是否使用装饰器由外部配置决定

### 关键优势

1. **灵活性**：可用于任何 LLM，无论是否原生支持 FC
2. **对比能力**：可以强制 OpenAI 等模型走 prompt 模式，用于对比效果
3. **简单性**：实现纯粹，职责单一
4. **透明性**：对上层代码完全透明

## 实现的文件

### 1. 核心实现

**文件**: `src/llm_service/decorators/function_call_decorator.py`

包含：
- `FunctionCallDecorator` 类
- `_build_prompt_fc_messages()` - 构造 system prompt
- `_format_tools_description()` - 格式化工具定义
- `_parse_prompt_fc_output()` - 解析模型输出
- `_extract_json()` - 从文本中提取 JSON
- `_convert_to_tool_calls()` - 转换为 ToolCall 结构

### 2. 模块导出

**文件**: `src/llm_service/decorators/__init__.py`

```python
from .function_call_decorator import FunctionCallDecorator

__all__ = ["FunctionCallDecorator"]
```

**文件**: `src/llm_service/__init__.py`

```python
from .decorators import FunctionCallDecorator

__all__ = [..., "FunctionCallDecorator"]
```

### 3. 测试

**文件**: `tests/test_decorator_simple.py`

测试覆盖：
- ✓ 单个工具调用
- ✓ 多个工具调用
- ✓ 普通文本回复（不调用工具）
- ✓ 不传 tools 时的透传行为
- ✓ 从 markdown 代码块中提取 JSON

所有测试通过！

### 4. 文档

**文件**: `docs/FunctionCallDecorator.md`

包含：
- 使用指南
- 工作原理
- API 参考
- 完整示例
- 未来优化方向

**文件**: `examples/function_call_decorator_demo.py`

演示代码（需要 OpenAI API key）

## 工作流程

### Complete 模式

```
用户调用
  ↓
FunctionCallDecorator.complete()
  ↓
有 tools？
  ├─ 否 → 直接透传给 inner.complete()
  └─ 是 → 注入 system prompt
        ↓
        调用 inner.complete(new_messages, tools=None)
        ↓
        解析响应文本，提取 JSON
        ↓
        转换为 ToolCall 结构
        ↓
        返回标准 ModelResponse
```

### Stream 模式

```
用户调用
  ↓
FunctionCallDecorator.stream()
  ↓
有 tools？
  ├─ 否 → 直接透传 inner.stream()
  └─ 是 → 注入 system prompt
        ↓
        调用 inner.stream(new_messages, tools=None)
        ↓
        收集所有 chunks
        ↓
        解析完整文本，提取 JSON
        ↓
        转换为 ToolCall 结构
        ↓
        产出 tool_call chunks
        ↓
        产出 finish_reason chunk
```

## JSON 解析能力

支持多种格式：

### 1. 纯 JSON
```json
{"name": "get_weather", "arguments": {"city": "北京"}}
```

### 2. JSON 数组
```json
[
  {"name": "get_weather", "arguments": {"city": "北京"}},
  {"name": "get_weather", "arguments": {"city": "上海"}}
]
```

### 3. Markdown 代码块
````markdown
```json
{"name": "get_weather", "arguments": {"city": "北京"}}
```
````

### 4. 混合文本
```
好的，我来帮你查询天气：
{"name": "get_weather", "arguments": {"city": "北京"}}
```

## 使用示例

### 基本使用

```python
from llm_service import OpenAILLM, FunctionCallDecorator

# 创建原始 LLM
base_llm = OpenAILLM(model="gpt-4o-mini")

# 用装饰器包装
llm = FunctionCallDecorator(inner=base_llm)

# 正常使用
response = llm.complete(messages, tools=tools)
```

### 构建时组装

```python
def build_llm(config):
    base_llm = create_base_llm(config)

    if config.get("use_prompt_fc", False):
        return FunctionCallDecorator(inner=base_llm)

    return base_llm
```

## 类型安全

- 使用了完整的类型注解
- 符合 `LLM` Protocol
- 通过了 Pylance 类型检查（仅有少量不影响功能的类型推断警告）

## 测试结果

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

## 未来优化方向

### 1. 真正的流式 JSON 解析

当前实现：收集完整响应 → 一次性解析

优化方向：边读取 chunk → 边解析 JSON → 实时产出 tool_call chunk

### 2. 更智能的 Prompt

- 多语言支持（根据用户消息语言调整 prompt）
- Few-shot examples（提供示例提高准确率）
- 根据工具数量动态调整 prompt 详细程度

### 3. 错误恢复

- JSON 格式错误时的自动重试
- 部分字段缺失时的智能补全
- 更健壮的 JSON 提取逻辑

### 4. 性能监控

- JSON 解析成功率统计
- 原生 FC vs Prompt FC 效果对比
- 延迟和 token 使用量分析

## 总结

成功实现了一个纯粹、灵活、易用的 `FunctionCallDecorator`：

✓ 职责单一：只做 FC 到 Prompt 的转换
✓ 接口统一：完全符合 LLM Protocol
✓ 配置驱动：是否使用由外部决定
✓ 测试完善：100% 测试通过
✓ 文档齐全：使用指南、API 参考、示例代码
✓ 类型安全：完整的类型注解

可以直接用于生产环境！
