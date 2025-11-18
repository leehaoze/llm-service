# FunctionCallDecorator 快速参考

## 一句话总结

`FunctionCallDecorator` 是一个装饰器，可以让任何 LLM（无论是否原生支持 function call）都能通过 prompt 模式来实现 function calling。

## 5 秒上手

```python
from llm_service import YourLLM, FunctionCallDecorator

# 包装你的 LLM
llm = FunctionCallDecorator(inner=YourLLM())

# 正常使用，就像它原生支持 FC 一样
response = llm.complete(messages, tools=tools)
```

## 为什么需要它？

### 场景 1：你的模型不支持原生 FC
```python
# 某些 Llama 模型不支持原生 FC
llm = LlamaLLM(model="llama-3-70b")  # ❌ 不支持 tools 参数

# 用装饰器包装后
llm = FunctionCallDecorator(inner=LlamaLLM(model="llama-3-70b"))
response = llm.complete(messages, tools=tools)  # ✓ 现在支持了！
```

### 场景 2：你想对比原生 FC 和 Prompt FC 的效果
```python
# 方式 A：原生 FC
openai_native = OpenAILLM(model="gpt-4")
response_a = openai_native.complete(messages, tools=tools)

# 方式 B：Prompt FC
openai_prompt = FunctionCallDecorator(inner=OpenAILLM(model="gpt-4"))
response_b = openai_prompt.complete(messages, tools=tools)

# 对比效果
compare_results(response_a, response_b)
```

## 它是怎么工作的？

### 输入（你的代码）
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取天气",
        "parameters": {...}
    }
}]

llm.complete(messages, tools=tools)
```

### 内部处理
```
1. 装饰器把 tools 转换成 system prompt：
   "You are an AI that calls functions.
    Available functions: get_weather - 获取天气
    When you need to call a function, output JSON: {...}"

2. 调用底层 LLM（不传 tools）

3. LLM 返回：'{"name": "get_weather", "arguments": {"city": "北京"}}'

4. 装饰器解析 JSON，转换成标准的 ToolCall 结构
```

### 输出（你收到的）
```python
response.message.tool_calls = [
    {
        "id": "call_0",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"city": "北京"}'
        }
    }
]
```

## 支持的 JSON 格式

装饰器能识别多种格式：

### ✓ 纯 JSON
```json
{"name": "get_weather", "arguments": {"city": "北京"}}
```

### ✓ JSON 数组（多个调用）
```json
[
  {"name": "get_weather", "arguments": {"city": "北京"}},
  {"name": "get_weather", "arguments": {"city": "上海"}}
]
```

### ✓ Markdown 代码块
````
```json
{"name": "get_weather", "arguments": {"city": "北京"}}
```
````

### ✓ 混合文本
```
好的，让我查询天气：
{"name": "get_weather", "arguments": {"city": "北京"}}
```

## 构建时决定是否使用

推荐做法：

```python
def build_llm(provider: str, use_prompt_fc: bool = False):
    # 1. 创建基础 LLM
    if provider == "openai":
        base = OpenAILLM(model="gpt-4")
    elif provider == "llama":
        base = LlamaLLM(model="llama-3")

    # 2. 根据配置决定是否装饰
    if use_prompt_fc:
        return FunctionCallDecorator(inner=base)

    return base

# 使用
llm = build_llm("openai", use_prompt_fc=True)  # 强制走 prompt 模式
llm = build_llm("llama", use_prompt_fc=True)   # 补充 FC 能力
llm = build_llm("openai", use_prompt_fc=False) # 使用原生 FC
```

## 常见问题

### Q: 装饰器会影响性能吗？
A:
- 当 `tools=None` 时：**零开销**，直接透传
- 当 `tools` 有值时：增加少量 prompt token 和 JSON 解析时间

### Q: 支持流式输出吗？
A: 支持！当前实现是先收集完整响应再解析。未来可以优化为真正的流式 JSON 解析。

### Q: 如果 JSON 解析失败怎么办？
A: 如果无法解析 JSON，会认为是普通文本回复，`tool_calls` 为 `None`。

### Q: 可以用于生产环境吗？
A: 可以！已经通过完整测试，类型安全，可以直接使用。

## 测试

```bash
python tests/test_decorator_simple.py
```

所有测试都会通过！

## 更多文档

- 详细文档：[docs/FunctionCallDecorator.md](docs/FunctionCallDecorator.md)
- 实现总结：[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- 示例代码：[examples/function_call_decorator_demo.py](examples/function_call_decorator_demo.py)
