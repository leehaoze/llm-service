# AutoLLM 使用指南

AutoLLM 是一个智能大模型，可以根据你的需求自动选择最合适的底层模型，或者手动指定特定模型。

## 特性

- **智能选择**: 根据速度/质量偏好自动选择最合适的模型
- **多模态支持**: 自动筛选支持多模态的模型
- **手动指定**: 支持直接指定使用某个模型
- **统一接口**: 与 OpenAIWrapper 相同的接口，无缝切换

## 支持的模型

目前支持以下国内大模型：

### 通义千问 (Qwen)
- `qwen-turbo`: 速度快，成本低，适合简单任务
- `qwen-plus`: 平衡性能，适合大多数场景
- `qwen-max`: 质量最高，适合复杂任务
- `qwen-vl-plus`: 多模态，平衡性能
- `qwen-vl-max`: 多模态，质量最高

### DeepSeek
- `deepseek-chat`: 速度快，质量好，性价比高

### 豆包 (Doubao)
- `Doubao-lite-4k`: 速度快，成本低
- `Doubao-pro-4k`: 专业版，平衡性能
- `Doubao-pro-32k`: 长上下文支持

## 环境变量配置

在 `.env` 文件中配置各个模型的 API Key 和 Base URL：

```bash
# 通义千问
QWEN_API_KEY=your_qwen_api_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 豆包
DOUBAO_API_KEY=your_doubao_api_key
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```

**注意**: 只有配置了相关环境变量的模型才能使用，否则会抛出异常。

## 基本使用

### 1. 自动选择 - 速度优先

当你需要快速响应，对质量要求不高时：

```python
from llm_service import AutoLLM, types

# 创建路由器：速度优先
llm = AutoLLM(prefer="speed")

# 查看选中的模型
print(f"使用模型: {llm.selected_model}")
# 输出: 使用模型: qwen-turbo 或 Doubao-lite-4k

# 发送消息
messages = [types.Message(role="user", content="你好")]
response = llm.complete(messages)
print(response.message.content)
```

### 2. 自动选择 - 质量优先

当你需要高质量的输出时：

```python
# 创建路由器：质量优先
llm = AutoLLM(prefer="quality")

print(f"使用模型: {llm.selected_model}")
# 输出: 使用模型: qwen-max

messages = [types.Message(role="user", content="写一篇关于 AI 的文章")]
response = llm.complete(messages)
print(response.message.content)
```

### 3. 多模态支持

当你需要处理图片、视频等多模态输入时：

```python
# 创建路由器：质量优先 + 多模态
llm = AutoLLM(prefer="quality", multimodal=True)

print(f"使用模型: {llm.selected_model}")
# 输出: 使用模型: qwen-vl-max

# 发送图片
messages = [
    types.Message(
        role="user",
        content=[
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "url": "https://example.com/image.jpg"}
        ]
    )
]
response = llm.complete(messages)
print(response.message.content)
```

### 4. 手动指定模型

当你明确知道要使用哪个模型时：

```python
# 手动指定使用 deepseek-chat
llm = AutoLLM(model="deepseek-chat")

print(f"使用模型: {llm.selected_model}")
# 输出: 使用模型: deepseek-chat

messages = [types.Message(role="user", content="解释什么是递归")]
response = llm.complete(messages)
print(response.message.content)
```

### 5. 流式输出

支持流式输出，与 `OpenAIWrapper` 接口一致：

```python
llm = AutoLLM(prefer="speed")

messages = [types.Message(role="user", content="讲个故事")]

for chunk in llm.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### 6. 列出所有可用模型

```python
from llm_service import list_available_models, get_model_capability

# 列出所有模型
models = list_available_models()
print(models)

# 查看某个模型的能力
capability = get_model_capability("qwen-max")
print(f"速度评分: {capability.speed_score}/10")
print(f"质量评分: {capability.quality_score}/10")
print(f"多模态支持: {capability.multimodal}")
```

## 选择算法

AutoLLM 的自动选择算法如下：

1. **筛选阶段**: 根据 `multimodal` 参数筛选出符合条件的模型
   - 如果 `multimodal=True`，只保留支持多模态的模型
   - 如果 `multimodal=False`，所有模型都符合条件

2. **排序阶段**: 根据 `prefer` 参数排序
   - `prefer="speed"`: 按速度评分降序排列，速度相同时按质量排序
   - `prefer="quality"`: 按质量评分降序排列，质量相同时按速度排序

3. **选择**: 选择排序后的第一个模型（评分最高的）

## 模型能力矩阵

| 模型 | 速度评分 | 质量评分 | 多模态 | 提供商 |
|------|---------|---------|--------|--------|
| qwen-turbo | 9 | 6 | ❌ | Qwen |
| qwen-plus | 7 | 8 | ❌ | Qwen |
| qwen-max | 5 | 9 | ❌ | Qwen |
| qwen-vl-plus | 7 | 8 | ✅ | Qwen |
| qwen-vl-max | 5 | 9 | ✅ | Qwen |
| deepseek-chat | 8 | 8 | ❌ | DeepSeek |
| Doubao-lite-4k | 9 | 6 | ❌ | Doubao |
| Doubao-pro-4k | 7 | 8 | ❌ | Doubao |
| Doubao-pro-32k | 6 | 8 | ❌ | Doubao |

评分说明：
- 速度评分: 1-10，越高越快
- 质量评分: 1-10，越高质量越好

## 与其他组件配合使用

### 与 FunctionCallDecorator 配合

```python
from llm_service import AutoLLM, FunctionCallDecorator

# 创建路由器
base_llm = AutoLLM(prefer="quality")

# 包装为 prompt 模式的 function call
llm = FunctionCallDecorator(inner=base_llm)

# 使用 function call
tools = [...]
response = llm.complete(messages, tools=tools)
```

## 错误处理

### 模型不存在

```python
try:
    llm = AutoLLM(model="nonexistent-model")
except ValueError as e:
    print(e)
    # 输出: Unknown model: nonexistent-model. Available models: ...
```

### 缺少环境变量

```python
try:
    llm = AutoLLM(model="qwen-max")
except ValueError as e:
    print(e)
    # 输出: Missing API key for qwen. Please set QWEN_API_KEY in .env file
```

### 无符合条件的模型

```python
# 如果所有支持多模态的模型都没有配置环境变量
try:
    llm = AutoLLM(prefer="speed", multimodal=True)
except ValueError as e:
    print(e)
    # 输出: Missing API key for ... 或 No model matches the requirements
```

## 最佳实践

1. **统一配置管理**: 在项目根目录创建 `.env` 文件，集中管理所有模型的配置
2. **按需配置**: 只配置你需要使用的模型的 API Key，避免浪费
3. **合理选择偏好**:
   - 聊天、问答等交互场景: `prefer="speed"`
   - 内容生成、翻译等质量敏感场景: `prefer="quality"`
   - 图片、视频理解: `multimodal=True`
4. **查看选中模型**: 使用 `llm.selected_model` 查看实际使用的模型，便于调试和日志记录

## 示例代码

完整示例代码请参考: [examples/auto_llm_demo.py](../examples/auto_llm_demo.py)
