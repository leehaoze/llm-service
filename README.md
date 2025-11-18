# llm-service

智能 LLM 路由，自动选择最优模型，支持多模态和 function calling。

## 安装

```bash
pip install -e .
```

## 用法

```python
from llm_service import AutoLLM, types

# 按需求自动选择模型
llm = AutoLLM(prefer="speed")  # 速度优先
llm = AutoLLM(prefer="quality")  # 质量优先
llm = AutoLLM(prefer="quality", multimodal=True)  # 需要多模态能力

# function call 模式
llm = AutoLLM(prefer="quality", fc_mode="native")  # 原生 FC（默认）
llm = AutoLLM(prefer="quality", fc_mode="prompt")  # prompt 模拟 FC

response = llm.complete([types.Message(role="user", content="你好")])
print(response.message.content)
```

### Function Call

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}]

response = llm.complete(messages, tools=tools)
if response.message.tool_calls:
    for tc in response.message.tool_calls:
        print(tc['function']['name'], tc['function']['arguments'])
```

## 源码

| 文件 | 说明 |
|------|------|
| `src/llm_service/auto_llm.py` | AutoLLM 实现 |
| `src/llm_service/model_registry.py` | 模型注册表和能力定义 |
| `src/llm_service/types.py` | Message、Response 类型 |

进阶用法见 [docs/AutoLLM.md](docs/AutoLLM.md)
