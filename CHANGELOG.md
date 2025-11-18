# Changelog

## [Unreleased]

### Added
- 新增 `FunctionCallDecorator` 装饰器，用于将任何 LLM 的 function call 转换为 prompt 模式
- 完整的类型注解支持
- 详细的文档和示例代码

### Fixed
- 修复了导入问题：统一使用 `OpenAIWrapper` 类名，同时提供 `OpenAILLM` 别名保持兼容性
- 修复了类型提示问题：
  - `_parse_prompt_fc_output` 现在正确使用 `MessageContent` 类型
  - 改进了 `ContentPart` 的类型处理
  - 增强了 `_convert_to_tool_calls` 的类型安全性

### Changed
- 主要导出类名从 `OpenAILLM` 改为 `OpenAIWrapper`（保留 `OpenAILLM` 作为别名）

## [0.1.0] - 2025-01-18

### Added
- 初始版本
- 统一的 LLM 接口
- OpenAI 兼容实现
- 基础类型定义
