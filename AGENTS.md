# Repository Guidelines

## 构建、测试与开发命令
- `uv sync`：安装并锁定依赖，首次克隆后执行。
- `uv run pytest`：运行全部 Pytest 套件，支持 `-k` / `-m` 过滤。
- `uv run python -m llm_service`：以模块方式调试包入口，便于集成演练。
- `uv run python -m build`：生成分发用的 sdist 与 wheel，可结合 CI 使用。

## 编码风格与命名规范
遵循 PEP 8 与类型提示优先策略，使用 4 空格缩进。模块名称保持蛇形（如 `data_loader.py`），公开 API 通过 `__all__` 明示。常量全大写，内部函数以下划线前缀表示。提交前请运行 `ruff format` 或 `black`（如已安装）保持一致性。

## 测试准则
统一使用 Pytest，测试函数命名为 `test_<行为>`，夹具放在 `conftest.py`。新特性需至少 80% 行覆盖，错误路径使用参数化测试展示多场景。对外接口新增时，请在 `tests/` 中创建对等模块，确保导入、边界与异常行为均被校验。

## 提交与 Pull Request 指南
现有历史表明采用 Conventional Commits（例：`chore: bootstrap project`）。在此基础上使用 `feat|fix|chore|docs|test|refactor` 前缀，主题不超过 72 字符。PR 需包含：变更摘要、测试结果（命令及输出概要）、关联议题编号，以及若涉及接口或配置调整请附示例或截图。保持分支命名 `type/feature-description`，并在获得一名维护者批准后合并。

## 安全与配置提示
切勿将 API Key、令牌或客户配置写入仓库，可通过环境变量与 `.env`（列入 `.gitignore`）管理。更新依赖前先审查 `uv.lock` 差异，确认无高危许可证或漏洞，再提交。
