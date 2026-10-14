"""
Prompt 模块：**主 agent 搭载的模型「是干什么的」** —— 职责提示词。

它与 app/agent/prompt 是两件事
----------------------------
    app/agent/prompt/     节点功能性提示词：这次调用**干什么**
                          （分类 / 摘要 / 记忆抽取…），键 = model_role，
                          进程级常量，随代码走
    app/services/prompt/  职责功能性提示词：这个模型**是干什么的**
                          （stock_research / python_programmer…），
                          键 = 职责标识，是**业务数据**

后者放在 services 而不是 agent 下，正是因为它属于业务：它将来会由谁来决定
（会话配置？Agent 目录？用户选择？）是一个业务问题，而提示词文本本身也会
从代码搬进库。放在这里，那一天要改的只有本模块。

谁在用
------
`app/agent/prompt/provider.py` 的 `duty_prompt()` —— 经 ServiceFactory 造
`PromptService` 再取（agent 侧只认 PromptProvider 那个端口，不直接依赖本
模块，见下面的分层规矩）。

⚠️ 本模块（及其子包）**不许** import app.agent.*
-----------------------------------------------
`app.agent.prompt.provider` 会惰性 import 本模块来取职责提示词；如果这里
反过来 import app/agent，就绕成了环（而 ring 的表现往往是某次启动顺序一变
就 ImportError）。本模块只依赖 app.common.*，业务无关。

⚠️ 本次刻意**没有**这些东西（不是疏漏，写下来免得下一个人补错方向）
-----------------------------------------------------------
    models.sql / models/sqlalchemy/   没有表。提示词先硬编码在 catalog.py。
    repository/                        没有持久化，也就没有仓储（有仓储就会
                                       把 needs_session 变成 True，为一张
                                       硬编码表去开数据库连接）。
    prompt.yaml（OpenAPI 契约）        没有 HTTP 端点。tool / capability
                                       那边有 yaml，是因为它们有路由。
    initializer.py 里加一行            没有 SQLAlchemy 模型要注册
                                       （见 infra/datasource/initializer.py）。

提示词进库那天要补的就是上面这四项，而**契约与调用方都不用改**：换掉
`application/prompt_service_impl.py` 的实现即可（工厂靠反射发现，见
app/common/introspect.py）。
"""

from __future__ import annotations

from .application import PromptService, PromptServiceImpl
from .catalog import DEFAULT_DUTY, DUTY_PROMPTS

__all__ = [
    "DEFAULT_DUTY",
    "DUTY_PROMPTS",
    "PromptService",
    "PromptServiceImpl",
]
