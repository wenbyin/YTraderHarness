"""
提示词模块：**所有提示词只有一个取用入口**。

两类提示词
---------
    app/agent/prompt/prompts/   ① 节点功能性：这次**调模型干什么**
                                （classification / summarization / 将来的
                                  memory_extraction、memory_deduplication）
                                键 = model_role，进程级常量，反射发现
    app/services/prompt/        ② 职责功能性：主 agent 搭载的模型
                                **是干什么的**
                                （stock_research / python_programmer…）
                                键 = 职责标识，业务数据（当前硬编码）

两者都从 `get_prompt_provider()` 的同一个端口出去：

    provider.node_prompt(ModelRole.CLASSIFICATION, candidates=...)   同步
    await provider.duty_prompt()                                     异步

（一侧同步一侧异步的理由见 contracts.py —— 是两侧调用点的形状决定的。）

⚠️ 主 agent 的提示词不在这里，而是从"职责"来
------------------------------------------
主循环（`model_role=reasoning`）**没有**节点提示词：它要回答的不是
"这次调用干什么"，而是"我这个模型是干什么的"。所以它在注册表里查不到，
这是设计，不是遗漏 —— 见 DefaultPromptProvider.node_prompt 的报错文案。

主循环的提示词最终以 **`context_policy.system_prompt`** 的形式落地
（`ContextBuilder` 的 system 消息出口，见 processor/context.py 的
`system_message()`），由 builder 节点写进 state —— 也就是说 `agent_node`
自己一行都不用改。

⚠️ 子包必须有 `__init__.py`
-------------------------
`prompts/` 靠 `pkgutil.walk_packages` 反射发现（见 registry.py），而它
**不会走进没有 `__init__.py` 的目录**：删掉那个文件不会报错，所有提示词
会无声消失。registry 那边对"空表"另有一道显式报错，但根因还是在
这里 —— 别删。

用法
----
    from app.agent.prompt import get_prompt_provider

    text = get_prompt_provider().node_prompt(ModelRole.SUMMARIZATION)

组合根在 lifespan 里装带 ServiceFactory 的实现（见 app/main.py 5.2），
脚本与测试可以 `set_prompt_provider(假的)` 整份替换。
"""

from __future__ import annotations

from .contracts import (
    PromptDefinition,
    PromptError,
    PromptNotFoundError,
    PromptProvider,
    PromptRegistryError,
    PromptUnavailableError,
)
from .provider import (
    DefaultPromptProvider,
    get_prompt_provider,
    reset_prompt_provider,
    set_prompt_provider,
)
from .registry import found_prompts, reset_prompt_registry

__all__ = [
    "DefaultPromptProvider",
    "PromptDefinition",
    "PromptError",
    "PromptNotFoundError",
    "PromptProvider",
    "PromptRegistryError",
    "PromptUnavailableError",
    "found_prompts",
    "get_prompt_provider",
    "reset_prompt_provider",
    "reset_prompt_registry",
    "set_prompt_provider",
]
