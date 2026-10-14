"""
外部上下文源（规格 §三、§十七、§二十二的边界）。

`context_builder.yaml` 把它们列为 `external_dependencies`：

    Memory Service / Document Service / Profile Service / Capability Selector

**它们的实现不属于 Context Builder** —— 本模块只定义"怎么问它们"。
默认全部是 None（= 一个都不问），此时 Processor 用 State 里已经有的
数据（Agent Builder / 前置节点注入的）。

⚠️ 为什么默认是 None 而不是"内置一个查库的实现"
-------------------------------------------
    ContextBuilder 不负责: Conversation CRUD / 用户消息持久化 /
                          Memory 存储 / 检索实现（规格 §二十二）

本仓库目前**没有**会话库、画像库、记忆库（`app/services/user` 与
`app/services/common` 是空目录，消息唯一的事实来源是 LangGraph 的
checkpointer + `ReactState.messages`）。因此这里给端口、不给实现：
真正接服务时只写一个类，ContextBuilder 与 Processor 一行都不用改。

⚠️ `ConversationSource` 已经删掉
----------------------------
它曾经是"比 Graph State 带得更多的历史"那个口子 —— §四B 的红线（后续
构建不许回库重放）就写在它的 docstring 里。删它的理由与 `HistoryRetriever`
相同：从落地起就没有过生产实现，而更早的内容现在归**记忆模块的 L0 层**
（见 processors/history.py 顶部的说明）。

⚠️ 那条不变量本身**没有消失**，只是换了落点
---------------------------------------
§四B 现在由记忆模块守着：一个运行只召回一次，后续迭代复用保活的分节
（processors/memory.py 的 `_replay`）。所以本模块少一个口子，不等于少了
一层约束。

⚠️ 源与 State 的优先级（唯一规则）
-------------------------------
    配了源 -> **源说了算**（即使它返回空 —— 空就是"没有"）
    没配源 -> 用 State 里绑定的值

反向的写法（"源返回空就回落到 State 里的旧值"）会让一次真实的
"记忆服务说没有"变成"用上次那份陈旧记忆"，而症状是模型看到一个
过期的事实。规则只有一条，就没有这种歧义。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "KnowledgeSource",
    "MemorySource",
    "ProfileSource",
    "ToolExposureSource",
]


@runtime_checkable
class ProfileSource(Protocol):
    """用户画像源（租户 / 偏好 / 权限摘要…）。"""

    async def load_profile(
            self,
            *,
            user_id: str,
    ) -> Mapping[str, Any] | None:
        ...


@runtime_checkable
class MemorySource(Protocol):
    """
    长期记忆检索源。

    ⚠️ 它失败是**可选的**（规格 §二十八）：Memory 拿不到不该让整轮
    对话失败。因此 MemoryProcessor 声明 `spec.optional=True`。
    """

    async def retrieve(
            self,
            *,
            user_id: str,
            query: str | None,
            limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        ...


@runtime_checkable
class KnowledgeSource(Protocol):
    """
    知识库 / RAG 源。

    注意它**不是**检索实现：检索属于 Document 模块（embedder /
    index / retrieve 那一整套）。这里只是"把检索结果交给我"的入口。
    """

    async def retrieve(
            self,
            *,
            query: str,
            conversation_id: str | None,
            limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        ...


@runtime_checkable
class ToolExposureSource(Protocol):
    """
    工具暴露源：按 key 解析出**可绑定的工具对象**。

    ⚠️ 它只负责"这次给模型看哪些工具"，**不**负责权限判定与执行
    （规格 §二十二）。

    默认 None：此时 CapabilityProcessor 直接用 State 里的
    `exposed_tools`（由 Capability Selector / Agent Builder 注入）。
    """

    async def resolve(
            self,
            *,
            tool_keys: Sequence[str],
            conversation_id: str | None,
            overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Sequence[Any]:
        """
        tool_keys -> 本次要 bind 给模型的工具对象。

        overrides 是**人工写的说明覆盖**（`{tool_key: {...}}`，见
        `ToolQueryService.tool_documentation_overrides`），由调用方
        从上下文里取好传进来。这一层**自己不查库** —— 它是每次模型
        调用都会跑的一层，而覆盖是慢变数据：读一次、传进来、用几次。

        可选：不传就是"没有人工覆盖"，工具说明全部回落到代码侧事实，
        与这个端口出现之前的行为一模一样。
        """

        ...
