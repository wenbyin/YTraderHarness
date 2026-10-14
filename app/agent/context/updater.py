"""
Context Updater：模型调用**之后**的上下文回写。

⚠️ 本次只保留接口，没有实现，也没有默认实现
-------------------------------------------
后置处理的形状**还没有定**，所以这里刻意只留下最小的一个口子：
一次调用结束、拿到 AIMessage 之后，允许有一个东西知道
"刚才是拿什么问的（request）、模型答了什么（response）"。

它将来要承载的事情大致是：把这一轮写回历史、抽取出值得进长期记忆
的东西、记 token 用量、决定要不要触发摘要。这些**都还没设计**，
因此本模块不替它们猜形状。

为什么是 Protocol 而不是 Base*
-------------------------------
仓库用 `Protocol` 表达"**被消费的端口**"（`GraphRunRecorder`、
`GraphRunContext`），用 `Base*` 表达"**被反射发现的子类家族**"
（`BaseToolProcessor`、`BaseContextBuilder`）。

Updater 是前者：由一次模型调用**消费**，不是被注册表枚举的一族。
名字因此叫 `ContextUpdater`，不带 `Base` 前缀。

为什么 update() 返回 None
-------------------------
返回 `None` 是**承重的决定**，不是偷懒：

    返回一个 `ContextUpdate(extra_state=...)` 会让 updater 有能力
    往图 state 里写**未声明的 channel**，而 LangGraph 对未声明的键
    是**静默丢弃**的 —— 写了、没生效、没有任何报错。这正是
    document/pipeline/state.py 用一整段 docstring 记录过的坑。

而"返回 None"将来要扩成"返回一个东西"是**零成本**的：Python 不强制
返回类型，现有实现的 `return None` 自动兼容。反过来（先给一个
猜出来的结构、再改）要动每一个实现。所以小的那个是安全的那个。

⚠️ 为什么没有 NoopContextUpdater
--------------------------------
一个"什么都不做"的实现也是**具体实现**，而本次的约束是零具体
实现。更重要的是：它会让人以为"后置处理已经接上了"，而实际上
**后置处理还不存在**这件事正是需要被看见的。

诚实的表达是"没有"：`ReactContext.context_updater` 是一个可空槽，
为 None 就是不回写（见 graph/react/context.py）。等真正实现了，
它再变成一个注册表，这个槽就是注入点。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage

    from .builder.contracts import ContextBuildRequest

__all__ = ["ContextUpdater"]


@runtime_checkable
class ContextUpdater(Protocol):
    """
    模型返回之后，按需要回写上下文。

    实现在这里只收两样东西，因为**那一刻只有这两样是确定的**：

        request    刚才问的是什么（含 thread_id / run_id，
                   实现靠它找到要写回哪个会话）
        response   模型答了什么

    刻意不给它 runtime / state / builder 实例：一旦给了，实现就会
    开始依赖"当前这次运行"的细节，而后置处理该不该看图 state 正是
    还没想清楚的那个问题。**接口越小，越不容易设计错。**

    实现约定
    --------
        不得抛出异常来表达"没什么可写的" —— 那就是 return None。
        失败应当抛，因为它意味着这一轮的上下文状态与实际不一致；
        静默吞掉会让下一轮读到错的上下文。

    ⚠️ 实现必须是**幂等**的：图会重试（见 graph/react/agent_node.py
    关于重试的说明），同一个 response 可能被交过来两次。
    """

    async def update(
            self,
            *,
            request: "ContextBuildRequest",
            response: "AIMessage",
    ) -> None:
        """按需回写。没有可写的就什么都不做。"""

        ...
