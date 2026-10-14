"""
Planner 节点：**占位**。

现在的行为：发一条观测，什么都不做，直接放行到 agent。

为什么先建一个什么都不做的节点
------------------------------
路由里已经有 `plan -> planner` 这条分支了（Agent Builder 会把需要先规划
的输入判到那儿）。分支指向一个不存在的节点，LangGraph 在装配期就会报
`Unknown branch` —— 于是"分类能判出 plan"这件事根本没法验证。一个空节点
让拓扑先闭合，分类与路由可以独立于规划的实现被验证。

为什么是"放行到 agent"而不是"接到 END"
------------------------------------
判到 plan 说明这件事**需要被做**，只是规划器还没写。接到 END 会让这类
输入得到一句空答复（而且观测里记成 completed）；放行到 agent 至少让它
按正常主循环被处理掉 —— 也就是这次改动之前的行为。

⚠️ 落地时要注意的事（留给实现者）
--------------------------------
1. `catalog.py` 里本节点的 `NodeSpec.sequence` 是**追加**的值（不是执行
   顺序），落地改动拓扑时别忘了抬 `GRAPH_VERSION`。
2. planner 要跑一次独立的模型调用 —— 那是**第三种** Context Builder
   类型（`ContextBuilderType.PLANNER`，其抽象基类已在
   builder/base.py 里，尚无具体实现），并且会需要 ReactContext 上
   再挂一个 builder 实例，与 `classifier_builder` 同样的形状。
3. 它产出的计划需要一个 state 通道来承载 —— 先在 graph/react/state.py
   里声明，未声明的键会被 LangGraph **静默丢弃**。
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.infra.graph import node_span

from .catalog import NODE_SPECS
from .context import ReactContext
from .state import ReactState

__all__ = ["planner_node"]

_SPEC = next(s for s in NODE_SPECS if s.node_id == "planner")


async def planner_node(
        state: ReactState,
        runtime: Runtime[ReactContext],
) -> dict[str, Any]:
    """
    占位：不产出任何 state，只落一条观测。

    Returns:
        {} —— 空更新。LangGraph 接受它，于是两条边上什么都不发生。
        将来产出计划时，这里会返回 {"plan": ...}（并记得先在
        ReactState 里声明那个通道）。
    """

    async with node_span(
            runtime,
            node_id=_SPEC.node_id,
            label=_SPEC.label,
            group=_SPEC.group,
            sequence=_SPEC.sequence,
            input_summary={
                "message_count": len(state.get("messages") or ()),
                "planned": False,
            },
    ) as span:

        # 让"这里什么都没做"在观测里是**显式**的，而不是靠人记得
        # 这个节点是占位。前端与排查都能直接看到。
        span.set_output_summary({"placeholder": True})

        return {}
