"""
把本图包装成 `app/infra/graph` 的 GraphDefinition。

这是业务模块与通用运行时之间的**唯一接缝**：有了它，新增一张图只需
要"定义自己的 state / context / catalog / nodes / build_graph，调一次
build_graph_definition() 并 registry.register()" —— 不需要新的 runner、
新的 dispatcher、新的应用服务。

与 document 的 definition.py 是同一份形状
-----------------------------------------
刻意**不 import 它**。两个图要能各自演进：一旦共用，给 agent 图加一个
字段就会动到文档流水线，而"改一处影响另一处"正是这类共享包装函数
最容易积累的问题。重复的是十几行样板，换来的是两张图互不相干。
"""

from __future__ import annotations

from typing import Any, Callable

from app.infra.graph import GraphDefinition, RunContext

from .catalog import (
    CALLBACK_NODE_IDS,
    EDGES,
    GRAPH_NAME,
    GRAPH_VERSION,
    NODE_SPECS,
    RETIRED_NODE_SPECS,
)

__all__ = ["build_graph_definition"]


def build_graph_definition(
        *,
        compiled: Any,
        context_factory: Callable[[RunContext], Any],
) -> GraphDefinition:
    """
    组装本图的 GraphDefinition。

    Parameters
    ----------
    compiled
        已用某个 checkpointer 编译好的图。用哪个 checkpointer
        由组合根决定 —— 生产用 Postgres，冒烟测试用 InMemory，
        图本身不关心。
    context_factory
        接收 RunContext（本次运行的身份 + recorder），
        返回 ReactContext。
    """

    return GraphDefinition(
        name=GRAPH_NAME,
        version=GRAPH_VERSION,
        compiled=compiled,
        node_specs=NODE_SPECS,
        edges=EDGES,
        context_factory=context_factory,
        callback_node_ids=CALLBACK_NODE_IDS,
        retired_node_specs=RETIRED_NODE_SPECS,
    )
