"""
图注册表与拓扑描述。

一个图要能被通用 runner 跑起来，需要四样东西：

    已编译的图        跑什么
    节点目录          观测表的稳定键（node_id / label / group / sequence）
    拓扑指纹与快照    节点兼容性（历史运行按当时的拓扑渲染）
    context 工厂      每次运行注入哪些依赖

把它们打包成 GraphDefinition 注册进来，runner 就不需要知道
任何业务细节。新增一个 pipeline = 新增一次 register 调用，
不需要新的 runner、新的 dispatcher、新的 service。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from .ids import topology_fingerprint

__all__ = [
    "GraphDefinition",
    "GraphRegistry",
    "NodeSpec",
    "build_topology_snapshot",
]


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """
    节点目录中的一项。

    node_id 是**稳定契约键**：
        - 不是下标、不是 str(i)
        - 一旦发布永不复用
        - sequence 永不重编号，新节点只往后追加

    label 会随节点行一起落库并冻结，因此它既是当前的展示名，
    也是历史行的兜底展示名 —— 即便该节点今天已从图里删掉。
    """

    node_id: str
    label: str
    group: str
    sequence: int
    description: str = ""

    #: 已从当前图中移除、但历史运行可能仍有其行的节点。
    #: 目录里永久保留它们，前端据此解释旧 node_id。
    deprecated: bool = False

    #: 若已被替代，给出替代节点的 node_id。
    replaced_by: str | None = None


def build_topology_snapshot(
        node_specs: tuple[NodeSpec, ...],
        edges: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    """
    构造写进 run 行的拓扑快照。

    运行开始时冻结。前端按**该次运行自己的**快照渲染，
    因此图演进后历史运行不会失真 —— 这是节点兼容性的主机制。
    """

    return {
        "nodes": [
            {
                "node_id": spec.node_id,
                "label": spec.label,
                "group": spec.group,
                "sequence": spec.sequence,
            }
            for spec in sorted(
                node_specs, key=lambda s: s.sequence
            )
        ],
        "edges": [list(edge) for edge in edges],
    }


@dataclass(frozen=True, slots=True)
class GraphDefinition:
    """
    一个可运行的图及其全部元数据。

    context_factory 的入参是 RunContext（本次运行的身份），
    返回该图的 context_schema 实例 —— 每次运行都要新建一个，
    因为它内含本次运行独有的 recorder 与 pipeline_run_id。
    """

    name: str
    version: str
    compiled: Any
    node_specs: tuple[NodeSpec, ...]
    edges: tuple[tuple[str, str], ...]
    context_factory: Callable[["RunContext"], Any]

    #: 交给 callback 兜底覆盖的节点。
    #: 通常是线性 + 控制节点；扇出节点由 node_span 逐分支负责，
    #: 详见 callbacks.PersistingCallbackHandler 的说明。
    callback_node_ids: frozenset[str] = field(
        default_factory=frozenset
    )

    #: 不在当前图中、但需要保留在目录里的历史节点。
    retired_node_specs: tuple[NodeSpec, ...] = ()

    # --------------------------------------------------------

    @property
    def topology_fingerprint(self) -> str:
        return topology_fingerprint(
            ((s.node_id, s.sequence) for s in self.node_specs),
            self.edges,
        )

    @property
    def topology_snapshot(self) -> dict[str, Any]:
        return build_topology_snapshot(
            self.node_specs, self.edges
        )

    @property
    def node_ids(self) -> frozenset[str]:
        return frozenset(s.node_id for s in self.node_specs)

    def spec(self, node_id: str) -> NodeSpec | None:
        for candidate in self.node_specs:
            if candidate.node_id == node_id:
                return candidate
        return None

    @property
    def catalog(self) -> tuple[NodeSpec, ...]:
        """当前节点 + 退役节点，供目录接口使用。"""
        return self.node_specs + self.retired_node_specs


@dataclass(frozen=True, slots=True)
class RunContext:
    """
    一次运行的身份。

    由 runner 构造后交给 GraphDefinition.context_factory，
    业务侧据此拼出自己的 context（例如加上 session_factory、
    artifact_store、capability registry）。
    """

    pipeline_run_id: Any
    thread_id: str
    trigger_source: str

    recorder: Any


@runtime_checkable
class _Compiled(Protocol):
    """仅用于类型提示，避免 infra 依赖具体的 langgraph 类型。"""


class GraphRegistry:
    """
    名称 -> GraphDefinition。

    全应用一个实例，挂在 app.state 上。
    """

    def __init__(self) -> None:
        self._definitions: dict[str, GraphDefinition] = {}

    def register(
            self,
            definition: GraphDefinition,
    ) -> None:

        if definition.name in self._definitions:
            raise ValueError(
                "Graph already registered: "
                f"{definition.name}"
            )

        self._definitions[definition.name] = definition

    def get(self, name: str) -> GraphDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown graph: {name!r}. "
                f"Registered: {sorted(self._definitions)}"
            ) from exc

    def get_optional(
            self,
            name: str,
    ) -> GraphDefinition | None:
        return self._definitions.get(name)

    def names(self) -> list[str]:
        return sorted(self._definitions)

    def __contains__(self, name: object) -> bool:
        return name in self._definitions
