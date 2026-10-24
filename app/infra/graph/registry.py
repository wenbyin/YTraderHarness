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

    #: 这条 thread 属于哪个**业务会话**（agent_session 的主键）。
    #:
    #: ⚠️ 与 thread_id 是两件事，虽然本仓里它们一对一
    #: --------------------------------------------------
    #: thread_id 是 **langgraph 的内部键**（形如 `dev:react:<uuid>`），
    #: 它按会话稳定是业务侧的安排，不是 infra 的契约。而会话详情页、
    #: 消息、运行列表这些**给人看的东西**全按 session_id 关联。
    #:
    #: 工具执行账本（tool_execution）用后者做维度键的第一段：存
    #: thread_id 的话，每次查账本都要先反查一次会话，而那个映射就在
    #: 业务侧手边（`run.session_id`）。
    #:
    #: 默认 None：只有 document 那条流水线不经过会话，它不关心这个。
    #: agent 那条路必须填（见 agent_run_dispatch）。
    session_id: Any = None

    #: 发起这次运行的调用者的**身份快照**（groups / roles 之类）。
    #:
    #: 为什么由 runner 交进来而不是运行时去查：这条链路是**异步派发**的
    #: （请求提交后才跑），跑起来时手里已经没有 token 了，而身份的唯一
    #: 真相（Casdoor）不在本地。所以业务侧必须在建运行/派发时把快照带
    #: 下来，infra 只负责转交，不解释它的形状。
    #:
    #: 形状由业务侧定义（agent 那边是 build_caller_context 那份 dict）；
    #: 不关心它的图传 None 即可。
    caller_identity: Any = None

    #: 这一轮用哪份**职责**（`agent_run.duty_key`）。
    #:
    #: 为什么它也得由 runner 交进来：图 state 里有一份同名的（`duty_key`，
    #: 建运行那一刻定下、负责注入职责提示词），但**工具分支读不到父状态**
    #: —— `Send` 的 payload 取代 state（见 react/state.py 的
    #: ToolBranchPayload）。而记忆的 scope 是 (agent_id, user|org)，工具要按
    #: 同一个 agent_id 定位，否则"注入里列出的场景"与"工具能取到的场景"
    #: 会是两份不同的数据。
    #:
    #: 与 caller_identity 同一条取法：infra 只负责转交，不解释它的形状。
    duty_key: Any = None

    #: 客户端传入的 Trace 聚合键（None = 未开启 Trace）。
    #:
    #: infra 只负责转交：它不读 Redis、不知道 trace 事件长什么样。
    #: 业务侧（ReactContext / 投影层）据此决定要不要发 trace 事件。
    trace_id: Any = None


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
