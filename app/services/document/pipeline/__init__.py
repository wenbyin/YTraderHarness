"""
Document 处理流水线（LangGraph 编排）。

    state.py       图状态与 Send payload 结构
    context.py     运行时依赖（context_schema）
    catalog.py     节点目录：观测契约的来源
    selection.py   纯函数：选哪些 RetrievalDefinition 参与
    fanout.py      扇出路由（含必要的"路由屏障"机制）
    graph.py       拓扑装配
    definition.py  注册进通用 GraphRegistry
    registry.py    能力注册表：名字 -> 实现
    nodes/         10 个阶段 + 1 个控制节点
    observability/ 产物存储、观测写入、读模型

通用的部分（checkpointer、埋点、runner、dispatcher）在
app/infra/graph/ 下，本包只贡献这个图特有的东西。
"""

from .catalog import (
    GRAPH_NAME,
    GRAPH_VERSION,
    NODE_SPECS,
)
from .context import DocumentPipelineContext
from .definition import build_graph_definition
from .graph import build_document_pipeline_graph
from .registry import (
    CapabilityNotFound,
    PipelineRegistry,
)
from .selection import (
    RetrievalCandidate,
    select_retrievals,
)
from .state import (
    ArtifactRef,
    BranchInput,
    DocumentPipelineState,
    RetrievalBranch,
    RetrievalResult,
)

__all__ = [
    "GRAPH_NAME",
    "GRAPH_VERSION",
    "NODE_SPECS",
    "ArtifactRef",
    "BranchInput",
    "CapabilityNotFound",
    "DocumentPipelineContext",
    "DocumentPipelineState",
    "PipelineRegistry",
    "RetrievalBranch",
    "RetrievalCandidate",
    "RetrievalResult",
    "build_document_pipeline_graph",
    "build_graph_definition",
    "select_retrievals",
]
