"""
Domain entities for pipeline observability.

两个实体对应两张表：

    PipelineRun          一次运行的执行历史
    PipelineNodeRun      单个节点单次尝试

中间产物**不在这里** —— 它落在 FileStorage 里，见
pipeline/observability/artifacts.py。

它们都是**追加型观测数据**，不参与业务状态派生：

    binding 表   -> 当前业务状态（这个版本现在可搜吗）
    本模块       -> 执行历史（发生了什么、多久、为什么挂的）

单向派生：finalize 节点写 binding 表，再从 binding 聚合出
DocumentVersion.processing_status。观测实体永远不喂业务状态。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.infra.graph.recorder import (
    NodeRunStatus,
    RunStatus,
)


class _PipelineBase(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
    )


# ============================================================
# PipelineRun
# ============================================================


class PipelineRun(_PipelineBase):
    """一次 Document 处理流水线的运行记录。"""

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    pipeline_run_id: UUID

    thread_id: str = Field(
        description="LangGraph thread_id，断点续跑靠它。",
    )

    langgraph_run_id: str | None = None

    # --------------------------------------------------------
    # Topology (frozen at run start)
    # --------------------------------------------------------

    graph_name: str

    graph_version: str = Field(
        description="图版本。任何拓扑变更必须 bump。",
    )

    topology_fingerprint: str

    topology_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "运行开始时冻结的拓扑。前端据此渲染历史运行，"
            "不受当前代码里的图影响。"
        ),
    )

    # --------------------------------------------------------
    # Trigger
    # --------------------------------------------------------

    trigger_source: str

    triggered_by: UUID | None = None

    parent_pipeline_run_id: UUID | None = None

    root_pipeline_run_id: UUID | None = None

    # --------------------------------------------------------
    # Subject
    # --------------------------------------------------------

    document_id: UUID

    document_version_id: UUID

    document_type_definition_id: UUID

    file_id: UUID | None = None

    tenant_id: UUID | None = None

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    run_config: dict[str, Any] = Field(
        default_factory=dict,
        description="配置快照。绝不含 DSN / session / 密钥。",
    )

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    status: RunStatus = RunStatus.PENDING

    started_at: datetime | None = None

    finished_at: datetime | None = None

    duration_ms: int | None = None

    resumed_from_checkpoint_id: str | None = None

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    error_type: str | None = None

    error_node_id: str | None = None

    error_message: str | None = None

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    node_count: int = 0

    failed_node_count: int = 0

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    created_at: datetime

    updated_at: datetime


# ============================================================
# PipelineNodeRun
# ============================================================


class PipelineNodeRun(_PipelineBase):
    """单个节点单次尝试的执行记录。"""

    pipeline_node_run_id: UUID

    pipeline_run_id: UUID

    # --------------------------------------------------------
    # Node identity (frozen at write time)
    # --------------------------------------------------------

    node_id: str = Field(
        description="稳定 slug。不是下标，一旦发布永不复用。",
    )

    node_label: str = Field(
        description=(
            "节点展示名，写入时冻结。"
            "前端遇到不认识的 node_id 时用它渲染兜底行。"
        ),
    )

    node_group: str

    sequence: int = Field(
        description="拓扑序号，写入时冻结，永不重编号。",
    )

    # --------------------------------------------------------
    # Branch identity
    # --------------------------------------------------------

    document_retrieval_definition_id: UUID | None = None

    branch_key: str = ""

    attempt: int = 1

    # --------------------------------------------------------
    # LangGraph correlation
    # --------------------------------------------------------

    langgraph_task_id: str | None = None

    langgraph_run_id: str | None = None

    parent_run_id: str | None = None

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    status: NodeRunStatus = NodeRunStatus.PENDING

    started_at: datetime | None = None

    finished_at: datetime | None = None

    duration_ms: int | None = None

    # --------------------------------------------------------
    # Payload summaries
    # --------------------------------------------------------

    input_summary: dict[str, Any] | None = None

    output_summary: dict[str, Any] | None = None

    metrics: dict[str, Any] | None = None

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    error_type: str | None = None

    error_message: str | None = None

    error_detail: dict[str, Any] | None = None

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    created_at: datetime

    updated_at: datetime
