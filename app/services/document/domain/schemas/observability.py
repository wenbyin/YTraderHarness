"""
流水线可观测性的对外契约。

对应 app/api/v1/pipeline_runs.py 暴露的接口。

与 runtime.py 的分工
--------------------
    runtime.py        业务视图：这个文档是什么、有哪些版本
    observability.py  执行视图：这个版本处理得怎么样、卡在哪、多久

前端的处理进度页与运行详情页消费本模块。

节点兼容性契约（前端必读）
--------------------------
图会演进：节点会被加、被删、被改名。为了让历史运行
永远能正确渲染，约定如下：

1. **详情接口返回该次运行自己的 topology_snapshot**，
   而不是当前图的拓扑。按它渲染，不要用你本地写死的节点列表。

2. **node_id 是稳定键**。展示名走 node_label，
   它是写入时冻结的，即便该节点今天已经不存在也仍然正确。

3. **遇到不认识的 node_id 不要丢弃这一行** ——
   用 node_label 渲染一行通用节点，按 sequence 排序。
   详情接口的 unknown_nodes 为 true 时，
   提示"此运行使用旧版流水线"。

4. **不要假设 sequence 连续或从 0 开始**。
   它是冻结的历史值，新节点只会往后追加序号。

节点目录接口（GET /document-pipeline-nodes）给出当前图版本
的完整节点表，用于把 node_id 翻译成人类可读的名称与说明。
退役节点会永久保留并标 deprecated=true。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from app.common.schemas import APIModel

from app.infra.graph.recorder import (
    NodeRunStatus,
    RunStatus,
)

from .enum import ProcessingStatus


# ============================================================
# Node run
# ============================================================


class PipelineNodeRunResponse(APIModel):
    """
    单个节点单次尝试。

    一次运行的节点行数不是固定的：扇出阶段每个
    RetrievalDefinition 各产生一行，因此总数是

        线性节点数 + 控制节点数 + 扇出节点数 × N

    error_detail 不在此暴露，避免把内部结构泄给前端；
    需要时另行提供诊断接口。
    """

    pipeline_node_run_id: UUID

    pipeline_run_id: UUID

    node_id: str = Field(
        description="稳定键。用它去节点目录查展示信息。",
    )

    node_label: str = Field(
        description="展示名，写入时冻结。目录里查不到 node_id 时用它兜底。",
    )

    node_group: str = Field(
        description="linear / fanout / control。",
    )

    sequence: int = Field(
        description="排序依据。冻结的历史值，不要假设连续。",
    )

    document_retrieval_definition_id: UUID | None = Field(
        default=None,
        description="扇出节点所属的 RetrievalDefinition。",
    )

    branch_key: str = Field(
        default="",
        description=(
            "分支键：线性/控制节点为空串，扇出为 retrieval id。"
            "同一节点下用它区分不同分支。"
        ),
    )

    attempt: int = Field(
        default=1,
        description="重试计数，1 起。大于 1 说明之前失败过。",
    )

    status: NodeRunStatus

    started_at: datetime | None = None

    finished_at: datetime | None = None

    duration_ms: int | None = None

    input_summary: dict[str, Any] | None = None

    output_summary: dict[str, Any] | None = None

    metrics: dict[str, Any] | None = Field(
        default=None,
        description=(
            "业务指标：chunk_count / token_count / "
            "embedding_dimensions / vector_count 等。"
        ),
    )

    error_type: str | None = None

    error_message: str | None = None


# ============================================================
# Run
# ============================================================


class PipelineRunResponse(APIModel):
    """运行记录。列表视图用，不含拓扑快照与节点明细。"""

    pipeline_run_id: UUID

    document_id: UUID

    document_version_id: UUID

    document_type_definition_id: UUID

    trigger_source: str = Field(
        description=(
            "create_document / update_document / reprocess / "
            "resume / retry / manual。"
        ),
    )

    triggered_by: UUID | None = None

    parent_pipeline_run_id: UUID | None = Field(
        default=None,
        description="reprocess 链上的上一次运行。",
    )

    status: RunStatus = Field(
        description=(
            "观测层状态。注意 partial 不在业务的 ProcessingStatus 里："
            "它表示多路 retrieval 部分成功，映射回业务状态时按 failed 算。"
        ),
    )

    started_at: datetime | None = None

    finished_at: datetime | None = None

    duration_ms: int | None = None

    resumed_from_checkpoint_id: str | None = Field(
        default=None,
        description="非空表示本次是从 checkpoint 续跑而来。",
    )

    error_type: str | None = None

    error_node_id: str | None = Field(
        default=None,
        description="失败发生在哪个节点，配合节点目录定位。",
    )

    error_message: str | None = None

    node_count: int = 0

    failed_node_count: int = 0

    created_at: datetime


class PipelineRunDetailResponse(PipelineRunResponse):
    """
    运行详情。

    topology_snapshot 是**该次运行自己的**拓扑，前端应按它渲染，
    不要用当前图的节点列表 —— 这是历史运行不失真的主机制。
    """

    graph_name: str

    graph_version: str

    topology_fingerprint: str

    topology_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "{nodes:[{node_id,label,group,sequence}], edges:[[from,to]]}。"
            "冻结于运行开始时。"
        ),
    )

    run_config: dict[str, Any] = Field(
        default_factory=dict,
        description="本次运行的配置快照。",
    )

    nodes: list[PipelineNodeRunResponse] = Field(
        default_factory=list,
    )

    unknown_nodes: bool = Field(
        default=False,
        description=(
            "该次运行的拓扑里有节点不在当前目录中。"
            "为 true 时前端应提示「此运行使用旧版流水线」，"
            "并依靠 node_label + sequence 渲染。"
        ),
    )


class PipelineRunListResponse(APIModel):
    """分页的运行列表。"""

    items: list[PipelineRunResponse] = Field(
        default_factory=list,
    )

    page: int

    page_size: int

    total: int

    total_pages: int


class PipelineNodeRunListResponse(APIModel):
    """分页的节点行列表，深扇出时单独翻页用。"""

    items: list[PipelineNodeRunResponse] = Field(
        default_factory=list,
    )

    page: int

    page_size: int

    total: int

    total_pages: int


# ============================================================
# Node catalog
# ============================================================


class PipelineNodeCatalogEntry(APIModel):
    """
    节点目录中的一项。

    目录是 node_id 的**契约表**：退役节点永久保留，
    标 deprecated=true 并给 replaced_by，
    这样历史运行里的旧 node_id 永远能查到解释。
    """

    node_id: str

    label: str

    group: str = Field(
        description="linear / fanout / control。",
    )

    sequence: int

    description: str = ""

    deprecated: bool = Field(
        default=False,
        description="该节点已不在当前图中，但历史运行可能仍有它的行。",
    )

    replaced_by: str | None = Field(
        default=None,
        description="若已被替代，给出替代节点的 node_id。",
    )


class PipelineNodeCatalogResponse(APIModel):
    """当前图版本的节点目录。"""

    graph_name: str

    graph_version: str

    topology_fingerprint: str

    nodes: list[PipelineNodeCatalogEntry] = Field(
        default_factory=list,
    )


# ============================================================
# Reprocess
# ============================================================


class ReprocessDocumentVersionRequest(APIModel):
    """
    重新处理某个 DocumentVersion。

    严格对齐 document_type_definition.yaml 的声明。

    document_processor_definition_ids 的语义（已确认）：

        空数组 / 不传
            按现有规则重算**全部**适用的 RetrievalDefinition。

        非空
            先用这些 processor definition 过一遍
            ProcessorRetrievalBinding，只重跑映射到的 retrieval。

    force
        true 时绕过"已经 completed 就跳过"的短路，强制重跑。
    """

    document_processor_definition_ids: list[UUID] = Field(
        default_factory=list,
        description=(
            "留空表示重跑全部适用的 retrieval；"
            "非空表示只重跑这些 processor 映射到的 retrieval。"
        ),
    )

    force: bool = Field(
        default=False,
        description="绕过已完成即跳过的短路。",
    )

    checkpoint_level: Literal["node", "run"] | None = Field(
        default=None,
        description=(
            "本次运行的 checkpoint 级别。留空用服务端默认（node）。"
            "node 每个节点落一次盘、可断点续跑；"
            "run 只在结束时落一次、写入最少但**崩溃后无法续跑**。"
            "级别必须在运行开始前定 —— 事后改救不回来。"
        ),
    )


class ReprocessDocumentVersionResponse(APIModel):
    """重新处理受理结果（HTTP 202）。"""

    document_version_id: UUID

    status: ProcessingStatus = Field(
        description="受理时该版本的聚合处理状态。",
    )

    job_id: str | None = Field(
        default=None,
        description=(
            "本次运行的 pipeline_run_id。"
            "用它去 /pipeline-runs/{id} 查进度。"
        ),
    )

    message: str = ""


# ============================================================
# 中间产物：刻意没有对外接口
# ============================================================
#
# 产物（解析后的正文等）落在 FileStorage 里，没有对应的库表，
# 也**不提供"取全文"的接口**。
#
# 前端需要的预览已经随节点行一起返回了：
# PipelineNodeRunResponse.output_summary 里的
#   {"ref": {"checksum": ..., "char_count": ..., "preview": "..."}}
# 其中 preview 有 512 字符上限。
#
# 不额外开一个取全文的端点，理由有两个：
#   1. 观测接口不应该变成整篇文档的导出通道
#   2. 真需要全文有文档内容接口，职责分开
#
# 若将来确实要在处理中途看正文，再加一个带权限控制的
# 独立端点，而不是挂在观测 API 上。
