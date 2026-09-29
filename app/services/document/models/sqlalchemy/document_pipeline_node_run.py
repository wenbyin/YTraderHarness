from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base
from app.infra.graph.recorder import NodeRunStatus


class DocumentPipelineNodeRun(Base):
    """
    单个节点单次尝试的执行记录。

    幂等的承重墙
    ------------
        UNIQUE(pipeline_run_id, node_id, branch_key, attempt)

    两个写入方都 upsert 到它：

        instrumentation.node_span    主写入方（节点内，带真实 attempt）
        callbacks.PersistingCallback  兜底网（只能捕获节点体之外的失败）

    重复事件因此天然幂等 —— 后到的只是覆盖同样的字段。
    只要任一机制触发，记录就是完整的。

    branch_key 为什么是空串而不是 NULL
    ---------------------------------
    线性阶段与控制节点没有分支身份，用空串表示。
    不能用 NULL：Postgres 的唯一约束不约束 NULL
    （多个 NULL 互不冲突），那样"线性节点重复写入"
    这条幂等保证就失效了。

    字段为什么在写入时冻结
    ----------------------
    node_label / node_group / sequence 落库时写死，
    不从当前代码里的 NODE_SPECS 反查。

    这样即使后来节点改名、分组调整、甚至整个节点被删除，
    历史行仍然带着当时的展示信息，前端有东西可画 ——
    这是"节点兼容性"的第三道兜底。

    ⚠️ 可空 JSONB 列必须带 none_as_null=True
    ----------------------------------------
    upsert 用 COALESCE(excluded.X, existing.X) 保证"后到的
    空值不覆盖先写的真值"（callback 不带 metrics，会在线性节点
    上后写一次）。

    但 JSONB 默认把 Python None 写成 **JSON 的 'null' 字面量**，
    而它不是 SQL NULL —— COALESCE 会认为"有新值"，把真数据覆盖掉。

    症状很有迷惑性：所有**线性**节点的 metrics / output_summary
    全空，而扇出节点正常（因为扇出不走 callback，没有第二次写入）。
    实测踩过，排查了很久才定位到是 JSON null 而非 SQL NULL。
    """

    __tablename__ = "document_pipeline_node_run"

    # ========================================================
    # Primary Key
    # ========================================================

    pipeline_node_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="PipelineNodeRun 主键。",
    )

    # ========================================================
    # Owning Run
    # ========================================================

    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_pipeline_run.pipeline_run_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
        comment="所属 PipelineRun。",
    )

    # ========================================================
    # Node Identity (frozen at write time)
    # ========================================================

    node_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment=(
            "稳定 slug，例如 parsing / chunking / finalize_version。"
            "不是下标，不是 str(i)。一旦发布永不复用。"
        ),
    )

    node_label: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment=(
            "节点展示名，写入时冻结。"
            "前端遇到不认识的 node_id 时用它渲染兜底行。"
        ),
    )

    node_group: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="linear / fanout / control（也含 callback）。",
    )

    sequence: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment=(
            "拓扑中的序号，写入时冻结。永不重编号。"
            "前端排序与进度展示依据。"
        ),
    )

    # ========================================================
    # Branch Identity
    # ========================================================

    document_retrieval_definition_id: Mapped[
        uuid.UUID | None
    ] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="扇出节点所属的 RetrievalDefinition；线性/控制节点为空。",
    )

    branch_key: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="",
        server_default="",
        comment=(
            "分支键：线性/控制为 ''，扇出为 retrieval definition id。"
            "空串而非 NULL，见类 docstring。"
        ),
    )

    attempt: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
        comment=(
            "重试计数，1 起。"
            "权威来源是 ExecutionInfo.node_attempt，只有节点内部能拿到。"
        ),
    )

    # ========================================================
    # LangGraph Correlation
    # ========================================================

    langgraph_task_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="LangGraph task_id，与 callback 事件的 join key。",
    )

    langgraph_run_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="该节点执行的 run_id。",
    )

    parent_run_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="父 run_id，用于还原调用链。",
    )

    # ========================================================
    # Execution State
    # ========================================================

    status: Mapped[NodeRunStatus] = mapped_column(
        String(32),
        nullable=False,
        default=NodeRunStatus.PENDING,
        index=True,
        comment="pending/running/completed/failed/skipped。",
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="节点开始执行时间。",
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="节点结束时间。",
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="节点耗时毫秒。",
    )

    # ========================================================
    # Payload Summaries
    # ========================================================

    input_summary: Mapped[dict | None] = mapped_column(
        JSONB(none_as_null=True),  # 见类 docstring：必须，否则 COALESCE 守卫失效
        nullable=True,
        comment=(
            "输入摘要。只放 id、checksum 和计数，"
            "例如 {clean_ref:{checksum,char_count}}。绝不放正文。"
        ),
    )

    output_summary: Mapped[dict | None] = mapped_column(
        JSONB(none_as_null=True),  # 见类 docstring：必须，否则 COALESCE 守卫失效
        nullable=True,
        comment="输出摘要。同样只放引用与计数。",
    )

    metrics: Mapped[dict | None] = mapped_column(
        JSONB(none_as_null=True),  # 见类 docstring：必须，否则 COALESCE 守卫失效
        nullable=True,
        comment=(
            "业务指标：chunk_count / token_count / "
            "embedding_dimensions / vector_count / upserted 等。"
            "callback 拿不到这些，只有节点内部能提供。"
        ),
    )

    # ========================================================
    # Error
    # ========================================================

    error_type: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="异常类型名。",
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="异常信息。",
    )

    error_detail: Mapped[dict | None] = mapped_column(
        JSONB(none_as_null=True),  # 见类 docstring：必须，否则 COALESCE 守卫失效
        nullable=True,
        comment="结构化错误上下文，例如失败的 index_identifier。",
    )

    # ========================================================
    # Audit
    # ========================================================

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="最后更新时间。",
    )

    # ========================================================
    # Relationships
    # ========================================================

    pipeline_run: Mapped["DocumentPipelineRun"] = relationship(
        back_populates="node_runs",
    )

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "pipeline_run_id",
            "node_id",
            "branch_key",
            "attempt",
            name="uq_document_pipeline_node_run_identity",
        ),
        Index(
            "ix_document_pipeline_node_run_order",
            "pipeline_run_id",
            "sequence",
            "started_at",
        ),
    )
