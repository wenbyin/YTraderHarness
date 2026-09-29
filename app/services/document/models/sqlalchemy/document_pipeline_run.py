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
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base
from app.infra.graph.recorder import RunStatus


class DocumentPipelineRun(Base):
    """
    一次 Document 处理流水线的运行记录。

    与 document_version_retrieval_binding 的分工
    --------------------------------------------
        binding 表        当前业务状态。每 (version, retrieval) 一行。
                          回答"这个版本现在可搜吗"。

        本表              追加型执行历史。每次运行一行，无上界。
                          回答"发生了什么、多久、为什么挂的"。

    单向派生：本表**不**喂业务状态。业务状态由 finalize 节点
    写 binding 表，再从 binding 聚合出来。

    节点兼容性
    ----------
    graph_version / topology_fingerprint / topology_snapshot
    三者都在**运行开始时冻结**。

    于是 2026-09 跑的一次运行，即使今天的图已经加了节点、
    改了边，前端仍能拿到当时的拓扑快照并按它渲染。
    这是"历史运行不因图演进而失真"的主机制。

    thread_id
    ---------
    LangGraph 的断点续跑标识。形如
        docver:<document_version_id>:run:<pipeline_run_id>
    每次运行唯一 —— reprocess 必须开新 thread，否则会从
    上一次的 checkpoint 续跑而不是重新处理。
    """

    __tablename__ = "document_pipeline_run"

    # ========================================================
    # Primary Key
    # ========================================================

    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="PipelineRun 主键。",
    )

    # ========================================================
    # LangGraph Identity
    # ========================================================

    thread_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        comment="LangGraph thread_id，断点续跑靠它。",
    )

    langgraph_run_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment=(
            "LangGraph 根 run_id。构造 config 时传入 pipeline_run_id，"
            "因此通常与主键相同；留空表示该次未启用 tracing。"
        ),
    )

    # ========================================================
    # Graph Topology (frozen at run start)
    # ========================================================

    graph_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="图名称，例如 document_ingestion_pipeline。",
    )

    graph_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="图版本（semver）。任何拓扑变更必须 bump。",
    )

    topology_fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment=(
            "拓扑指纹：sha256(node_id, sequence, edges)。"
            "CI 用它对比签入的 golden 值，防止误改拓扑。"
        ),
    )

    topology_snapshot: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "运行开始时冻结的完整拓扑："
            "{nodes:[{node_id,label,group,sequence}], edges:[[a,b]]}。"
            "前端按它渲染历史运行，不依赖当前代码里的目录。"
        ),
    )

    # ========================================================
    # Trigger
    # ========================================================

    trigger_source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment=(
            "触发来源：create_document / update_document / "
            "reprocess / resume / retry / manual。"
        ),
    )

    triggered_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="触发者用户 ID。",
    )

    parent_pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_pipeline_run.pipeline_run_id",
            ondelete="SET NULL",
        ),
        nullable=True,
        comment="reprocess 链上的上一次运行。",
    )

    root_pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_pipeline_run.pipeline_run_id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
        comment=(
            "扁平化后的链根。一次索引扫描就能取到某个版本的全部历史，"
            "不必递归回溯 parent 链。"
        ),
    )

    # ========================================================
    # Subject
    # ========================================================

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document.document_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
        comment="所属 Document。",
    )

    document_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_version.document_version_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
        comment="处理的 DocumentVersion。",
    )

    document_type_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_type_definition.document_type_definition_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
        comment="该版本所属类型定义（拓扑选择依据）。",
    )

    file_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="本次处理的源文件。",
    )

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="租户 ID。",
    )

    # ========================================================
    # Config
    # ========================================================

    run_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "本次运行的配置快照：analysis_mode / pipeline_config / "
            "retrieval_definition_ids / force 等。"
            "绝不含 DSN、session 或任何密钥。"
        ),
    )

    # ========================================================
    # Execution State
    # ========================================================

    status: Mapped[RunStatus] = mapped_column(
        String(32),
        nullable=False,
        default=RunStatus.PENDING,
        index=True,
        comment=(
            "pending/running/completed/failed/partial/cancelled。"
            "这是观测层状态，与业务的 ProcessingStatus 是两套；"
            "partial 映射回业务状态时按 failed 处理。"
        ),
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="图开始运行的时间。",
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="图结束（成功或失败）的时间。",
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="总耗时毫秒。",
    )

    resumed_from_checkpoint_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="非空表示本次是从 checkpoint 续跑而来。",
    )

    # ========================================================
    # Error
    # ========================================================

    error_type: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="失败异常类型名。",
    )

    error_node_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="失败发生在哪个节点（稳定 node_id）。",
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="失败异常信息。",
    )

    # ========================================================
    # Counters (for list views)
    # ========================================================

    node_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="本次运行落下的节点行数（含各扇出分支）。",
    )

    failed_node_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="其中失败的行数。",
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

    node_runs: Mapped[list["DocumentPipelineNodeRun"]] = relationship(
        back_populates="pipeline_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # 注意：中间产物**没有**对应的表，也没有 relationship。
    # 它落在 FileStorage 里，storage_key 是内容寻址的，
    # 不随 run 行的生命周期走。

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        Index(
            "ix_document_pipeline_run_version_started",
            "document_version_id",
            "started_at",
        ),
        Index(
            "ix_document_pipeline_run_status_started",
            "status",
            "started_at",
        ),
    )
