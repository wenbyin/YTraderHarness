"""
把图装配进通用运行时 —— 业务侧与 infra 之间的接缝。

两件事：

    register_document_pipeline(...)   建图、包装成 GraphDefinition、
                                      注册进 GraphRegistry

    build_dispatch_handler(...)       产出一个 dispatcher handler：
                                      收到 DispatchEnvelope 后
                                      建 RunSnapshot、跑图、写终态

main.py 的 lifespan 各调一次，就完成了全部接线。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID, uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from app.infra.graph import (
    DispatchEnvelope,
    GraphDefinition,
    GraphRegistry,
    GraphRunRecorder,
    GraphRunner,
    RunRequest,
    RunSnapshot,
)

from .catalog import GRAPH_NAME, GRAPH_VERSION
from .context import DocumentPipelineContext
from .definition import build_graph_definition
from .graph import build_document_pipeline_graph
from .observability.artifacts import ArtifactStore
from .registry import PipelineRegistry
from .settings import PipelineSettings
from .state import DocumentPipelineState

logger = logging.getLogger(__name__)

__all__ = [
    "SUBJECT_TYPE",
    "build_dispatch_handler",
    "register_document_pipeline",
]


#: 本流水线的主体类型。infra 用它区分不同业务的运行，
#: 仓储据此断言并映射到 document 语义的列。
SUBJECT_TYPE = "document_version"


# ============================================================
# 1. 建图 + 注册
# ============================================================


def register_document_pipeline(
        *,
        registry: GraphRegistry,
        checkpointer: BaseCheckpointSaver,
        recorder: GraphRunRecorder,
        artifact_store: ArtifactStore,
        capabilities: PipelineRegistry,
        session_factory: async_sessionmaker[AsyncSession],
        settings: PipelineSettings | None = None,
) -> GraphDefinition:
    """
    编译图、包装成 GraphDefinition、注册进注册表。

    返回注册好的 definition，供调用方（通常是 dispatcher 的
    handler 构造）复用。

    checkpointer 由调用方传入而不是在这里从 provider 取 ——
    这样本函数不关心用哪种 checkpointer（生产 Postgres、
    冒烟 InMemory），也更容易在测试里替换。
    """

    settings = settings or PipelineSettings()

    state_graph: StateGraph = build_document_pipeline_graph()

    compiled = state_graph.compile(checkpointer=checkpointer)

    def context_factory(run_context: Any) -> DocumentPipelineContext:
        """
        每次运行新建一个 context。

        不能复用 —— 它内含本次运行独有的 pipeline_run_id 与
        recorder，复用会让观测写串到别的运行上。
        """
        return DocumentPipelineContext(
            pipeline_run_id=run_context.pipeline_run_id,
            recorder=recorder,
            artifact_store=artifact_store,
            registry=capabilities,
            session_factory=session_factory,
            assert_state_size=settings.assert_state_size,
        )

    definition = build_graph_definition(
        compiled=compiled,
        context_factory=context_factory,
    )

    registry.register(definition)

    logger.info(
        "Registered graph %s v%s (fingerprint %s).",
        definition.name,
        definition.version,
        definition.topology_fingerprint[:12],
    )

    return definition


# ============================================================
# 2. dispatcher handler
# ============================================================


def build_dispatch_handler(
        *,
        definition: GraphDefinition,
        recorder: GraphRunRecorder,
        resume_state_loader: (
            Callable[[UUID], Awaitable[dict[str, Any] | None]]
            | None
        ) = None,
        settings: PipelineSettings | None = None,
) -> Callable[[DispatchEnvelope], Awaitable[None]]:
    """
    产出 dispatcher 用的 handler。

    handler 的职责是把**业务载荷**翻译成运行器的输入：

        DispatchEnvelope.payload  ->  RunSnapshot + input_state

    它跑在后台任务里，不在任何请求事务中。
    """

    settings = settings or PipelineSettings()

    runner = GraphRunner(recorder=recorder)

    async def handler(
            envelope: DispatchEnvelope,
    ) -> None:

        payload = envelope.payload

        document_version_id = _uuid(
            payload, "document_version_id"
        )

        # 主体信息（document_id / type_definition_id / file_id /
        # tenant_id）**只从库里的 loader 取**，不从 payload 传 ——
        # 两处都放会漂移，而且 payload 是调用方拼的，容易漏字段
        # （实测就漏过一次）。
        loaded_state = await _load_state(
            document_version_id, resume_state_loader
        )

        if loaded_state is None:
            raise RuntimeError(
                "Cannot start pipeline run: document version "
                f"{document_version_id} was not found."
            )

        parent_run_id = _optional_uuid(
            payload, "parent_pipeline_run_id"
        )

        # checkpoint 级别：按次覆盖优先，否则用应用级默认。
        #
        # 必须在运行开始前定 —— run 级启动的运行中途崩溃后
        # 没有中间 checkpoint 可恢复。
        level = settings.resolve_level(
            payload.get("checkpoint_level")
        )
        durability = (
            "sync" if level == "node" else "exit"
        )

        snapshot = RunSnapshot(
            pipeline_run_id=envelope.pipeline_run_id,
            thread_id=envelope.thread_id,
            graph_name=definition.name,
            graph_version=definition.version,
            topology_fingerprint=(
                definition.topology_fingerprint
            ),
            topology_snapshot=definition.topology_snapshot,
            subject_type=SUBJECT_TYPE,
            subject_id=document_version_id,
            subject_ref={
                "document_id": loaded_state["document_id"],
                "document_type_definition_id": (
                    loaded_state[
                        "document_type_definition_id"
                    ]
                ),
                "file_id": loaded_state.get("file_id"),
                "tenant_id": loaded_state.get("tenant_id"),
            },
            trigger_source=str(
                payload.get("trigger_source") or "manual"
            ),
            triggered_by=_optional_uuid(
                payload, "triggered_by"
            ),
            parent_pipeline_run_id=parent_run_id,
            # 链根：reprocess 链上第一次运行是根。
            # 扁平化后一次索引扫描就能取到某版本的全部历史。
            root_pipeline_run_id=(
                parent_run_id or envelope.pipeline_run_id
            ),
            run_config={
                "analysis_mode": loaded_state.get(
                    "analysis_mode"
                ),
                "pipeline_config": loaded_state.get(
                    "pipeline_config", {}
                ),
                "document_processor_definition_ids": payload.get(
                    "document_processor_definition_ids", []
                ),
                "force": bool(payload.get("force", False)),
                "graph_version": definition.version,
                # 记进 provenance：事后能查这一次到底可不可续跑。
                # run 级的运行崩溃后没有任何中间 checkpoint，
                # 前端不该提示"可以从断点恢复"。
                "checkpoint_level": level,
            },
            resumed_from_checkpoint_id=payload.get(
                "resumed_from_checkpoint_id"
            ),
        )

        # 续跑时传 None 让 LangGraph 从 checkpoint 恢复；
        # 否则用刚读出来的那份 state。
        input_state = (
            None
            if payload.get("resumed_from_checkpoint_id")
            else loaded_state
        )

        await runner.run(
            RunRequest(
                definition=definition,
                pipeline_run_id=envelope.pipeline_run_id,
                thread_id=envelope.thread_id,
                input_state=input_state,
                run_snapshot=snapshot,
                resume=input_state is None,
                durability=durability,
            )
        )

    return handler


# ============================================================
# Helpers
# ============================================================


async def _load_state(
        document_version_id: UUID,
        loader: (
            Callable[[UUID], Awaitable[dict[str, Any] | None]]
            | None
        ),
) -> dict[str, Any] | None:
    """
    从库里读出该版本的初始 state。

    必须由调用方提供 loader —— 初始 state 里的 version_number /
    pipeline_config / analysis_mode / document_type_definition_id
    只在库里，infra 侧不认识它们。

    续跑时**也要**读一次：虽然不会拿它当输入（传 None 让
    LangGraph 从 checkpoint 恢复），但 subject_ref 里那几个
    字段还是得从这里来。
    """

    if loader is None:
        raise RuntimeError(
            "No resume_state_loader provided; cannot build the "
            "initial pipeline state."
        )

    return await loader(document_version_id)


def _uuid(payload: dict[str, Any], key: str) -> UUID:
    value = payload.get(key)

    if value is None:
        raise ValueError(
            f"Dispatch payload is missing required key {key!r}. "
            f"Got: {sorted(payload)}."
        )

    return value if isinstance(value, UUID) else UUID(str(value))


def _optional_uuid(
        payload: dict[str, Any],
        key: str,
) -> UUID | None:
    value = payload.get(key)

    if value is None:
        return None

    return value if isinstance(value, UUID) else UUID(str(value))
