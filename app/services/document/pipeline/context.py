"""
流水线的运行时依赖（LangGraph 的 context_schema）。

传法（容易踩的坑）
------------------
LangGraph 1.x 的 context **不是**放在 config 里传的，而是
ainvoke / astream 的关键字参数：

    await graph.ainvoke(input, config, context=ctx)   ✅
    await graph.ainvoke(input, {"context": ctx})      ❌ 会静默变成 None

传错时 runtime.context 是 None，报错离原因很远；
instrumentation._require_context 会给出能直接照着修的提示。

为什么持 session_factory 而不是 session
---------------------------------------
Send 扇出的 N 个分支是**并发**执行的，共享一个 session 会直接
踩到 asyncpg 的并发使用限制。每个分支从工厂取自己的会话。

session_factory 的使用纪律
--------------------------
节点**默认不该碰它**：

    观测写入    -> recorder（内部自理短事务）
    产物读写    -> artifact_store（自持 FileStorage，不碰库）
    业务状态写  -> 只有 finalize 一处，见下

允许使用的两种情况：

    1. metadata_extraction 读一次，解析出要扇出哪些 retrieval。
       这是**只读**决策，开一段短会话即可。

    2. finalize_version 原子地写回业务状态
       （DocumentVersionRetrievalBinding 多行 +
       DocumentVersion.processing_status）。
       这几行必须同事务，否则会出现"绑定写了但版本状态没更新"
       的中间态。

除这两处外新增用法前请先想清楚：写观测走 recorder，
不要顺手在这里开事务 —— 那会让"观测写入自成事务、
业务回滚带不走观测"这条保证失效。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from app.infra.graph.recorder import GraphRunRecorder

if TYPE_CHECKING:
    from .observability.artifacts import ArtifactStore
    from .registry import PipelineRegistry

__all__ = ["DocumentPipelineContext"]


@dataclass(frozen=True, slots=True)
class DocumentPipelineContext:
    """
    一次运行的依赖容器。

    实现 infra 的 GraphRunContext Protocol（pipeline_run_id +
    recorder 两个字段），因此 node_span 能直接用。
    """

    pipeline_run_id: UUID

    recorder: GraphRunRecorder

    artifact_store: "ArtifactStore"

    registry: "PipelineRegistry"

    #: 只给 metadata_extraction（只读决策）与 finalize_version
    #: （原子写回业务状态）用。详见模块 docstring 的使用纪律。
    session_factory: async_sessionmaker[AsyncSession] | None = None

    #: 本次运行是否要断言 state 体积。
    #: 开发/冒烟时打开，生产上关掉以避免每次节点后多一次序列化。
    assert_state_size: bool = False
