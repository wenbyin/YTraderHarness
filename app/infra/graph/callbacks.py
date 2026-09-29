"""
LangGraph callback 兜底写入。

定位：**辅助网，不是主写入方**
--------------------------------
节点观测的主写入方是 instrumentation.node_span —— 因为只有
节点内部才能拿到 ExecutionInfo.node_attempt（重试计数），
也只有节点内部才知道自己属于哪个 Send 扇出分支。

callback 存在的唯一理由是覆盖 node_span 够不到的那一小块：

    on_chain_error 是**唯一**能捕获"节点已正常 return、
    但在 state 合并阶段抛错"的写入方。
    那种情况下节点体内的 try/except 根本不会执行。

覆盖范围刻意的收窄
------------------
本 handler 只处理**非扇出节点**（由调用方通过 handled_node_ids
指定）。原因：

    callback 拿到的 metadata 里，N 个 Send 扇出分支共享同一个
    langgraph_node 名字，无法区分是哪个 retrieval。
    若强行写入，只能用 branch_key='' 落一行"无名聚合行"，
    它与 node_span 写的每个分支行并存，会让节点计数重复、
    前端的扇出视图出现一个没有归属的幽灵节点。

    扇出分支的完整覆盖（start / end / error 三者）本来就由
    node_span 逐分支负责，这里不重复。

代价是：扇出节点若在 state 合并阶段失败，可能没有记录。
这是明确接受的 trade-off，不是遗漏。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

from .recorder import (
    GraphRunRecorder,
    NodeRunStatus,
    NodeRunUpdate,
)

logger = logging.getLogger(__name__)

__all__ = ["PersistingCallbackHandler"]


class PersistingCallbackHandler(AsyncCallbackHandler):
    """
    把 LangGraph 的节点生命周期事件落成观测行。

    每次运行构造一个实例，持有该运行的身份。
    callback 在调用时通过 config 传入（而不是构造在图上），
    这样不同运行之间不会串。
    """

    #: 抑制 LangChain 对未实现方法发出的告警噪音
    raise_error: bool = False

    def __init__(
            self,
            *,
            pipeline_run_id: UUID,
            recorder: GraphRunRecorder,
            handled_node_ids: frozenset[str],
            node_labels: Mapping[str, str] | None = None,
            node_sequences: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()

        self._pipeline_run_id = pipeline_run_id
        self._recorder = recorder
        self._handled = handled_node_ids
        self._labels = dict(node_labels or {})
        self._sequences = dict(node_sequences or {})

        # 已开始但尚未结束的节点，按 langgraph run_id 索引。
        # on_chain_end / on_chain_error 靠它把事件对回节点。
        self._open: dict[str, tuple[str, datetime]] = {}

        self._root_run_id: str | None = None

    # ========================================================
    # Chain lifecycle
    # ========================================================

    async def on_chain_start(
            self,
            serialized: dict[str, Any] | None,
            inputs: dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
    ) -> None:
        node_id = self._resolve_node(metadata)

        if node_id is None:
            return

        self._open[str(run_id)] = (
            node_id,
            datetime.now(timezone.utc),
        )

        await self._record(
            node_id,
            status=NodeRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

    async def on_chain_end(
            self,
            outputs: dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            tags: list[str] | None = None,
            **kwargs: Any,
    ) -> None:
        opened = self._open.pop(str(run_id), None)

        if opened is None:
            return

        node_id, started_at = opened

        await self._record(
            node_id,
            status=NodeRunStatus.COMPLETED,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )

    async def on_chain_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            tags: list[str] | None = None,
            **kwargs: Any,
    ) -> None:
        opened = self._open.pop(str(run_id), None)

        if opened is None:
            # 节点体从未成功启动（例如参数校验就炸了），
            # 此时没有 node_id 可归属，交给 runner 记到 run 行上。
            return

        node_id, started_at = opened

        finished_at = datetime.now(timezone.utc)

        await self._record(
            node_id,
            status=NodeRunStatus.FAILED,
            started_at=started_at,
            finished_at=finished_at,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    # ========================================================
    # Internals
    # ========================================================

    def _resolve_node(
            self,
            metadata: dict[str, Any] | None,
    ) -> str | None:
        """
        从 metadata 里解析出所属节点。

        必须双重过滤，否则会产生幻影行：
            1. metadata 里要有 langgraph_node
               （嵌套的 model / tool / retriever runnable 没有）
            2. 该名字要在本图声明过、且属于本次交给 callback
               覆盖的集合里
        """

        if not metadata:
            return None

        node_id = metadata.get("langgraph_node")

        if not isinstance(node_id, str):
            return None

        if node_id not in self._handled:
            return None

        return node_id

    async def _record(
            self,
            node_id: str,
            *,
            status: NodeRunStatus,
            started_at: datetime | None = None,
            finished_at: datetime | None = None,
            error_type: str | None = None,
            error_message: str | None = None,
    ) -> None:
        """
        写一行节点记录。

        attempt 固定为 1：callback 拿不到 ExecutionInfo.node_attempt，
        重试计数只有节点内部才知道。node_span 会用真实 attempt
        upsert，若二者指向同一行则以后写的为准。

        观测写入失败不打断业务，与 node_span 的处理一致。
        """

        duration_ms: int | None = None

        if started_at is not None and finished_at is not None:
            duration_ms = int(
                (finished_at - started_at).total_seconds() * 1000
            )

        update = NodeRunUpdate(
            pipeline_run_id=self._pipeline_run_id,
            node_id=node_id,
            node_label=self._labels.get(node_id, node_id),
            node_group="callback",
            sequence=self._sequences.get(node_id, -1),
            attempt=1,
            branch_key="",
            langgraph_run_id=self._root_run_id,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            error_type=error_type,
            error_message=error_message,
        )

        try:
            await self._recorder.upsert_node_run(update)

        except Exception:
            logger.exception(
                "Callback failed to record node=%s "
                "pipeline_run_id=%s; business flow continues.",
                node_id,
                self._pipeline_run_id,
            )

    # ========================================================

    def bind_root_run_id(self, run_id: str | None) -> None:
        """
        由 runner 在拿到 LangGraph 根 run_id 后回填。

        用于把观测行与 LangSmith 上的 trace 关联起来。
        """
        self._root_run_id = run_id
