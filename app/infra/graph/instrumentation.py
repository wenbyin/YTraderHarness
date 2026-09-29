"""
节点埋点与 state 体积护栏。

这是「共同部分」的核心：所有节点用同一段 `node_span`
包住函数体，节点作者不必写任何计时、写库、错误归属的样板。

为什么必须由节点自己写，而不是靠 callback
------------------------------------------
两个硬约束决定了这件事：

1. Send 扇出的 N 个分支**共享同一个 langgraph_node 名字**。
   callback 拿到的 metadata 里区分不出这是哪个 retrieval，
   分支身份只能由节点从 Send payload 显式带入。

2. ExecutionInfo.node_attempt（重试计数）**只在节点内部可见**，
   callback 完全拿不到。

再加上业务指标（chunk 数、向量维度）callback 也看不到，
结论就是：节点内的 node_span 是主写入方，
callback（callbacks.py）只是兜底的网。
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from uuid import UUID

from .ids import branch_key as make_branch_key
from .recorder import (
    GraphRunContext,
    NodeRunStatus,
    NodeRunUpdate,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_STATE_BYTES",
    "MAX_PREVIEW_CHARS",
    "ExecutionIds",
    "NodeSpan",
    "assert_checkpoint_safe",
    "execution_ids",
    "node_span",
    "truncate_preview",
]


# ============================================================
# 常量
# ============================================================

#: state 序列化后的体积上限。超过说明有人把正文塞进了 state。
DEFAULT_MAX_STATE_BYTES = 64 * 1024

#: ArtifactRef.preview 的最大长度。它只是给 UI 看的片段。
MAX_PREVIEW_CHARS = 256

#: ArtifactRef 必须包含的键。infra 不知道业务类型，
#: 因此按结构校验而非按类校验。
#:
#: file_id 而非 storage_key：产物是 File 模块里的一条 File，
#: 身份是 file_id；storage_key 是存储层实现细节，不进 state。
ARTIFACT_REF_REQUIRED_KEYS = frozenset({
    "file_id",
    "checksum",
    "kind",
    "byte_size",
    "preview",
})


# ============================================================
# Execution identity
# ============================================================


@dataclass(frozen=True, slots=True)
class ExecutionIds:
    """从 Runtime 里取出的当前节点执行标识。"""

    task_id: str | None = None
    attempt: int = 1
    thread_id: str | None = None
    run_id: str | None = None


def execution_ids(runtime: Any) -> ExecutionIds:
    """
    从 Runtime 提取执行标识。

    `runtime.execution_info` 需要 langgraph >= 1.1.5。
    低版本上取不到，此时降级为 attempt=1 而不是抛错 ——
    重试计数会失真，但流程不崩。

    用 getattr 而非直接属性访问，就是为了让这段代码
    在 1.0.x 上也能跑。
    """

    info = getattr(runtime, "execution_info", None)

    if info is None:
        return ExecutionIds()

    return ExecutionIds(
        task_id=getattr(info, "task_id", None),
        attempt=getattr(info, "node_attempt", 1) or 1,
        thread_id=getattr(info, "thread_id", None),
        run_id=getattr(info, "run_id", None),
    )


# ============================================================
# Node span
# ============================================================


def truncate_preview(
        value: str,
        limit: int = MAX_PREVIEW_CHARS,
) -> str:
    """
    把一段文本截成安全的 preview。

    刻意不保留尾部：preview 是给人快速确认"这一步产出的是
    不是我以为的东西"，不是样本，保留头部的信息量最大。
    """

    if len(value) <= limit:
        return value

    return value[:limit]


@dataclass
class NodeSpan:
    """
    节点执行期间可写的观测数据。

    节点在自己的函数体里按需设置，node_span 在退出时统一落库。
    """

    pipeline_run_id: UUID
    node_id: str
    node_label: str
    node_group: str
    sequence: int
    attempt: int
    branch_key: str
    branch_scope_id: UUID | None

    task_id: str | None = None
    run_id: str | None = None

    input_summary: dict[str, Any] | None = None
    output_summary: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None

    error_type: str | None = None
    error_message: str | None = None
    error_detail: dict[str, Any] | None = None

    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    _started_perf: float = field(default_factory=time.perf_counter)

    # --------------------------------------------------------
    # 写入接口（节点内部调用）
    # --------------------------------------------------------

    def set_input_summary(
            self,
            summary: dict[str, Any],
    ) -> None:
        """输入摘要。只放 id 和计数，绝不放正文。"""
        self.input_summary = summary

    def set_output_summary(
            self,
            summary: dict[str, Any],
    ) -> None:
        """输出摘要。只放 id、checksum 和计数。"""
        self.output_summary = summary

    def set_metrics(
            self,
            metrics: dict[str, Any],
    ) -> None:
        """
        业务指标：chunk_count / token_count /
        embedding_dimensions / vector_count 等。

        callback 拿不到这些，这也是 node_span 必须存在的理由之一。
        """
        self.metrics = metrics

    def set_error(
            self,
            exc: BaseException,
            *,
            detail: dict[str, Any] | None = None,
    ) -> None:
        self.error_type = type(exc).__name__
        self.error_message = str(exc)
        self.error_detail = detail

    # --------------------------------------------------------

    @property
    def duration_ms(self) -> int:
        return int(
            (time.perf_counter() - self._started_perf) * 1000
        )

    def to_update(
            self,
            *,
            status: NodeRunStatus,
            finished_at: datetime | None = None,
    ) -> NodeRunUpdate:
        return NodeRunUpdate(
            pipeline_run_id=self.pipeline_run_id,
            node_id=self.node_id,
            node_label=self.node_label,
            node_group=self.node_group,
            sequence=self.sequence,
            attempt=self.attempt,
            branch_key=self.branch_key,
            branch_scope_id=self.branch_scope_id,
            langgraph_task_id=self.task_id,
            langgraph_run_id=self.run_id,
            status=status,
            started_at=self.started_at,
            finished_at=finished_at,
            duration_ms=self.duration_ms,
            input_summary=self.input_summary,
            output_summary=self.output_summary,
            metrics=self.metrics,
            error_type=self.error_type,
            error_message=self.error_message,
            error_detail=self.error_detail,
        )


@asynccontextmanager
async def node_span(
        runtime: Any,
        *,
        node_id: str,
        label: str,
        group: str,
        sequence: int,
        branch_scope_id: UUID | None = None,
        input_summary: dict[str, Any] | None = None,
) -> AsyncIterator[NodeSpan]:
    """
    包住一个节点的函数体，自动记录开始、结束、耗时与错误。

    用法：

        async with node_span(runtime, node_id="parsing",
                             label="解析", group="linear",
                             sequence=2) as span:
            ...
            span.set_metrics({"char_count": len(text)})
            return {"parse_ref": ref}

    观测写入失败**不会**让业务流程失败：文档该处理还是要处理完，
    丢的是这一次的可观测性，记 exception 日志而不是吞掉。
    """

    context = _require_context(runtime)
    ids = execution_ids(runtime)

    span = NodeSpan(
        pipeline_run_id=context.pipeline_run_id,
        node_id=node_id,
        node_label=label,
        node_group=group,
        sequence=sequence,
        attempt=ids.attempt,
        branch_key=make_branch_key(branch_scope_id),
        branch_scope_id=branch_scope_id,
        task_id=ids.task_id,
        run_id=ids.run_id,
        input_summary=input_summary,
    )

    await _safe_record(
        context,
        span.to_update(status=NodeRunStatus.RUNNING),
        node_id=node_id,
        phase="start",
    )

    try:
        yield span

    except BaseException as exc:
        span.set_error(exc)
        await _safe_record(
            context,
            span.to_update(
                status=NodeRunStatus.FAILED,
                finished_at=datetime.now(timezone.utc),
            ),
            node_id=node_id,
            phase="error",
        )
        raise

    else:
        await _safe_record(
            context,
            span.to_update(
                status=NodeRunStatus.COMPLETED,
                finished_at=datetime.now(timezone.utc),
            ),
            node_id=node_id,
            phase="end",
        )


def _require_context(runtime: Any) -> GraphRunContext:
    """
    取出 Runtime.context，取不到时给一个能直接照着修的报错。

    LangGraph 1.x 的 context **不是**通过 config 传的，而是
    ainvoke / astream 的关键字参数：

        await graph.ainvoke(input, config, context=ctx)
                                       ^^^^^^^^^^^^^ 不是 config={"context": ctx}

    传错时 runtime.context 是 None，默认报错是
    "'NoneType' object has no attribute ..."，离原因很远。
    """

    context = getattr(runtime, "context", None)

    if context is None:
        raise RuntimeError(
            "runtime.context is None. LangGraph 1.x passes context as a "
            "keyword argument of ainvoke/astream, not inside config:\n"
            "    await graph.ainvoke(input, config, context=ctx)\n"
            "Check the graph runner passes it."
        )

    return context


async def _safe_record(
        context: GraphRunContext,
        update: NodeRunUpdate,
        *,
        node_id: str,
        phase: str,
) -> None:
    """
    写入观测行，但绝不因为写不进去而打断业务。

    注意这里吞掉的只是观测异常。业务异常由 node_span 的
    except 分支原样 re-raise，不受影响。
    """

    try:
        await context.recorder.upsert_node_run(update)

    except Exception:
        logger.exception(
            "Failed to record node run (%s) for node=%s "
            "pipeline_run_id=%s; business flow continues.",
            phase,
            node_id,
            context.pipeline_run_id,
        )


# ============================================================
# Checkpoint safety
# ============================================================


def assert_checkpoint_safe(
        state: dict[str, Any],
        *,
        ref_keys: tuple[str, ...] = (),
        max_bytes: int = DEFAULT_MAX_STATE_BYTES,
        max_preview: int = MAX_PREVIEW_CHARS,
) -> None:
    """
    断言 state 小到可以安全地反复 checkpoint。

    为什么需要这道闸
    ----------------
    checkpointer 在**每个 super-step 之后重新序列化整个
    channel 的累积值**，写库成本随线程长度增长而不是随增量。
    几 MB 的正文 × 11 个节点 × N 个扇出分支会被反复写库。

    这个断言是回归护栏：没有它，一个月内一定会有人
    为了方便把 text 塞回 state。

    Parameters
    ----------
    ref_keys
        应当存放 ArtifactRef 的字段名。每个值要么是 None，
        要么是含 ARTIFACT_REF_REQUIRED_KEYS 的 dict。
    max_bytes
        序列化后的体积上限。

    Raises
    ------
    AssertionError
        体积超限，或某个 ref 字段不是合法引用。
    """

    import orjson

    serialized = orjson.dumps(
        state,
        default=str,
    )

    if len(serialized) > max_bytes:
        raise AssertionError(
            "Pipeline state is too large to checkpoint safely: "
            f"{len(serialized)} bytes > {max_bytes}. "
            "Heavy payloads (document text, vectors) must be "
            "written to the artifact store and referenced via "
            "ArtifactRef instead of being carried in state. "
            f"Largest fields: {_largest_fields(state)}"
        )

    for key in ref_keys:

        value = state.get(key)

        if value is None:
            continue

        if not isinstance(value, dict):
            raise AssertionError(
                f"state[{key!r}] must be an ArtifactRef dict "
                f"or None, got {type(value).__name__}."
            )

        missing = ARTIFACT_REF_REQUIRED_KEYS - value.keys()

        if missing:
            raise AssertionError(
                f"state[{key!r}] is not a valid ArtifactRef; "
                f"missing keys: {sorted(missing)}."
            )

        # 键存在还不够 —— 值为 None 的引用在续跑时同样取不回内容，
        # 而且这种半成品比缺字段更难在运行期发现。
        for required in ("file_id", "checksum", "kind"):
            if not value.get(required):
                raise AssertionError(
                    f"state[{key!r}].{required} is empty; "
                    "an ArtifactRef must be resolvable after "
                    "a resume, not just present."
                )

        byte_size = value.get("byte_size")

        if not isinstance(byte_size, int) or byte_size < 0:
            raise AssertionError(
                f"state[{key!r}].byte_size must be a "
                f"non-negative int, got {byte_size!r}."
            )

        preview = value.get("preview") or ""

        if len(preview) > max_preview:
            raise AssertionError(
                f"state[{key!r}].preview is too long: "
                f"{len(preview)} > {max_preview} chars. "
                "preview must be a short UI snippet, not content."
            )


def _largest_fields(
        state: dict[str, Any],
        *,
        top: int = 5,
) -> list[tuple[str, int]]:
    """给体积超限的报错附上"是谁把 state 撑大的"。"""

    import orjson

    sizes: list[tuple[str, int]] = []

    for key, value in state.items():
        try:
            size = len(orjson.dumps(value, default=str))
        except Exception:
            size = -1
        sizes.append((key, size))

    sizes.sort(key=lambda item: item[1], reverse=True)

    return sizes[:top]
