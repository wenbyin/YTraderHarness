"""
ContextBuilder 的 Trace 上下文与阶段映射。

阶段名是**提示词里的固定契约**：

    context.profile_load       <- StageId.PROFILE_LOADER
    context.summary_load       <- StageId.SUMMARY_LOADER
    context.history_select     <- StageId.HISTORY_SELECTOR
    context.memory_retrieve    <- StageId.MEMORY_RETRIEVER
    context.knowledge_retrieve <- StageId.KNOWLEDGE_RETRIEVER
    context.prompt_assemble    <- StageId.PROMPT_ASSEMBLER

其余阶段（resolver / capability / budget / compactor / validator）不在
提示词的词表里，**不发事件** —— 不新增 event_type。

每一对前后缀都带 `.started` / `.completed` / `.failed`，与提示词逐字一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.common.context import StageId
from app.common.event import (
    EventType,
    SpanKind,
    TraceEmitter,
    TraceSpan,
)

__all__ = ["ContextTrace", "stage_span"]


@dataclass(frozen=True, slots=True)
class ContextTrace:
    """一次 context.build 的 trace 句柄（由节点创建）。"""

    #: 与节点共用同一个 custom stream writer。
    emitter: TraceEmitter

    #: context.build 的 span_id —— 所有阶段 Span 的父。
    parent_span_id: str


_STAGE_SPANS: dict[StageId, tuple[EventType, EventType, EventType, str]] = {
    StageId.PROFILE_LOADER: (
        EventType.CONTEXT_PROFILE_LOAD_STARTED,
        EventType.CONTEXT_PROFILE_LOAD_COMPLETED,
        EventType.CONTEXT_PROFILE_LOAD_FAILED,
        "context.profile_load",
    ),
    StageId.SUMMARY_LOADER: (
        EventType.CONTEXT_SUMMARY_LOAD_STARTED,
        EventType.CONTEXT_SUMMARY_LOAD_COMPLETED,
        EventType.CONTEXT_SUMMARY_LOAD_FAILED,
        "context.summary_load",
    ),
    StageId.HISTORY_SELECTOR: (
        EventType.CONTEXT_HISTORY_SELECT_STARTED,
        EventType.CONTEXT_HISTORY_SELECT_COMPLETED,
        EventType.CONTEXT_HISTORY_SELECT_FAILED,
        "context.history_select",
    ),
    StageId.MEMORY_RETRIEVER: (
        EventType.CONTEXT_MEMORY_RETRIEVE_STARTED,
        EventType.CONTEXT_MEMORY_RETRIEVE_COMPLETED,
        EventType.CONTEXT_MEMORY_RETRIEVE_FAILED,
        "context.memory_retrieve",
    ),
    StageId.KNOWLEDGE_RETRIEVER: (
        EventType.CONTEXT_KNOWLEDGE_RETRIEVE_STARTED,
        EventType.CONTEXT_KNOWLEDGE_RETRIEVE_COMPLETED,
        EventType.CONTEXT_KNOWLEDGE_RETRIEVE_FAILED,
        "context.knowledge_retrieve",
    ),
    StageId.PROMPT_ASSEMBLER: (
        EventType.CONTEXT_PROMPT_ASSEMBLE_STARTED,
        EventType.CONTEXT_PROMPT_ASSEMBLE_COMPLETED,
        EventType.CONTEXT_PROMPT_ASSEMBLE_FAILED,
        "context.prompt_assemble",
    ),
}


def stage_span(
        trace: ContextTrace | None,
        stage: StageId,
        *,
        attempt: int = 1,
) -> TraceSpan | None:
    """
    为一个**实际执行**的阶段开 Span。

    返回 None 的两种情况：trace 未开启，或这个 stage 不在提示词词表里
    （resolver / capability / budget / ...）。两者都不应该发事件。
    """

    if trace is None:
        return None

    spec = _STAGE_SPANS.get(stage)

    if spec is None:
        return None

    started, completed, failed, span_name = spec

    return trace.emitter.start(
        started,
        span_kind=SpanKind.CONTEXT_STAGE,
        span_name=span_name,
        end_type=completed,
        failed_type=failed,
        parent_span_id=trace.parent_span_id,
        attempt=attempt,
        payload={},
    )


def stage_metrics(state: Any) -> dict[str, Any]:
    """
    阶段完成时可可靠拿到的计数。

    第一阶段**不硬凑数值**：没有全阶段统一的 item_count 字段，
    而 estimated_input_tokens 要到最终组装完才算得出来。这里返回空，
    等真正需要时按阶段补（提示词允许"只有有意义的数据才填写"）。
    """

    return {}
