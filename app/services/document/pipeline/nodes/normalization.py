"""
阶段 4：Normalization。

归属：Processor（ProcessorMode.COMMON 默认链的第 1 步）。

职责
----
统一编码、换行、空白、全半角、Unicode 规范化形式。
不做语义清洗 —— 那是阶段 5。

当 analysis_mode == "none" 时
----------------------------
不解析就不该归一化：输入被声明为"已是标准化内容"，
再动它就是违背调用方的约定。此时把 parse_ref 原样透传。

三个 Processor 节点（normalize / clean / extract）形状完全一样，
因此提升成一个共享的执行函数，只在 processor_name 上分叉。
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.infra.graph import node_span

from ..catalog import NODE_SPECS
from ..context import DocumentPipelineContext
from ..state import DocumentPipelineState
from ._processor import (
    processor_name_of,
    run_processor_stage,
)

_SPEC = next(
    s for s in NODE_SPECS if s.node_id == "normalization"
)

__all__ = ["normalization_node"]


async def normalization_node(
        state: DocumentPipelineState,
        runtime: Runtime[DocumentPipelineContext],
) -> dict[str, Any]:
    """
    归一化 parse 产出的文本。

    Returns:
        {"normalize_ref": ArtifactRef}
    """

    async with node_span(
            runtime,
            node_id=_SPEC.node_id,
            label=_SPEC.label,
            group=_SPEC.group,
            sequence=_SPEC.sequence,
            input_summary={"parse_ref": state.get("parse_ref")},
    ) as span:

        return await run_processor_stage(
            state,
            runtime,
            span=span,
            spec=_SPEC,
            input_key="parse_ref",
            output_kind="normalized_text",
            processor_name=processor_name_of(state),
        )
