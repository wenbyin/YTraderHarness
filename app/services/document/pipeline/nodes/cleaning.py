"""
阶段 5：Cleaning。

归属：Processor（ProcessorMode.COMMON 默认链的第 2 步）。

职责
----
去掉噪声内容：页眉页脚、页码、水印、重复分隔线、
扫描件识别产生的乱码片段。

与阶段 4 归一化的区别：
    归一化 = 统一"同一个字符的不同写法"（形）
    清洗   = 删掉"不该进检索的内容"（质）

执行逻辑全在 _processor.py，本文件只负责绑定节点身份。
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

_SPEC = next(s for s in NODE_SPECS if s.node_id == "cleaning")

__all__ = ["cleaning_node"]


async def cleaning_node(
        state: DocumentPipelineState,
        runtime: Runtime[DocumentPipelineContext],
) -> dict[str, Any]:
    """清洗归一化后的文本。返回 {"clean_ref": ArtifactRef}。"""

    async with node_span(
            runtime,
            node_id=_SPEC.node_id,
            label=_SPEC.label,
            group=_SPEC.group,
            sequence=_SPEC.sequence,
            input_summary={
                "normalize_ref": state.get("normalize_ref"),
            },
    ) as span:

        return await run_processor_stage(
            state,
            runtime,
            span=span,
            spec=_SPEC,
            input_key="normalize_ref",
            output_kind="clean_text",
            processor_name=processor_name_of(state),
        )
