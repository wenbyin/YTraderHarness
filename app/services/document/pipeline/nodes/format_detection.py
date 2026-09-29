"""
阶段 2：Format Detection。

归属：系统，按 suffix / MIME 判断。

职责
----
决定这份文件**由谁来解析**，不解析内容本身。

选路依据（按优先级）
--------------------
    DocumentProcessorDefinition
        当前版本已绑定的前处理定义。它在 create_document 时
        已按扩展名匹配好并写进 version.document_processor_definition_id，
        这里直接取用，不重新匹配 —— 否则"创建时匹配 A、
        处理时匹配到 B"会造成配置漂移且难以解释。

    analysis_mode
        none        不解析，要求输入已是标准化内容
        default     用系统预制的中缀名 Reader
        config      用该类型自定义的 DocumentProcessorDefinition

    suffix / MIME
        DocumentParser.supports() 的输入。

本节点只产出"谁来做 + 用什么参数"，不产出内容。
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.infra.graph import node_span

from ..catalog import NODE_SPECS
from ..context import DocumentPipelineContext
from ..state import DocumentPipelineState

_SPEC = next(
    s for s in NODE_SPECS if s.node_id == "format_detection"
)

__all__ = ["format_detection_node"]


async def format_detection_node(
        state: DocumentPipelineState,
        runtime: Runtime[DocumentPipelineContext],
) -> dict[str, Any]:
    """
    选定解析器。

    Returns:
        {"detected_format": {
            "loader_name": str | None,
            "parser": str,          # 选中的实现标识
            "file_extension": str,
            "content_type": str | None,
        }}
    """

    context = runtime.context

    async with node_span(
            runtime,
            node_id=_SPEC.node_id,
            label=_SPEC.label,
            group=_SPEC.group,
            sequence=_SPEC.sequence,
            input_summary={
                "source_ref": state.get("source_ref"),
            },
    ) as span:

        analysis_mode = state["analysis_mode"]

        loader = (
            state["pipeline_config"].get("loader") or {}
        )

        # ----------------------------------------------------
        # TODO(format_detection): 按 suffix / MIME 选定 parser。
        #
        # 现在是占位实现：只把已知的配置信息整理出来，
        # **不调用** registry.find_parser() —— 因为
        # parser/ 下目前一个具体实现都没有，注册表是空的，
        # 调用必然抛 CapabilityNotFound。
        #
        # 真实实现应当是：
        #
        #   from pathlib import Path
        #   ext = Path(file_name).suffix.lower()
        #   parser = context.registry.find_parser(content_type, file_name)
        #
        #   analysis_mode == "none"   -> 跳过解析，把 source_ref
        #                                直接当 parse_ref，不要造空产物
        #   analysis_mode == "config" 且无匹配的
        #       DocumentProcessorDefinition -> 显式失败
        #       （这本该在 create_document 就拦下）
        # ----------------------------------------------------

        detected = {
            "loader_name": loader.get("name"),
            "parser": None,
            "file_extension": state["pipeline_config"].get(
                "file_extension"
            ),
            "analysis_mode": analysis_mode,
            "registry_size": len(
                context.registry.parsers
            ),
            "placeholder": True,
        }

        span.set_metrics({
            "parser_registry_size": len(
                context.registry.parsers
            ),
            "analysis_mode": analysis_mode,
            "placeholder": True,
        })

        span.set_output_summary({
            "detected_format": detected,
            "note": "占位实现：未真正选择 parser（注册表为空）。",
        })

        return {"detected_format": detected}
