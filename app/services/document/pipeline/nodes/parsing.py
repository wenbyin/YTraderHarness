"""
阶段 3：Parsing。

归属：Loader / Reader。

职责
----
    file bytes -> normalized text / structured content

不负责：Chunk、Index、Storage。

用 ParseRequest / ParseResult 这对已有 DTO（processing.py），
不要另造一套 —— 它们是 DocumentParser 契约的一部分，
换实现时两边必须对得上。
"""
# from pathlib import Path
from __future__ import annotations
from typing import Any
from langgraph.runtime import Runtime
from app.infra.graph import node_span

from ..catalog import NODE_SPECS
from ..context import DocumentPipelineContext
from ..state import DocumentPipelineState
from ._common import file_id_of, tenant_of

_SPEC = next(s for s in NODE_SPECS if s.node_id == "parsing")

__all__ = ["parsing_node"]


async def parsing_node(
        state: DocumentPipelineState,
        runtime: Runtime[DocumentPipelineContext],
) -> dict[str, Any]:
    """
    解析源文件为文本。

    Returns:
        {"parse_ref": ArtifactRef}
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
                "detected_format": state.get("detected_format"),
            },
    ) as span:

        from uuid import UUID

        document_version_id = UUID(
            state["document_version_id"]
        )

        source = state["source_ref"]
        analysis_mode = state["analysis_mode"]
        detected_format = state["detected_format"]

        if analysis_mode == "default":
            parser = context.registry.find_default_parser(
                suffix=detected_format["file_extension"],
            )
        elif analysis_mode == "config":
            parser = context.registry.find_parser_by_key(
                parser_key=detected_format["loader_name"],
            )
        elif analysis_mode == "none":
            parser = None
        else:
            raise TypeError(f"Unknown analysis mode: {analysis_mode}")

        origin_text = await context.artifact_store.get_text(
            file_id_of(source)
        )

        # ⚠️ 不要写成 state["pipeline_config"]["loader"]["config"]。
        #
        # 快照里 loader 在"未匹配到前处理定义"时是**存在且值为 None**
        # 的（见 document_service_impl._build_version），而那正是
        # analysis_mode="default" 的常态 —— 也就是系统默认 Reader 这条
        # 必经之路。dict.get 的默认值只在**键不存在**时生效，键存在而
        # 值为 None 时它照样返回 None，接着下标就炸。
        #
        # 与 _processor.py 的 processor_name_of 是同一个陷阱，那里已经
        # 踩过一次并留了注释。
        loader_config = (
            ((state.get("pipeline_config") or {}).get("loader") or {})
            .get("config") or {}
        )

        # 必须先绑定：parser 为 None（analysis_mode="none"）时下面的分支
        # 不会赋值，直接读会 UnboundLocalError。
        parsed_text: str | None = None

        if parser:
            from app.services.document.domain.schemas import ParseRequest

            result: ParseResult = await parser.process(
                ParseRequest(
                    document_id=state["document_id"],
                    document_version_id=state["document_version_id"],
                    storage_key=source["storage_key"],
                    file_name=source["kind"],
                    content_type=source["mime_type"],
                    origin_text=origin_text,
                    config=dict(loader_config),
                )
            )

            parsed_text = result.text

        #   # result.text_checksum 应当等于新产物的 checksum
        # ----------------------------------------------------

        if parsed_text is None:
            parsed_text = ""

        ref = await context.artifact_store.put_text(
            kind="parsed_text",
            text=parsed_text,
            tenant_id=tenant_of(state),
            content_type="text/plain",
        )

        span.set_metrics({
            "char_count": len(parsed_text),
            "byte_size": ref["byte_size"],
            "placeholder": True,
        })

        span.set_output_summary({
            "ref": {
                "checksum": ref["checksum"],
                "char_count": ref["char_count"],
                "preview": ref["preview"],
            },
            "note": "占位实现：未调用 DocumentParser，源文本原样透传。",
        })

        return {"parse_ref": ref}
