"""
阶段 1：Ingestion。

归属：File / Document Service。

职责
----
把 state 里的 file_id 变成后续阶段能用的 ArtifactRef。

本节点**不复制源文件**。源文件本来就是 file 表里的一条 File
（用户上传的那个），产物改走 File 之后直接引用它即可。

早期版本会把它复制成一份 source_bytes 产物，理由是"重跑时
不必回 File 模块再读一次盘"。那个理由在产物还是独立存储时
成立，现在不成立了 —— 复制等于一份 432 KB 的文档白存 432 KB，
还会在 file 表里多出一行内容完全相同的记录。

本节点不做 IO。真实实现可以在这里确认源文件可读
（FileStorage.exists），读不到就显式失败，而不是拖到 parsing
阶段才发现。
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.infra.graph import node_span
from app.common.utils import assert_readable

from ..catalog import NODE_SPECS
from ..context import DocumentPipelineContext
from ..state import DocumentPipelineState, ArtifactRef
from ._common import tenant_of

_SPEC = next(
    s for s in NODE_SPECS if s.node_id == "ingestion"
)

__all__ = ["ingestion_node"]


async def ingestion_node(
        state: DocumentPipelineState,
        runtime: Runtime[DocumentPipelineContext],
) -> dict[str, Any]:
    """
    读源文件并落成产物。

    Returns:
        {"source_ref": ArtifactRef, "detected_format": None}
    """

    context = runtime.context

    async with node_span(
            runtime,
            node_id=_SPEC.node_id,
            label=_SPEC.label,
            group=_SPEC.group,
            sequence=_SPEC.sequence,
    ) as span:
        file_id = state["file_id"]

        span.set_input_summary({
            "file_id": file_id,
            "document_version_id": state["document_version_id"],
        })

        from uuid import UUID

        uuid_file_id = UUID(file_id)

        # 源文件本来就是 file 表里的一条 File（用户上传的那个），
        # 产物改走 File 之后**不需要再复制一份** —— 直接引用。
        #
        # 之前那样做等于一份 432 KB 的文档白存 432 KB，
        # 而且会在 file 表里多出一行内容完全相同的记录。
        #
        # TODO(ingestion): 真实实现应当先确认源文件可读
        #   （FileRepository.get + 必要时 FileStorage.exists），
        #   读不到要显式失败，而不是让 parsing 阶段才发现。
        ref = await context.artifact_store.reference(
            uuid_file_id
        )

        file_name = ref["kind"]

        raw = await context.artifact_store.get_bytes(
            uuid_file_id
        )

        assert_readable(
            raw=raw,
            file_name=file_name
        )

        span.set_metrics({
            "byte_size": ref["byte_size"],
            "referenced": True,
        })

        span.set_output_summary({
            "ref": {
                "file_id": ref["file_id"],
                "checksum": ref["checksum"],
                "byte_size": ref["byte_size"],
            },
            "note": "直接引用用户上传的源文件，未复制。",
        })

        return {"source_ref": ref}
