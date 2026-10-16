"""
阶段 8：Chunk Enrichment（扇出）。

归属：DocumentRetrievalDefinition / Chunker 配置。

给上一阶段切出的 chunk 补定位信息与元数据（见 enricher/base.py）。
增强规则随 retrieval 不同而不同，因此仍是扇出阶段。

输入是上一阶段在本分支的产物，由路由屏障按 retrieval id
注入到 payload 的 upstream_ref（分支节点看不到父状态）。
"""

from __future__ import annotations

import json
from typing import Any

from langgraph.runtime import Runtime

from app.infra.graph import node_span

from app.services.document.domain.schemas import EnrichRequest
from app.services.document.domain.schemas.enum import (
    ProcessorRole,
)

from ..catalog import NODE_SPECS
from ..context import DocumentPipelineContext
from ..state import BranchInput
from ._branch import branch_span_kwargs, ref_summary
from ._common import file_id_of, tenant_of

_SPEC = next(
    s for s in NODE_SPECS if s.node_id == "chunk_enrichment"
)

__all__ = ["chunk_enrichment_node"]


async def chunk_enrichment_node(
        state: BranchInput,
        runtime: Runtime[DocumentPipelineContext],
) -> dict[str, Any]:
    """增强本分支的 chunk。返回 {"enrich_refs": {rid: ArtifactRef}}。"""

    context = runtime.context
    branch = state["branch"]
    rid = branch["document_retrieval_definition_id"]

    async with node_span(
            runtime,
            **branch_span_kwargs(_SPEC, state),
            input_summary={
                "chunker_name": branch["chunker_name"],
                "upstream_ref": ref_summary(state.get("upstream_ref")),
            },
    ) as span:

        from uuid import UUID

        document_version_id = UUID(
            state["document_version_id"]
        )

        upstream = state.get("upstream_ref")

        if upstream is None:
            raise ValueError(
                "chunk_enrichment: upstream_ref is missing; "
                "the routing barrier did not inject this "
                "branch's chunk artifact."
            )

        # ----------------------------------------------------
        # TODO(chunk_enrichment): 调用 ChunkEnricher。
        #
        # 现在是占位实现：原样透传上游 chunk，只把产物 kind 换成
        # chunks_enriched。
        #
        # 真实实现应当是：
        #
        #   chunks = json.loads(await artifact_store.get_text(...))
        #   enricher = context.registry.enricher(<配置里的名字>)
        #   enriched = await enricher.enrich(
        #       chunks, config=branch["chunker_config"],
        #       document_metadata=<extract_ref 指向的元数据>)
        #   更新 document_chunk.metadata（按 document_chunk_id upsert）
        #
        # 实现**不得**改动 content 与 chunk_index ——
        # 那会破坏 uq_document_chunk_version_retrieval_index
        # 与 content_hash 的一致性。
        # ----------------------------------------------------

        upstream_text = await context.artifact_store.get_text(
            file_id_of(upstream)
        ) or ""

        items = json.loads(upstream_text) if upstream_text else []

        if not items:
            return {"enrich_refs": {rid: upstream}}

        # ----------------------------------------------------
        # 取文档级元数据 —— 增强的主要输入
        # ----------------------------------------------------
        #
        # 它由阶段 6（metadata_extraction）产出，经 extract_ref 指向。
        # ⚠️ 要读**产物内容**，不是那个 ref 本身 —— ref 只是
        # {file_id, checksum, kind}，把它当 document_metadata 传下去，
        # 增强会因为"没有可继承的东西"而静默什么都不做。
        document_metadata = await _document_metadata(
            context, state
        )

        enricher = context.registry.processor_by_role(
            ProcessorRole.CHUNK_ENRICH
        )

        if enricher is None:
            # 没注册增强实现：透传而不是失败。增强是锦上添花，
            # 缺它不该让整条写入链挂掉（与 chunker / embedder 不同，
            # 那两个缺了产不出东西）。
            span.set_metrics({"skipped": True, "reason": "无增强实现"})
            return {"enrich_refs": {rid: upstream}}

        # 走 **process()** 这个统一入口，与其它阶段一致 —— 直接调
        # enrich() 会让 process() 成为一条没人走的死路（里面还可能
        # 悄悄用错字段，正是它之前把累积 metadata 当文档事实传的原因）。
        result = await enricher.process(
            EnrichRequest(
                document_id=UUID(state["document_id"]),
                document_version_id=UUID(
                    state["document_version_id"]
                ),
                chunks=[_to_chunk(item) for item in items],
                document_metadata=document_metadata,
                config=dict(branch.get("chunker_config") or {}),
            )
        )

        enriched = list(result.chunks or [])

        # ----------------------------------------------------
        # 写回产物
        # ----------------------------------------------------
        #
        # 除 metadata 外还要带上 **locator** —— chunking 的载荷里没有
        # 它，不在这里补上就永远流不到 indexing（见 index/es.py 的
        # _MAPPING：locator 是 ES 文档的字段之一）。
        #
        # ⚠️ embed_text 同理，必须**显式**写出来：`**item` 展开的是
        # chunking 那份原始载荷，里面没有这个键 —— 增强算出来的值会因为
        # "忘了加一行"而静默消失，下游只表现为向量里没有上下文。
        text = json.dumps(
            [
                {
                    **item,
                    "metadata": dict(chunk.metadata or {}),
                    "locator": dict(chunk.locator or {}),
                    "embed_text": chunk.embed_text,
                }
                for item, chunk in zip(items, enriched)
            ],
            ensure_ascii=False,
        )

        ref = await context.artifact_store.put_text(
            kind="chunks_enriched",
            text=text,
            tenant_id=tenant_of(state),
            content_type="application/json",
        )

        span.set_metrics({
            "char_count": ref["char_count"],
            "enricher": enricher.get_key(),
            "enriched_count": len(enriched),
            # 有 title 说明文档级元数据真的拿到了；没有就说明上游
            # metadata 阶段还没产出内容（它目前也是占位实现）。
            "has_document_metadata": bool(document_metadata.get("title")),
        })

        span.set_output_summary({
            "ref": {
                "checksum": ref["checksum"],
                "file_id": ref["file_id"],
            },
            "enricher": enricher.get_key(),
        })

        return {"enrich_refs": {rid: ref}}


async def _document_metadata(
        context,
        state: BranchInput,
) -> dict[str, Any]:
    """
    读 extract_ref 指向的元数据产物。

    读不到（还没产出 / 阶段 6 仍是占位）就返回空 dict —— 增强会因此
    什么都不加，但不会报错。这比抛异常合适：写入链不该因为"元数据
    阶段还没做完"而整条挂掉。
    """

    extract_ref = state.get("extract_ref")

    if not extract_ref:
        return {}

    raw = await context.artifact_store.get_text(
        file_id_of(extract_ref)
    )

    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _to_chunk(item: dict[str, Any]):
    """
    产物里的条目 -> DocumentChunk 实体。

    enricher 的入参是**实体**（它有 metadata / locator 属性），而产物
    里是 JSON —— 这一步把形状还原回去。增强完再序列化回产物。
    """

    from ...domain.entities import DocumentChunk

    return DocumentChunk(
        document_chunk_id=item["document_chunk_id"],
        document_version_id=item["document_version_id"],
        chunker_key=item["chunker_key"],
        chunker_config=dict(item.get("chunker_config") or {}),
        chunk_index=item["chunk_index"],
        start_index=item.get("start_index"),
        end_index=item.get("end_index"),
        content=item.get("content") or "",
        content_hash=item.get("content_hash"),
        token_count=item.get("token_count"),
        char_count=item.get("char_count"),
        locator=item.get("locator") or {},
        metadata=item.get("metadata") or {},
        # 上一轮的产物里可能已经带了（重跑时），原样带回去让增强重算 ——
        # 不可选、不可省：漏掉这一行的话，增强算出来的值写不回产物。
        embed_text=item.get("embed_text"),
    )
