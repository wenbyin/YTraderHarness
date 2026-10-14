"""
阶段 7：Chunking（扇出）。

归属：DocumentRetrievalDefinition。

产出的是**版本级的 chunk**，归属是 chunker 身份，不是 retrieval ——
见 DocumentChunk 的说明。document_chunk 上的唯一约束是
UNIQUE(document_version_id, chunker_key, chunker_config, chunk_index)。

因此共用同一个 chunker 的两路 retrieval 只会切一次、只落一份 chunk，
查询期两路命中同一块时拿到的是同一个 document_chunk_id。

本节点**按 chunker 身份扇出**（`fanout._chunking_plan` 按身份收敛），
因此共用 chunker 的几路只切一次。分支里带下来的 `chunker_digest`
既是收敛的依据，也是产物的键。

产物去向
--------
切分结果写进父状态的 chunk_refs 通道（按 **chunker 身份**归集），
**不是**留在本地状态 —— Send payload 只作用于本节点，
下游的 chunk_enrichment 看到的是父状态（见 fanout.py 的说明）。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langgraph.runtime import Runtime

from app.common.repository.factory import RepositoryFactory
from app.infra.graph import node_span

from app.services.document.chunker.identity import (
    resolve_chunker,
)
from app.services.document.domain.entities import DocumentChunk
from app.services.document.domain.schemas import ChunkRequest

from ...repository.document_chunk import DocumentChunkRepository
from ..catalog import NODE_SPECS
from ..context import DocumentPipelineContext
from ..state import BranchInput
from ._branch import branch_span_kwargs, ref_summary
from ._common import file_id_of, tenant_of

_SPEC = next(s for s in NODE_SPECS if s.node_id == "chunking")

__all__ = ["chunking_node"]


async def chunking_node(
        state: BranchInput,
        runtime: Runtime[DocumentPipelineContext],
) -> dict[str, Any]:
    """
    按本分支的 chunker 切分。

    Returns:
        {"chunk_refs": {retrieval_id: ArtifactRef}}
    """

    context = runtime.context

    branch = state["branch"]

    async with node_span(
            runtime,
            **branch_span_kwargs(_SPEC, state),
            input_summary={
                "retrieval_type": branch["retrieval_type"],
                "chunker_name": branch["chunker_name"],
                "clean_ref": ref_summary(state["clean_ref"]),
            },
    ) as span:

        # ----------------------------------------------------
        # 1. 取上游正文
        # ----------------------------------------------------

        text = await context.artifact_store.get_text(
            file_id_of(state["clean_ref"])
        ) or ""

        # ----------------------------------------------------
        # 2. 取 chunker 并切分
        # ----------------------------------------------------
        #
        # 顺序：**配置点名的优先，没有才按 role 取默认**。
        #
        # DocumentRetrievalDefinition.chunker_name 是每个 retrieval 各自
        # 的一份配置（不像 processor_name 要同时服务三个阶段），所以
        # 这里能干净地支持"点名" —— 前端在能力目录里选了哪个 chunker
        # （包括"不分块"那个），配置里就是哪个键，节点照单取用。
        #
        # 这也是把"不分块"做成一个能力项而不是 if 分支的价值：
        # 选择由配置表达，节点永远只做"按 key 取能力、调 process()"。
        # ----------------------------------------------------

        # 身份在计划阶段就算好了（见 metadata_extraction）—— 扇出与
        # 产物归集都要它，这里只是取用。
        digest = branch["chunker_digest"]

        chunker = resolve_chunker(
            context.registry, branch.get("chunker_name")
        )

        chunk_request = ChunkRequest(
            document_id=state["document_id"],
            document_version_id=state["document_version_id"],
            document_type_definition_id=state[
                "document_type_definition_id"
            ],
            text=text,
            metadata=dict(state.get("extract_ref") or {}),
            # branch 里这份配置在计划阶段已经归一化过（见
            # metadata_extraction），chunker 会把它原样落到
            # document_chunk.chunker_config —— 查询侧反查归属时用的
            # 是同一形式，两边必须一致。
            config=dict(branch.get("chunker_config") or {}),
        )

        result = await chunker.process(chunk_request)

        chunks: list[DocumentChunk] = list(result.chunks or [])

        # ----------------------------------------------------
        # 3. 落库
        # ----------------------------------------------------
        #
        # ⚠️ 这段用了 context.session_factory —— 而 context.py 的
        # "session_factory 使用纪律"目前只列了两种允许用法
        # （metadata_extraction 只读、finalize_version 原子写回）。
        # 这里是第三种：扇出节点写业务行。
        #
        # 之所以可以这么做：每个分支从**工厂**取自己的会话，不共享，
        # 因此不会踩 asyncpg 的并发使用限制。但那条纪律说明需要同步
        # 更新，否则下一个读 context.py 的人会以为这里越界了。
        # ----------------------------------------------------

        #: 上一次同身份的行数（重跑时非 0）。定义在 if 之外 ——
        #: 下面的埋点在两种情况下都要能读到它。
        removed = 0

        if context.session_factory is not None:
            # 每个分支从工厂取**自己的**会话，不共享 —— 扇出的 N 路是
            # 并发执行的，共享一个 session 会踩 asyncpg 的并发使用限制。
            #
            # scope() 默认开事务（这是写路径），退出时提交。
            repositories = RepositoryFactory(
                session_factory=context.session_factory
            )

            async with repositories.scope() as repos:

                chunk_repo = repos.create(DocumentChunkRepository)

                # --------------------------------------------------
                # 3a. 清掉上一次同身份的行（重跑路径）
                # --------------------------------------------------
                #
                # 重跑时这一步是**必需**的，不是保险：新 chunk 的
                # document_chunk_id 是新的 uuid，而 chunk_index 从 0
                # 重来，唯一约束
                # uq_document_chunk_version_chunker_index 会直接拒绝
                # 插入。删与插在同一个事务里 —— 中间不留"这个版本没有
                # chunk"的可见窗口（另一个请求读到空表会以为没切过）。
                #
                # 身份取**调用方给的那份配置**（branch 里这份在计划阶段
                # 已归一化，chunker 会把它原样存进 chunker_config，
                # 见 chunker/text.py 的 chunk()）—— 而不是从 chunks[0]
                # 里取，那样切出 0 条时就拿不到身份了，而"这次切出 0 条"
                # 恰恰是旧行**最该**被清掉的场景：留着它们，检索会继续
                # 命中一批这次运行已经不存在的 chunk。
                removed = await chunk_repo.delete_by_identity(
                    document_version_id=UUID(
                        state["document_version_id"]
                    ),
                    chunker_key=chunker.get_key(),
                    chunker_config=dict(
                        branch.get("chunker_config") or {}
                    ),
                )

                if chunks:
                    await chunk_repo.add_many(chunks)

        # ----------------------------------------------------
        # 4. 产物写出去，供阶段 8 取回（省一次回查）
        # ----------------------------------------------------

        import json

        payload = [
            {
                "document_chunk_id": str(c.document_chunk_id),
                # 下面三项是**身份**：阶段 8 要把条目还原成 DocumentChunk
                # 实体交给 enricher，而实体上这三个是必填（chunker_key /
                # chunker_config 还撑着 document_chunk 的唯一约束）。
                # 载荷里不带，下游就只能去查库 —— 那是扇出节点里再开
                # 一次事务，没必要。
                "document_version_id": str(c.document_version_id),
                "chunker_key": c.chunker_key,
                "chunker_config": c.chunker_config,
                "chunk_index": c.chunk_index,
                "content": c.content,
                "content_hash": c.content_hash,
                "char_count": c.char_count,
                "token_count": c.token_count,
                "start_index": c.start_index,
                "end_index": c.end_index,
                # locator 也要带上：chunk_enrichment 会往里面补 section /
                # document_title，最终进 ES 文档。
                "locator": dict(c.locator or {}),
                "metadata": c.metadata,
            }
            for c in chunks
        ]

        ref = await context.artifact_store.put_text(
            kind="chunks",
            text=json.dumps(payload, ensure_ascii=False),
            tenant_id=tenant_of(state),
            content_type="application/json",
        )

        span.set_metrics({
            "chunk_count": len(chunks),
            "source_char_count": len(text),
            "chunk_total_chars": sum(
                c.char_count or 0 for c in chunks
            ),
            "chunker_key": chunker.get_key(),
            # 重跑的可见证据：> 0 说明这次是覆盖上一次的产出。
            # 第一次跑恒为 0，所以它也是一个"这个版本被重跑过"的信号。
            "replaced_chunk_count": removed,
        })

        span.set_output_summary({
            # file_id 必须给 —— 见 embedding.py 里同样的说明。
            "ref": {
                "file_id": ref["file_id"],
                "checksum": ref["checksum"],
                "kind": ref["kind"],
                "char_count": ref["char_count"],
            },
            "chunker": chunker.get_key(),
        })

        # 键是 **chunker 身份**而不是 retrieval —— 切分产出是版本级的，
        # 共用 chunker 的几路共用这一份（见 fanout.py 的 _pool_key）。
        return {"chunk_refs": {digest: ref}}
