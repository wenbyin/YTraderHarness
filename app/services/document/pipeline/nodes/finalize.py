"""
扇入：finalize_version（控制节点，不属于 10 个阶段之一）。

职责
----
所有扇出分支汇合后跑**一次**，做两件事：

    1. 按 retrieval_results 逐条 upsert
       DocumentVersionRetrievalBinding（业务状态，每 (version,
       retrieval) 一行）
    2. 从这些绑定聚合出 DocumentVersion.processing_status 与
       completed_at

这是**唯一**一处把执行结果写回业务状态的地方。

单向派生
--------
观测表（pipeline_run / pipeline_node_run）永不喂业务状态；
binding 表永不携带耗时与历史。两者只有这一条派生边：
执行结果 -> binding -> version.processing_status。

partial 如何映射
----------------
运行状态可能是 partial（部分 retrieval 成功），但业务状态
只有 pending/processing/completed/failed —— 业务要回答的是
"这个版本能不能用"，没有"部分能用"。

映射规则见 PipelineRunRepository.map_to_business_status：
    partial -> failed

这是刻意的：宁可让调用方知道"这版没准备好"，
也不要让它在一次检索里才发现少了一半内容。
部分可用的细节通过观测接口呈现，不塞进业务状态。
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.common.repository.factory import RepositoryFactory
from app.infra.graph import node_span

from ...repository.document import DocumentRepository
from ...repository.document_version import DocumentVersionRepository
from ...repository.document_version_retrieval_binding import (
    DocumentVersionRetrievalBindingRepository,
)
from ..catalog import NODE_SPECS
from ..context import DocumentPipelineContext
from ..state import DocumentPipelineState

_SPEC = next(
    s for s in NODE_SPECS if s.node_id == "finalize_version"
)

__all__ = ["finalize_version_node"]


async def finalize_version_node(
        state: DocumentPipelineState,
        runtime: Runtime[DocumentPipelineContext],
) -> dict[str, Any]:
    """
    汇总各分支结果并回写业务状态。

    输入是**父状态**（所有分支合并后），不是 Send payload ——
    finalize 是经普通边从 indexing 收敛过来的。
    """

    context = runtime.context

    results = list(state.get("retrieval_results") or [])

    expected = len(state.get("branches") or [])

    async with node_span(
            runtime,
            node_id=_SPEC.node_id,
            label=_SPEC.label,
            group=_SPEC.group,
            sequence=_SPEC.sequence,
            input_summary={
                "result_count": len(results),
                "expected_branch_count": expected,
            },
    ) as span:

        # 有分支没回传结果 = 该分支在扇出中途失败。
        # 这不是错误，是 partial 的来源，记为指标供前端解释。
        span.set_metrics({
            "completed": sum(
                1 for r in results
                if r.get("status") == "completed"
            ),
            "failed": sum(
                1 for r in results
                if r.get("status") == "failed"
            ),
            "missing": max(expected - len(results), 0),
        })

        # ----------------------------------------------------
        # 回写业务状态。
        #
        # 这是**唯一**允许节点开事务的地方：绑定行的多行写入与
        # 版本状态必须同事务，否则会出现"绑定写了、版本状态没更新"
        # 的中间态，前端看到的进度就是错的。
        # 见 context.py 的 session_factory 使用纪律。
        # ----------------------------------------------------

        from datetime import datetime, timezone
        from uuid import UUID, uuid4

        from ...domain.entities import (
            DocumentVersionRetrievalBinding,
        )
        from ...domain.schemas.enum import ProcessingStatus
        context = runtime.context

        if context.session_factory is None:
            raise RuntimeError(
                "finalize_version needs session_factory to write "
                "business state, but the pipeline context has none."
            )

        version_id = UUID(state["document_version_id"])
        now = datetime.now(timezone.utc)

        by_retrieval = {
            r["document_retrieval_definition_id"]: r
            for r in results
        }

        repositories = RepositoryFactory(
            session_factory=context.session_factory
        )

        # 三个仓储都从**同一个** scope 取，因此共享同一个 session 与
        # 同一个事务 —— 绑定行的多行写入与版本状态必须一起生效，
        # 否则会出现"绑定写了、版本状态没更新"的中间态（见上）。
        async with repositories.scope() as repos:

            doc_repo = repos.create(DocumentRepository)

            binding_repo = repos.create(
                DocumentVersionRetrievalBindingRepository
            )

            for branch in state.get("branches") or []:
                rid = branch[
                    "document_retrieval_definition_id"
                ]

                result = by_retrieval.get(rid)

                # 没有回传结果 = 该分支在扇出中途失败/被跳过。
                # **必须**也写一行 failed，否则绑定表少一行，
                # 聚合出的版本状态会偏乐观。
                await binding_repo.upsert(
                    DocumentVersionRetrievalBinding(
                        document_version_retrieval_binding_id=(
                            uuid4()
                        ),
                        document_version_id=version_id,
                        document_retrieval_definition_id=(
                            UUID(rid)
                        ),
                        document_index_binding_id=(
                            None
                            if result is None
                               or not result.get(
                                "document_index_binding_id"
                            )
                            else UUID(
                                result[
                                    "document_index_binding_id"
                                ]
                            )
                        ),
                        processing_status=(
                            ProcessingStatus.FAILED
                            if result is None
                            else ProcessingStatus(
                                result["status"]
                            )
                        ),
                        error_message=(
                            "Branch produced no result."
                            if result is None
                            else result.get("error_message")
                        ),
                        created_at=now,
                        updated_at=now,
                    )
                )

            bindings = await binding_repo.list_by_version(
                version_id
            )

            status = (
                DocumentVersionRetrievalBindingRepository
                .aggregate_status(bindings)
            )

            version_repo = repos.create(DocumentVersionRepository)
            version = await version_repo.get(version_id)

            superseded_count = 0

            if version is not None:

                version.processing_status = status

                version.completed_at = (
                    now
                    if status
                       == ProcessingStatus.COMPLETED
                    else None
                )

                version.updated_at = now
                await version_repo.update(version)

                # ----------------------------------------
                # 全部成功 -> 这个版本成为文档的生效版本，
                # 其余仍在生效的版本让位。
                # ----------------------------------------
                #
                # 聚合状态是 COMPLETED 才算成功 —— 它由 binding 逐行
                # 聚合而来，partial 会映射成 failed（见模块 docstring），
                # 所以这里不必再单独判 partial。
                #
                # 让位规则（只动 active 的那些、不碰 deleted/failed/
                # pending）写在 DocumentVersionRepository.activate 上 ——
                # update_document 切版本时走的是同一个方法，两处必须
                # 一致，因此不做第二份实现。
                #
                # 仍在**同一个事务**里：版本转 active 与旧版本让位
                # 必须一起生效，否则中途崩溃会留下"两个 active"或
                # "零个 active"的中间态。
                # ----------------------------------------

                if status == ProcessingStatus.COMPLETED:
                    superseded_count = await version_repo.activate(
                        document_id=UUID(state["document_id"]),
                        version_id=version_id,
                        now=now,
                    )

            document = await doc_repo.get(
                state["document_id"]
            )

            document.current_document_version_id = (
                state["document_version_id"]
            )
            document.updated_at = now

            await doc_repo.update(
                document
            )

        span.set_output_summary({
            "aggregated_status": status.value,
            "binding_count": len(bindings),
            "missing_branch_count": max(
                expected - len(results), 0
            ),
            "version_activated": (
                status == ProcessingStatus.COMPLETED
            ),
            "superseded_version_count": superseded_count,
        })

        return {"status": status.value}
