"""
PostgreSQL implementation of RetrievalEvalCaseRepository.

幂等的落点
----------
`add_cases` / `add_gold_chunks` 都走 `ON CONFLICT DO NOTHING`。这不是
"顺手用了个方便的写法" —— 它是"重试不产生重复"这条要求**唯一**可靠的
落点：两个 worker 同时重试同一个窗口时，应用层的"先查再插"两边都会查到
"没有"，然后都插。库上的唯一约束才排得掉这个竞态。

Transaction rule
----------------
This repository NEVER calls session.commit() / session.rollback().
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from ..domain.entities import (
    RetrievalEvalCase,
    RetrievalEvalCaseGoldChunk,
)
from ..domain.schemas.enum import EvalCaseValidationStatus
from ..models.sqlalchemy.retrieval_eval import (
    RetrievalEvalCase as RetrievalEvalCaseModel,
)
from ..models.sqlalchemy.retrieval_eval import (
    RetrievalEvalCaseGoldChunk as RetrievalEvalCaseGoldChunkModel,
)
from .retrieval_eval_case import RetrievalEvalCaseRepository

__all__ = ["RetrievalEvalCaseRepositoryImpl"]


#: 导出/预览的固定排序。
#:
#: ⚠️ 抽成常量而不是在两处各写一遍：导出物的字节稳定性（`dataset_sha256`
#: 建在它上面）与预览的"第 3 条"必须指向同一行。两处顺序一旦漂开，
#: 复核可疑 case 时的对应关系就断了，而那种错不会报错。
_ORDER_COLUMNS = (
    RetrievalEvalCaseModel.document_type_definition_id,
    RetrievalEvalCaseModel.source_chunk_id,
    RetrievalEvalCaseModel.question_order,
)


def _omit_unset_server_defaults(
        rows: list[dict],
        columns: tuple[str, ...] = ("created_at",),
) -> list[dict]:
    """
    把值为 None 的"服务端默认列"从 VALUES 里**删掉**，而不是留成 NULL。

    ⚠️ 这一步不是可选的
    -----------------
    ORM 路径（`session.add(model)`）会自动处理这件事：属性是 None 且列上
    有 server_default 时，SQLAlchemy 把该列从 INSERT 里省掉，于是
    `now()` 生效。

    但 **Core 的 `insert().values(rows)` 不这么做** —— 显式给出的 None
    就是字面量 NULL，直接撞 `NOT NULL` 约束。表现得像"服务端根本没建
    默认值"，而默认值明明就在模型上写着，很难往这个方向想。

    （这个 bug 是被 tests/services/document/test_retrieval_eval_repositories.py
    的真库用例抓出来的 —— 用假仓储测等于把约束抄一遍再断言它等于自己，
    抓不到。）
    """

    for row in rows:
        for column in columns:
            if row.get(column) is None:
                row.pop(column, None)

    return rows


class RetrievalEvalCaseRepositoryImpl(
    RetrievalEvalCaseRepository
):
    """PostgreSQL implementation of RetrievalEvalCaseRepository."""

    # ========================================================
    # 写入（幂等）
    # ========================================================

    async def add_cases(
            self,
            cases: list[RetrievalEvalCase],
    ) -> int:

        if not cases:
            return 0

        rows = [
            {
                "eval_case_id": case.eval_case_id,
                "bundle_id": case.bundle_id,
                "generation_window_id": case.generation_window_id,
                "document_type_definition_id": (
                    case.document_type_definition_id
                ),
                "document_version_id": case.document_version_id,
                "document_retrieval_definition_id": (
                    case.document_retrieval_definition_id
                ),
                "document_id": case.document_id,
                "source_document_title": case.document_title,
                "source_document_key": case.document_key,
                "source_chunk_id": case.source_chunk_id,
                "source_content_hash": case.source_content_hash,
                "question": case.question,
                "answer_original": case.answer_original,
                "answer_start_offset": case.answer_start_offset,
                "answer_end_offset": case.answer_end_offset,
                "question_order": case.question_order,
                "question_sha256": case.question_sha256,
                "validation_status": case.validation_status.value,
                "validation_error": case.validation_error,
                "created_at": case.created_at,
            }
            for case in cases
        ]

        rows = _omit_unset_server_defaults(rows)

        # ⚠️ `insert(...).on_conflict_do_nothing()` **不带 conflict 目标**
        # 时覆盖**全部**唯一约束 —— 正是这里想要的：
        #
        #     uq_retrieval_eval_case_order     (bundle, chunk, 题序)
        #     uq_retrieval_eval_case_question  (bundle, chunk, 问题指纹)
        #
        # 撞上任意一条都该跳过。指定单条目标的话，另一条撞上会直接抛
        # IntegrityError 把整个窗口的重试打挂 —— 而那正是我们想避免的。
        stmt = insert(RetrievalEvalCaseModel).values(rows)
        stmt = stmt.on_conflict_do_nothing()

        result = await self._session.execute(stmt)

        return result.rowcount or 0

    async def add_gold_chunks(
            self,
            gold_chunks: list[RetrievalEvalCaseGoldChunk],
    ) -> int:

        if not gold_chunks:
            return 0

        rows = [
            {
                "eval_case_gold_chunk_id": gold.eval_case_gold_chunk_id,
                "eval_case_id": gold.eval_case_id,
                "chunk_id": gold.chunk_id,
                "is_source_chunk": gold.is_source_chunk,
                "validation_method": gold.validation_method.value,
                "evidence_excerpt": gold.evidence_excerpt,
                "source_content_hash": gold.source_content_hash,
                "judgment": dict(gold.judgment),
                "created_at": gold.created_at,
            }
            for gold in gold_chunks
        ]

        rows = _omit_unset_server_defaults(rows)

        stmt = insert(RetrievalEvalCaseGoldChunkModel).values(rows)
        stmt = stmt.on_conflict_do_nothing()

        result = await self._session.execute(stmt)

        return result.rowcount or 0

    # ========================================================
    # 读取
    # ========================================================

    async def list_cases(
            self,
            *,
            bundle_id: UUID,
            document_type_definition_id: UUID | None = None,
            validation_status: EvalCaseValidationStatus | None = None,
            offset: int = 0,
            limit: int = 50,
    ) -> tuple[list[RetrievalEvalCase], int]:

        conditions = [
            RetrievalEvalCaseModel.bundle_id == bundle_id
        ]

        if document_type_definition_id is not None:
            conditions.append(
                RetrievalEvalCaseModel.document_type_definition_id
                == document_type_definition_id
            )

        if validation_status is not None:
            conditions.append(
                RetrievalEvalCaseModel.validation_status
                == validation_status.value
            )

        count_stmt = (
            select(func.count())
            .select_from(RetrievalEvalCaseModel)
            .where(*conditions)
        )

        total = int(
            (await self._session.execute(count_stmt)).scalar_one()
        )

        page_stmt = (
            select(RetrievalEvalCaseModel)
            .where(*conditions)
            .order_by(*_ORDER_COLUMNS)
            .offset(offset)
            .limit(limit)
        )

        models = (
            await self._session.execute(page_stmt)
        ).scalars().all()

        return [_to_case(model) for model in models], total

    async def list_all_validated_cases(
            self,
            bundle_id: UUID,
    ) -> list[RetrievalEvalCase]:

        stmt = (
            select(RetrievalEvalCaseModel)
            .where(
                RetrievalEvalCaseModel.bundle_id == bundle_id,
                RetrievalEvalCaseModel.validation_status
                == EvalCaseValidationStatus.VALIDATED.value,
            )
            .order_by(*_ORDER_COLUMNS)
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [_to_case(model) for model in models]

    async def count_cases(
            self,
            bundle_id: UUID,
            *,
            validation_status: EvalCaseValidationStatus | None = None,
    ) -> int:

        conditions = [
            RetrievalEvalCaseModel.bundle_id == bundle_id
        ]

        if validation_status is not None:
            conditions.append(
                RetrievalEvalCaseModel.validation_status
                == validation_status.value
            )

        stmt = (
            select(func.count())
            .select_from(RetrievalEvalCaseModel)
            .where(*conditions)
        )

        return int(
            (await self._session.execute(stmt)).scalar_one()
        )

    async def list_gold_chunks(
            self,
            eval_case_ids: list[UUID],
    ) -> dict[UUID, list[RetrievalEvalCaseGoldChunk]]:

        if not eval_case_ids:
            return {}

        stmt = (
            select(RetrievalEvalCaseGoldChunkModel)
            .where(
                RetrievalEvalCaseGoldChunkModel.eval_case_id.in_(
                    eval_case_ids
                )
            )
            .order_by(
                RetrievalEvalCaseGoldChunkModel.eval_case_id,
                # 源 chunk 排在最前：导出与预览里"出题的那一块"应当是
                # 第一个被看到的 —— 它是构造出来的事实，可信度最高。
                RetrievalEvalCaseGoldChunkModel
                .is_source_chunk
                .desc(),
                RetrievalEvalCaseGoldChunkModel.chunk_id,
            )
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        grouped: dict[UUID, list[RetrievalEvalCaseGoldChunk]] = {}

        for model in models:
            grouped.setdefault(model.eval_case_id, []).append(
                _to_gold_chunk(model)
            )

        return grouped

    async def get_many(
            self,
            eval_case_ids: list[UUID],
    ) -> list[RetrievalEvalCase]:

        if not eval_case_ids:
            return []

        stmt = select(RetrievalEvalCaseModel).where(
            RetrievalEvalCaseModel.eval_case_id.in_(eval_case_ids)
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [_to_case(model) for model in models]

    async def count_distinct_source_chunks(
            self,
            bundle_id: UUID,
    ) -> int:

        stmt = (
            select(
                func.count(
                    func.distinct(
                        RetrievalEvalCaseModel.source_chunk_id
                    )
                )
            )
            .select_from(RetrievalEvalCaseModel)
            .where(RetrievalEvalCaseModel.bundle_id == bundle_id)
        )

        return int(
            (await self._session.execute(stmt)).scalar_one()
        )


# ============================================================
# Mapping
# ============================================================


def _to_case(model: RetrievalEvalCaseModel) -> RetrievalEvalCase:
    """ORM -> 实体。不做任何业务判断。"""

    return RetrievalEvalCase(
        eval_case_id=model.eval_case_id,
        bundle_id=model.bundle_id,
        generation_window_id=model.generation_window_id,
        document_type_definition_id=(
            model.document_type_definition_id
        ),
        document_version_id=model.document_version_id,
        document_retrieval_definition_id=(
            model.document_retrieval_definition_id
        ),
        document_id=model.document_id,
        document_title=model.source_document_title,
        document_key=model.source_document_key,
        source_chunk_id=model.source_chunk_id,
        source_content_hash=model.source_content_hash,
        question=model.question,
        answer_original=model.answer_original,
        answer_start_offset=model.answer_start_offset,
        answer_end_offset=model.answer_end_offset,
        question_order=model.question_order,
        question_sha256=model.question_sha256,
        validation_status=model.validation_status,
        validation_error=model.validation_error,
        created_at=model.created_at,
    )


def _to_gold_chunk(
        model: RetrievalEvalCaseGoldChunkModel,
) -> RetrievalEvalCaseGoldChunk:

    return RetrievalEvalCaseGoldChunk(
        eval_case_gold_chunk_id=model.eval_case_gold_chunk_id,
        eval_case_id=model.eval_case_id,
        chunk_id=model.chunk_id,
        is_source_chunk=model.is_source_chunk,
        validation_method=model.validation_method,
        evidence_excerpt=model.evidence_excerpt,
        source_content_hash=model.source_content_hash,
        judgment=dict(model.judgment or {}),
        created_at=model.created_at,
    )
