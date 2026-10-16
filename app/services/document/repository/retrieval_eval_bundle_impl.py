"""
PostgreSQL implementation of RetrievalEvalBundleRepository.

Non-responsibilities
--------------------
- Transaction commit / rollback
- 抽样、切窗口、调模型（那些在 eval/ 与 application/ 层）
- "ready 之后不许改"这条业务规则（在 Service 层，见接口的说明）

Transaction rule
----------------
This repository NEVER calls session.commit() / session.rollback().
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select, update

from ..domain.entities import (
    RetrievalEvalBundle,
    RetrievalEvalGenerationRun,
    RetrievalEvalGenerationWindow,
)
from ..domain.schemas.enum import EvalBundleStatus, EvalJobStatus
from ..models.sqlalchemy.retrieval_eval import (
    RetrievalEvalBundle as RetrievalEvalBundleModel,
)
from ..models.sqlalchemy.retrieval_eval import (
    RetrievalEvalGenerationRun as RetrievalEvalGenerationRunModel,
)
from ..models.sqlalchemy.retrieval_eval import (
    RetrievalEvalGenerationWindow as RetrievalEvalGenerationWindowModel,
)
from .retrieval_eval_bundle import RetrievalEvalBundleRepository

__all__ = ["RetrievalEvalBundleRepositoryImpl"]


#: "还没成功"的窗口状态 —— 也就是**可以被认领去跑**的那些。
#:
#: ⚠️ `queued` 在内，见接口 `list_unfinished_windows` 的说明：
#: 一次执行有窗口数上限，跑不完的留在 `queued`，再提交一次就继续。
#: `running` / `completed` / `cancelled` 都不在内 —— 前者正在被别的执行
#: 处理，后两者已经结束。
_UNFINISHED_WINDOW_STATUSES = (
    EvalJobStatus.QUEUED.value,
    EvalJobStatus.FAILED.value,
    EvalJobStatus.PARTIAL_FAILED.value,
)


class RetrievalEvalBundleRepositoryImpl(
    RetrievalEvalBundleRepository
):
    """PostgreSQL implementation of RetrievalEvalBundleRepository."""

    # ========================================================
    # Bundle
    # ========================================================

    async def add_bundle(
            self,
            bundle: RetrievalEvalBundle,
    ) -> RetrievalEvalBundle:

        model = RetrievalEvalBundleModel(
            bundle_id=bundle.bundle_id,
            tenant_id=bundle.tenant_id,
            name=bundle.name,
            description=bundle.description,
            status=bundle.status,
            schema_version=bundle.schema_version,
            selection_config=dict(bundle.selection_config),
            generation_config=dict(bundle.generation_config),
            dataset_sha256=bundle.dataset_sha256,
            source_chunk_count=bundle.source_chunk_count,
            case_count=bundle.case_count,
            created_by=bundle.created_by,
            created_at=bundle.created_at,
            updated_at=bundle.updated_at,
            completed_at=bundle.completed_at,
        )

        self._session.add(model)

        # 让 INSERT 在**当前事务**内可见。不 commit。
        await self._session.flush()

        return _to_bundle(model)

    async def get_bundle(
            self,
            bundle_id: UUID,
    ) -> RetrievalEvalBundle | None:

        stmt = select(RetrievalEvalBundleModel).where(
            RetrievalEvalBundleModel.bundle_id == bundle_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        return None if model is None else _to_bundle(model)

    async def update_bundle(
            self,
            bundle: RetrievalEvalBundle,
    ) -> RetrievalEvalBundle:

        model = await self._require_bundle(bundle.bundle_id)

        # 可变字段。`bundle_id` / `created_at` / `created_by` 不更新
        # —— 它们是身份与出处，不是状态。
        model.name = bundle.name
        model.description = bundle.description
        model.status = bundle.status
        model.schema_version = bundle.schema_version
        model.selection_config = dict(bundle.selection_config)
        model.generation_config = dict(bundle.generation_config)
        model.dataset_sha256 = bundle.dataset_sha256
        model.source_chunk_count = bundle.source_chunk_count
        model.case_count = bundle.case_count
        model.completed_at = bundle.completed_at

        # ⚠️ 这一行**不能省**，它不是"顺手同步一下"。
        #
        # `updated_at` 列上有 `onupdate=func.now()`。不显式赋值时，SQLAlchemy
        # 让**数据库**去算它，于是这个属性在 flush 之后处于 **expired** 状态
        # —— 而紧接着 `_to_bundle(model)` 要读它，读一个 expired 属性会触发
        # 一次**懒加载 SELECT**。在 asyncio 里那次 IO 发生在同步上下文，
        # 结果是：
        #
        #     sqlalchemy.exc.MissingGreenlet: greenlet_spawn has not been
        #     called; can't call await_only() here.
        #
        # 报错点在 `_to_bundle` 读 `model.updated_at` 那一行，而**原因**在
        # 这里少了一行赋值 —— 隔着一次 flush，很难往这个方向想。
        #
        # 显式赋值同时解决两件事：写进去的是带时区的 `now()`（与
        # `created_at` 同一口径，见服务层的 `_now()`），且属性不再是
        # expired，返回值不必再回库。
        #
        # 这也是本仓库既有仓储的写法（见 type_definition_impl.update：
        # `model.updated_at = definition.updated_at`）。**服务层负责盖章，
        # 仓储负责赋值。**
        model.updated_at = bundle.updated_at

        await self._session.flush()

        return _to_bundle(model)

    async def list_bundles(
            self,
            *,
            tenant_id: UUID | None = None,
            status: EvalBundleStatus | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[RetrievalEvalBundle], int]:

        conditions = []

        if tenant_id is not None:
            conditions.append(
                RetrievalEvalBundleModel.tenant_id == tenant_id
            )

        if status is not None:
            conditions.append(
                RetrievalEvalBundleModel.status == status.value
            )

        count_stmt = (
            select(func.count())
            .select_from(RetrievalEvalBundleModel)
        )

        if conditions:
            count_stmt = count_stmt.where(*conditions)

        total = int(
            (await self._session.execute(count_stmt)).scalar_one()
        )

        page_stmt = select(RetrievalEvalBundleModel)

        if conditions:
            page_stmt = page_stmt.where(*conditions)

        page_stmt = (
            page_stmt
            .order_by(
                RetrievalEvalBundleModel.created_at.desc(),
                RetrievalEvalBundleModel.bundle_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        models = (
            await self._session.execute(page_stmt)
        ).scalars().all()

        return [_to_bundle(model) for model in models], total

    # ========================================================
    # Generation run
    # ========================================================

    async def add_generation_run(
            self,
            run: RetrievalEvalGenerationRun,
    ) -> RetrievalEvalGenerationRun:

        model = RetrievalEvalGenerationRunModel(
            generation_run_id=run.generation_run_id,
            bundle_id=run.bundle_id,
            status=run.status,
            sample_seed=run.sample_seed,
            config_snapshot=dict(run.config_snapshot),
            model_role=run.model_role,
            model_name=run.model_name,
            total_chunks=run.total_chunks,
            total_windows=run.total_windows,
            completed_windows=run.completed_windows,
            failed_windows=run.failed_windows,
            generated_cases=run.generated_cases,
            error_message=run.error_message,
            started_at=run.started_at,
            completed_at=run.completed_at,
            created_at=run.created_at,
        )

        self._session.add(model)
        await self._session.flush()

        return _to_generation_run(model)

    async def get_generation_run(
            self,
            generation_run_id: UUID,
    ) -> RetrievalEvalGenerationRun | None:

        stmt = select(RetrievalEvalGenerationRunModel).where(
            RetrievalEvalGenerationRunModel.generation_run_id
            == generation_run_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        return None if model is None else _to_generation_run(model)

    async def get_latest_generation_run(
            self,
            bundle_id: UUID,
    ) -> RetrievalEvalGenerationRun | None:

        stmt = (
            select(RetrievalEvalGenerationRunModel)
            .where(
                RetrievalEvalGenerationRunModel.bundle_id == bundle_id
            )
            .order_by(
                RetrievalEvalGenerationRunModel.created_at.desc(),
                RetrievalEvalGenerationRunModel
                .generation_run_id
                .desc(),
            )
            .limit(1)
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        return None if model is None else _to_generation_run(model)

    async def list_latest_generation_runs(
            self,
            bundle_ids: list[UUID],
    ) -> dict[UUID, RetrievalEvalGenerationRun]:

        if not bundle_ids:
            return {}

        # DISTINCT ON 是 PostgreSQL 专有的，而本模块已经在用 JSONB 与
        # UUID 类型 —— 绑定 PG 不是新引入的约束。
        #
        # 为什么用它而不是"先查最大 created_at 再 join 回去"：
        # created_at 可能重复（同一毫秒内建两次运行），那时 join 会返回
        # 两行，而"最近一次"就没了确定答案。DISTINCT ON 配一个到
        # generation_run_id 的 tiebreaker 是确定的。
        stmt = (
            select(RetrievalEvalGenerationRunModel)
            .where(
                RetrievalEvalGenerationRunModel.bundle_id.in_(bundle_ids)
            )
            .distinct(
                RetrievalEvalGenerationRunModel.bundle_id,
            )
            .order_by(
                RetrievalEvalGenerationRunModel.bundle_id,
                RetrievalEvalGenerationRunModel.created_at.desc(),
                RetrievalEvalGenerationRunModel
                .generation_run_id
                .desc(),
            )
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return {
            model.bundle_id: _to_generation_run(model)
            for model in models
        }

    async def update_generation_run(
            self,
            run: RetrievalEvalGenerationRun,
    ) -> RetrievalEvalGenerationRun:

        model = await self._require_generation_run(run.generation_run_id)

        model.status = run.status
        model.model_name = run.model_name
        model.config_snapshot = dict(run.config_snapshot)
        model.total_chunks = run.total_chunks
        model.total_windows = run.total_windows
        model.completed_windows = run.completed_windows
        model.failed_windows = run.failed_windows
        model.generated_cases = run.generated_cases
        model.error_message = run.error_message
        model.started_at = run.started_at
        model.completed_at = run.completed_at

        # `sample_seed` / `model_role` / `bundle_id` / `created_at` 不更新：
        # 前三个是这次运行的**身份**（复现的钥匙与它调的是谁），改它们
        # 等于把一次运行伪装成另一次。

        await self._session.flush()

        return _to_generation_run(model)

    async def count_windows_by_status(
            self,
            generation_run_id: UUID,
    ) -> dict[str, int]:

        stmt = (
            select(
                RetrievalEvalGenerationWindowModel.status,
                func.count(),
            )
            .where(
                RetrievalEvalGenerationWindowModel.generation_run_id
                == generation_run_id
            )
            .group_by(RetrievalEvalGenerationWindowModel.status)
        )

        rows = await self._session.execute(stmt)

        return {
            str(status): int(count)
            for status, count in rows.all()
        }

    # ========================================================
    # Generation window
    # ========================================================

    async def add_windows(
            self,
            windows: list[RetrievalEvalGenerationWindow],
    ) -> int:

        if not windows:
            return 0

        models = [
            RetrievalEvalGenerationWindowModel(
                generation_window_id=window.generation_window_id,
                generation_run_id=window.generation_run_id,
                window_index=window.window_index,
                status=window.status,
                input_chunks=list(window.input_chunks),
                input_token_estimate=window.input_token_estimate,
                request_sha256=window.request_sha256,
                raw_response=window.raw_response,
                attempt_count=window.attempt_count,
                valid_case_count=window.valid_case_count,
                error_code=window.error_code,
                error_message=window.error_message,
                started_at=window.started_at,
                completed_at=window.completed_at,
                created_at=window.created_at,
            )
            for window in windows
        ]

        self._session.add_all(models)
        await self._session.flush()

        return len(models)

    async def list_windows(
            self,
            generation_run_id: UUID,
    ) -> list[RetrievalEvalGenerationWindow]:

        stmt = (
            select(RetrievalEvalGenerationWindowModel)
            .where(
                RetrievalEvalGenerationWindowModel.generation_run_id
                == generation_run_id
            )
            .order_by(
                RetrievalEvalGenerationWindowModel.window_index,
            )
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [_to_window(model) for model in models]

    async def list_unfinished_windows(
            self,
            generation_run_id: UUID,
    ) -> list[RetrievalEvalGenerationWindow]:

        stmt = (
            select(RetrievalEvalGenerationWindowModel)
            .where(
                RetrievalEvalGenerationWindowModel.generation_run_id
                == generation_run_id,
                RetrievalEvalGenerationWindowModel.status.in_(
                    _UNFINISHED_WINDOW_STATUSES
                ),
            )
            .order_by(
                RetrievalEvalGenerationWindowModel.window_index,
            )
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [_to_window(model) for model in models]

    async def claim_window(
            self,
            generation_window_id: UUID,
    ) -> bool:

        stmt = (
            update(RetrievalEvalGenerationWindowModel)
            .where(
                RetrievalEvalGenerationWindowModel.generation_window_id
                == generation_window_id,
                RetrievalEvalGenerationWindowModel.status.in_(
                    _UNFINISHED_WINDOW_STATUSES
                ),
            )
            .values(
                status=EvalJobStatus.RUNNING.value,
                # 认领即算一次尝试。放在这里而不是调用方，
                # 是因为"认领成功"与"尝试次数 +1"必须是同一件事 ——
                # 分开写的话，认领失败的那次也会把计数加上去。
                attempt_count=(
                    RetrievalEvalGenerationWindowModel.attempt_count + 1
                ),
            )
        )

        result = await self._session.execute(stmt)

        return bool(result.rowcount)

    async def get_window(
            self,
            generation_window_id: UUID,
    ) -> RetrievalEvalGenerationWindow | None:

        stmt = select(RetrievalEvalGenerationWindowModel).where(
            RetrievalEvalGenerationWindowModel.generation_window_id
            == generation_window_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        return None if model is None else _to_window(model)

    async def update_window(
            self,
            window: RetrievalEvalGenerationWindow,
    ) -> RetrievalEvalGenerationWindow:

        stmt = select(RetrievalEvalGenerationWindowModel).where(
            RetrievalEvalGenerationWindowModel.generation_window_id
            == window.generation_window_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            raise ValueError(
                "RetrievalEvalGenerationWindow not found: "
                f"{window.generation_window_id}"
            )

        # ⚠️ `input_chunks` 与 `window_index` **刻意不更新** ——
        # 它们是这个窗口的身份，重试靠它们复现当时的输入。
        # 要换输入就是另一个窗口。
        model.status = window.status
        model.request_sha256 = window.request_sha256
        model.raw_response = window.raw_response
        model.attempt_count = window.attempt_count
        model.valid_case_count = window.valid_case_count
        model.input_token_estimate = window.input_token_estimate
        model.error_code = window.error_code
        model.error_message = window.error_message
        model.started_at = window.started_at
        model.completed_at = window.completed_at

        await self._session.flush()

        return _to_window(model)

    # ========================================================
    # internals
    # ========================================================

    async def _require_bundle(
            self,
            bundle_id: UUID,
    ) -> RetrievalEvalBundleModel:

        stmt = select(RetrievalEvalBundleModel).where(
            RetrievalEvalBundleModel.bundle_id == bundle_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            raise ValueError(
                f"RetrievalEvalBundle not found: {bundle_id}"
            )

        return model

    async def _require_generation_run(
            self,
            generation_run_id: UUID,
    ) -> RetrievalEvalGenerationRunModel:

        stmt = select(RetrievalEvalGenerationRunModel).where(
            RetrievalEvalGenerationRunModel.generation_run_id
            == generation_run_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            raise ValueError(
                "RetrievalEvalGenerationRun not found: "
                f"{generation_run_id}"
            )

        return model


# ============================================================
# Mapping
# ============================================================


def _to_bundle(
        model: RetrievalEvalBundleModel,
) -> RetrievalEvalBundle:
    """ORM -> 实体。不做任何业务判断。"""

    return RetrievalEvalBundle(
        bundle_id=model.bundle_id,
        tenant_id=model.tenant_id,
        name=model.name,
        description=model.description,
        status=model.status,
        schema_version=model.schema_version,
        selection_config=dict(model.selection_config or {}),
        generation_config=dict(model.generation_config or {}),
        dataset_sha256=model.dataset_sha256,
        source_chunk_count=model.source_chunk_count,
        case_count=model.case_count,
        created_by=model.created_by,
        created_at=model.created_at,
        updated_at=model.updated_at,
        completed_at=model.completed_at,
    )


def _to_generation_run(
        model: RetrievalEvalGenerationRunModel,
) -> RetrievalEvalGenerationRun:

    return RetrievalEvalGenerationRun(
        generation_run_id=model.generation_run_id,
        bundle_id=model.bundle_id,
        status=model.status,
        sample_seed=model.sample_seed,
        config_snapshot=dict(model.config_snapshot or {}),
        model_role=model.model_role,
        model_name=model.model_name,
        total_chunks=model.total_chunks,
        total_windows=model.total_windows,
        completed_windows=model.completed_windows,
        failed_windows=model.failed_windows,
        generated_cases=model.generated_cases,
        error_message=model.error_message,
        started_at=model.started_at,
        completed_at=model.completed_at,
        created_at=model.created_at,
    )


def _to_window(
        model: RetrievalEvalGenerationWindowModel,
) -> RetrievalEvalGenerationWindow:

    return RetrievalEvalGenerationWindow(
        generation_window_id=model.generation_window_id,
        generation_run_id=model.generation_run_id,
        window_index=model.window_index,
        status=model.status,
        input_chunks=[dict(chunk) for chunk in (model.input_chunks or [])],
        input_token_estimate=model.input_token_estimate,
        request_sha256=model.request_sha256,
        raw_response=model.raw_response,
        attempt_count=model.attempt_count,
        valid_case_count=model.valid_case_count,
        error_code=model.error_code,
        error_message=model.error_message,
        started_at=model.started_at,
        completed_at=model.completed_at,
        created_at=model.created_at,
    )
