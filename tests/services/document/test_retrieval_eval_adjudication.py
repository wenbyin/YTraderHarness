"""
跨 chunk 判金标：**候选过滤**与**成本保证**。

为什么这两条值得单独焊住
----------------------
判金标是**逐题一次模型调用**。而它的候选池过滤（只取**同一份文档**里的
其它块）决定了调用次数：

    随机跨公司抽样（20 块抽自 100 家）→ 同文档候选通常为空 → **零调用**
    一份文档贡献多块          → 有候选 → 每题一次调用

那个过滤一旦被改动或绕过，成本会**静默涨一个量级**：题库照出、结果照对，
只是每一道题都去问一次模型。没有任何症状。

所以这里用假 runtime 数**调用次数** —— 不是断言"结果对不对"。

⚠️ 用假 runtime，不打真模型：判定的语义由 adjudication.py 的提示词与它
自己的机械校验负责（evidence_excerpt 必须在候选正文里逐字找得到），
那部分与这里要验的"该不该调、调几次"无关。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
import yaml
from langchain_core.messages import AIMessage
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.common.models import Base
from app.services.document.domain.entities import (
    RetrievalEvalBundle,
    RetrievalEvalCase,
    RetrievalEvalGenerationRun,
    RetrievalEvalGenerationWindow,
)
from app.services.document.repository.retrieval_eval_bundle_impl import (
    RetrievalEvalBundleRepositoryImpl,
)
from app.services.document.repository.retrieval_eval_case_impl import (
    RetrievalEvalCaseRepositoryImpl,
)

POSTGRE_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "app" / "config" / "datasource" / "postgre.yaml"
)

#: 候选正文里逐字存在的证据（`evidence_excerpt` 的机械校验要求它）。
_EVIDENCE = "经审计的净资产为 12.3 亿元"

_QUESTION = "特锐德 截至 2007-12-31 经审计的净资产是多少？"


class _FakeRuntime:
    """记下被调用了几次，返回一份预设的判定。"""

    def __init__(self, verdict: dict) -> None:
        self._verdict = verdict
        self.requests: list[object] = []

    async def ainvoke(self, request: object) -> object:
        self.requests.append(request)
        return SimpleNamespace(
            message=AIMessage(
                content=json.dumps(self._verdict, ensure_ascii=False)
            )
        )


@pytest.fixture
async def session():
    """真库会话（用完回滚）。与 test_retrieval_eval_repositories 同一形状。"""

    from app.infra.datasource.initializer import import_all_models

    import_all_models()

    dsn = yaml.safe_load(
        POSTGRE_CONFIG.read_text(encoding="utf-8")
    )["config"]["dsn"]

    engine = create_async_engine(dsn)

    try:
        connection = await engine.connect()
        await connection.close()

    except Exception as exc:  # noqa: BLE001
        await engine.dispose()
        pytest.skip(f"连不上本地 Postgres，跳过判金标回归：{exc}")

    session = AsyncSession(engine, expire_on_commit=False)

    try:
        tables = [
            table
            for name, table in Base.metadata.tables.items()
            if name.startswith("retrieval_eval")
        ]

        connection = await session.connection()
        await connection.run_sync(
            Base.metadata.create_all, tables=tables,
        )

        yield session

    finally:
        await session.rollback()
        await session.close()
        await engine.dispose()


def _service(session: AsyncSession):
    """直接造服务（不经过 lifespan），仓储由工厂按注解注入。"""

    from app.common.repository.factory import RepositoryFactory
    from app.common.service.factory import ServiceFactory
    from app.services.document.application.retrieval_eval_service import (
        RetrievalEvalService,
    )

    return ServiceFactory(
        repositories=RepositoryFactory(session=session)
    ).create(RetrievalEvalService)


async def _fixture(session: AsyncSession):
    """
    三块来自**两份**文档的 chunk：

        c1, c2  同一份文档（特锐德）→ c1 出题时 c2 是候选
        c3      另一份文档（万科A）→ 出题时**没有候选**
    """

    service = _service(session)

    document_a = str(uuid.uuid4())
    document_b = str(uuid.uuid4())
    chunk_a1, chunk_a2, chunk_b1 = (
        str(uuid.uuid4()) for _ in range(3)
    )

    bundle = RetrievalEvalBundle(
        bundle_id=uuid.uuid4(),
        name="判金标回归",
        selection_config={},
        generation_config={},
    )

    await service._bundles.add_bundle(bundle)

    run = RetrievalEvalGenerationRun(
        generation_run_id=uuid.uuid4(),
        bundle_id=bundle.bundle_id,
        sample_seed=1,
    )

    await service._bundles.add_generation_run(run)

    await service._bundles.add_windows([
        RetrievalEvalGenerationWindow(
            generation_window_id=uuid.uuid4(),
            generation_run_id=run.generation_run_id,
            window_index=0,
            input_chunks=[
                {
                    "chunk_id": chunk_a1,
                    "document_id": document_a,
                    "document_title": "特锐德",
                    "document_key": "300001.SZ",
                    "content": f"本公司{_EVIDENCE}。",
                    "content_hash": "hash-a1",
                },
                {
                    "chunk_id": chunk_a2,
                    "document_id": document_a,
                    "document_title": "特锐德",
                    "document_key": "300001.SZ",
                    "content": f"该公司{_EVIDENCE}。",
                    "content_hash": "hash-a2",
                },
                {
                    "chunk_id": chunk_b1,
                    "document_id": document_b,
                    "document_title": "万科A",
                    "document_key": "000002.SZ",
                    "content": "本公司经审计的净资产为 900 亿元。",
                    "content_hash": "hash-b1",
                },
            ],
        )
    ])

    def _case(chunk_id: str, document_id: str) -> RetrievalEvalCase:

        return RetrievalEvalCase(
            eval_case_id=uuid.uuid4(),
            bundle_id=bundle.bundle_id,
            document_type_definition_id=uuid.uuid4(),
            source_chunk_id=UUID(chunk_id),
            document_id=UUID(document_id),
            document_title="特锐德",
            source_content_hash="hash",
            question=_QUESTION,
            answer_original="12.3 亿元",
            question_order=0,
            question_sha256=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        )

    return (
        service,
        run,
        _case(chunk_a1, document_a),   # 有同文档候选
        _case(chunk_b1, document_b),   # 没有
        chunk_a2,
    )


async def test_candidates_are_grouped_by_document(session):

    service, run, *_, = await _fixture(session)

    pool = await service._load_pool_by_document(run.generation_run_id)

    assert sorted(len(chunks) for chunks in pool.values()) == [1, 2]


async def test_no_same_document_candidate_costs_zero_calls(session):
    """
    **没有一个同文档候选时，一次模型都不调。**

    这是成本保证，不是优化。跨公司抽样时它是常态（20 块抽自 100 家），
    所以这条路径决定了判金标这一环**通常不花钱**。

    它一旦被绕过（比如把候选池放宽成整个池子），题库照出、结果照对，
    只是每道题都去问一次模型 —— 没有任何症状，只有账单。
    """

    service, run, case_a, case_b, candidate = await _fixture(session)

    await service._cases.add_cases([case_a, case_b])

    pool = await service._load_pool_by_document(run.generation_run_id)

    runtime = _FakeRuntime({
        "answerable_chunk_ids": [candidate],
        "evidence_excerpts": {candidate: _EVIDENCE},
    })

    added = await service._adjudicate_cases(
        cases=[case_b],  # 只有"没有候选"的那条
        runtime=runtime,
        pool_by_document=pool,
    )

    assert runtime.requests == [], "没有候选却调了模型 —— 成本保证破了"
    assert added == 0


async def test_same_document_candidate_is_judged_and_recorded(session):
    """
    有同文档候选时：判一次，写一行，且 `is_source_chunk=False`。

    `is_source_chunk` 分开记是有意义的：**这一行才是裁定真正救回来的
    那些**。命中来自源 chunk 还是来自别处，是两个不同的信号，统计里要
    分得开。
    """

    service, run, case_a, _, candidate = await _fixture(session)

    await service._cases.add_cases([case_a])

    pool = await service._load_pool_by_document(run.generation_run_id)

    runtime = _FakeRuntime({
        "answerable_chunk_ids": [candidate],
        "evidence_excerpts": {candidate: _EVIDENCE},
    })

    added = await service._adjudicate_cases(
        cases=[case_a],
        runtime=runtime,
        pool_by_document=pool,
    )

    assert len(runtime.requests) == 1
    assert added == 1

    gold = await service._cases.list_gold_chunks([case_a.eval_case_id])

    rows = gold[case_a.eval_case_id]

    assert [str(row.chunk_id) for row in rows] == [candidate]
    assert rows[0].is_source_chunk is False
    assert rows[0].validation_method.value == "llm_cross_chunk_judge"
    assert rows[0].source_content_hash == "hash-a2"


async def test_verdict_without_verbatim_evidence_is_discarded(session):
    """
    证据在候选正文里找不到 -> 这条判定作废。

    模型最容易犯的错是"相关就算能答"。要求它给出**逐字**证据，并且真的
    去正文里核对，是唯一能自动验证"它的判定落到了原文上"的手段 ——
    没有这道校验，模型说一句"能回答"那个 chunk 就进金标集合。
    """

    service, run, case_a, _, candidate = await _fixture(session)

    await service._cases.add_cases([case_a])

    pool = await service._load_pool_by_document(run.generation_run_id)

    runtime = _FakeRuntime({
        "answerable_chunk_ids": [candidate],
        # 这一句**不存在于任何候选正文里**
        "evidence_excerpts": {candidate: "这句话是我编的"},
    })

    added = await service._adjudicate_cases(
        cases=[case_a],
        runtime=runtime,
        pool_by_document=pool,
    )

    assert added == 0, "编造的证据不该被采信"

    gold = await service._cases.list_gold_chunks([case_a.eval_case_id])

    assert gold == {}


async def test_judge_role_is_not_even_resolved_when_disabled(
        session,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    开关关着时，judge 的 model role **一次都不解析**。

    这是「默认不要接通」的最强形式：不是「解析了但不用」，而是**根本没碰**。
    区别很实在 —— `get_runtime` 在 role 没配绑定时**直接抛
    `ModelResolutionError`**，所以无条件解析的话，「judge 这一路还没接通」
    会让**整批造题失败**，而那恰恰是关闭开关本该避免的事。

    换句话说：关着的时候，judge 的配置对不对**与我们无关**。
    """

    import app.services.document.application.retrieval_eval_service_impl as impl
    from app.agent.model import ModelRole
    from app.services.document.domain.tasks import (
        RetrievalEvalGenerateParams,
    )

    service, run, *_ = await _fixture(session)

    resolved: list[str] = []

    class _Registry:
        def get_runtime(self, *, agent_type: str, model_role: str) -> object:

            resolved.append(model_role)

            if model_role == ModelRole.EVAL_CROSS_CHUNK_JUDGE:
                raise AssertionError(
                    "开关关着时不该解析 judge 角色 —— 它没配好也不该"
                    "影响造题"
                )

            # 出题那一路给个空壳（下面把窗口生成也换掉了，不会真调模型）。
            return SimpleNamespace()

    monkeypatch.setattr(impl, "get_model_registry", lambda: _Registry())

    # 窗口生成换成"什么都没产出"，避免真调模型。
    async def _empty(**_: object) -> object:
        from app.services.document.eval.generator import WindowOutcome

        return WindowOutcome()

    monkeypatch.setattr(
        impl.eval_generator, "generate_for_window", _empty,
    )

    await service.generate(
        RetrievalEvalGenerateParams(generation_run_id=run.generation_run_id),
        None,
    )

    assert ModelRole.EVAL_QA_GENERATION in resolved, "出题那一路该被解析"
    assert ModelRole.EVAL_CROSS_CHUNK_JUDGE not in resolved, (
        "judge 角色被解析了 —— 开关没生效"
    )


async def test_judge_failure_degrades_to_source_chunk_only(session):
    """
    裁定失败（模型挂了）只让金标退化成"只有源 chunk"，**不影响这一步**。

    金标集合是**下界**，退化是安全的：题还在、源 chunk 还在，只是少了
    可能存在的其它答案。抛出去会让整个窗口陪葬，而那要重付一遍出题的
    模型调用。
    """

    class _Broken:
        async def ainvoke(self, request: object) -> object:
            raise RuntimeError("模型挂了")

    service, run, case_a, _, _candidate = await _fixture(session)

    await service._cases.add_cases([case_a])

    pool = await service._load_pool_by_document(run.generation_run_id)

    added = await service._adjudicate_cases(
        cases=[case_a],
        runtime=_Broken(),
        pool_by_document=pool,
    )

    assert added == 0
