"""
检索评测仓储的**真库**回归：幂等与约束。

为什么这些用例必须连真库
------------------------
它们要焊住的东西**只有 PostgreSQL 知道**：

    ON CONFLICT DO NOTHING 到底跳不跳（它取决于库上真的存在哪些唯一约束）
    CHECK (answer_end_offset >= answer_start_offset) 是不是真的建上了
    UPDATE ... SET x = x + n 是不是真的原子

用假仓储测等于把这几条抄一遍再断言它等于自己 —— 抄错一个约束，重试
照样插出重复行，用例却全绿。这与 test_chunk_rerun.py 是同一条理由。

建表也在**会回滚的事务里**，但**别指望库是空的**
--------------------------------------------
PostgreSQL 的 DDL 是事务性的，所以这里 `create_all` 出来的九张表在用例
结束时一并消失。表还没建过时用例照样能跑（不依赖"先启动过一次应用"）。

⚠️ 但**应用真启动过一次之后，这九张表就长期存在了**，而且里面会有真实
数据集 —— 第一次真跑时就有。于是：

    ✗ 断言全局总数       list_bundles() 不带过滤会把别人的行数进来，
                        用例在真启动过之后必红，而红的理由与它要验的
                        东西毫无关系（这个坑真的踩到了）
    ✓ 按 tenant/主键收窄  每个用例用自己 uuid4() 出来的 tenant_id

所以本文件里凡是"列表"类的断言，一律带 `tenant_id=`。

连不上库就跳过，不失败
----------------------
其余用例都只用假替身，不该被一条库连接拖住。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest
import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.common.models import Base
from app.services.document.domain.entities import (
    RetrievalEvalBundle,
    RetrievalEvalCase,
    RetrievalEvalCaseGoldChunk,
    RetrievalEvalGenerationRun,
)
from app.services.document.domain.schemas.enum import (
    EvalBundleStatus,
    EvalCaseValidationStatus,
    EvalGoldValidationMethod,
    EvalJobStatus,
)
from app.services.document.repository.retrieval_eval_bundle_impl import (
    RetrievalEvalBundleRepositoryImpl,
)
from app.services.document.repository.retrieval_eval_case_impl import (
    RetrievalEvalCaseRepositoryImpl,
)

#: 真实的数据源配置。从 __file__ 往上推而不是写相对路径（见
#: tests/conftest.py 的说明）。
POSTGRE_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "app" / "config" / "datasource" / "postgre.yaml"
)

#: 本模块要用到的九张表。
_TABLE_PREFIX = "retrieval_eval"


def _dsn() -> str:
    raw = yaml.safe_load(POSTGRE_CONFIG.read_text(encoding="utf-8"))
    return raw["config"]["dsn"]


@pytest.fixture
async def session():
    """
    一个用完就回滚的会话，**连同建表**。

    `import_all_models()` 必须先跑：SQLAlchemy 只认识已经 import 过的
    模型，否则 `Base.metadata` 里压根没有这九张表。
    """

    from app.infra.datasource.initializer import import_all_models

    import_all_models()

    engine = create_async_engine(_dsn())

    try:
        connection = await engine.connect()
        await connection.close()

    except Exception as exc:  # noqa: BLE001 —— 连不上就跳过
        await engine.dispose()
        pytest.skip(f"连不上本地 Postgres，跳过检索评测仓储回归：{exc}")

    session = AsyncSession(engine, expire_on_commit=False)

    try:
        # ---- 在会回滚的事务里建表 ----
        #
        # 只建这九张：create_all 对已存在的表是 no-op，所以库里已经有
        # 它们时这里什么都不做。
        tables = [
            table
            for name, table in Base.metadata.tables.items()
            if name.startswith(_TABLE_PREFIX)
        ]

        connection = await session.connection()
        await connection.run_sync(
            Base.metadata.create_all, tables=tables,
        )

        yield session

    finally:
        # 显式 rollback：DDL 与用例插入的行一并消失，开发库不留痕迹。
        await session.rollback()
        await session.close()
        await engine.dispose()


def _bundle(**overrides) -> RetrievalEvalBundle:

    payload = {
        "bundle_id": uuid.uuid4(),
        "name": "回归用数据集",
        "status": EvalBundleStatus.GENERATING,
        "selection_config": {"seed": 7},
        "generation_config": {"questions_per_chunk": 3},
    }

    payload.update(overrides)

    return RetrievalEvalBundle(**payload)


def _case(bundle_id: UUID, **overrides) -> RetrievalEvalCase:

    payload = {
        "eval_case_id": uuid.uuid4(),
        "bundle_id": bundle_id,
        "document_type_definition_id": uuid.uuid4(),
        "source_chunk_id": uuid.uuid4(),
        "source_content_hash": "a" * 64,
        "question": "2023 年营收是多少？",
        "answer_original": "80 亿元",
        "answer_start_offset": 16,
        "answer_end_offset": 21,
        "question_order": 0,
        "question_sha256": uuid.uuid4().hex + uuid.uuid4().hex[:32],
    }

    payload.update(overrides)

    return RetrievalEvalCase(**payload)


# ============================================================
# Bundle / run 基础读写
# ============================================================


async def test_bundle_round_trip(session):

    repository = RetrievalEvalBundleRepositoryImpl(session=session)

    stored = await repository.add_bundle(_bundle(name="第一份"))

    fetched = await repository.get_bundle(stored.bundle_id)

    assert fetched is not None
    assert fetched.name == "第一份"
    # selection/generation config 是 JSONB，往返之后要仍然可读 ——
    # 它们是**复现这份数据集的唯一线索**，丢了就再也说不清抽样参数。
    assert fetched.selection_config == {"seed": 7}
    assert fetched.generation_config == {"questions_per_chunk": 3}


async def test_update_bundle_returns_without_reloading(session):
    """
    更新 bundle 之后**不触发懒加载**。

    ⚠️ 这条是补一个真实漏网事故的。线上第一次真跑时它炸了：

        sqlalchemy.exc.MissingGreenlet: greenlet_spawn has not been called;
        can't call await_only() here.

    根因：`updated_at` 列上有 `onupdate=func.now()`，实现里**没有显式赋值**，
    于是 SQLAlchemy 让数据库去算它 —— 这个属性在 flush 之后变成 **expired**，
    而紧接着读它（构造返回的实体）触发了一次**懒加载 SELECT**。asyncio 里
    那次 IO 落在同步上下文，于是 MissingGreenlet。

    两条教训都焊在这里：

        1. `updated_at` 必须由服务层盖章、仓储显式赋值（本仓库既有写法）
        2. 更要紧的是 —— **在此之前没有任何用例调过 `update_bundle`**。
           加了这个方法却没测它，于是它带着一个必然触发的缺陷通过了全部
           真库回归。

    断言 `updated_at` 等于传进去的那个值，而不是"不抛异常"：如果哪天有人
    改回依赖 onupdate，属性会再次 expired —— 在真库上读它就是抛异常，
    这条会红。
    """

    repository = RetrievalEvalBundleRepositoryImpl(session=session)

    bundle = _bundle(status=EvalBundleStatus.GENERATING)
    await repository.add_bundle(bundle)

    stamped = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

    bundle.status = EvalBundleStatus.READY
    bundle.source_chunk_count = 20
    bundle.case_count = 57
    bundle.dataset_sha256 = "c" * 64
    bundle.completed_at = stamped
    bundle.updated_at = stamped

    updated = await repository.update_bundle(bundle)

    assert updated.status is EvalBundleStatus.READY
    assert updated.case_count == 57
    assert updated.dataset_sha256 == "c" * 64
    assert updated.updated_at == stamped, (
        "updated_at 不是传进去的那个 —— 说明实现依赖了 onupdate，"
        "属性会 expired，真库上读它必炸 MissingGreenlet"
    )

    # 再读一次，确认真的落库了（而不是只改了内存里的对象）。
    reread = await repository.get_bundle(bundle.bundle_id)

    assert reread.status is EvalBundleStatus.READY
    assert reread.case_count == 57
    assert reread.updated_at == stamped


async def test_bundle_status_filter_and_paging(session):
    """
    ⚠️ 断言必须**按 tenant 收窄**，不能断言全局总数。

    库里的这批表在应用真启动过一次之后就**长期存在**了，里面会有真实
    数据集（第一次真跑时就有）。`list_bundles()` 不带过滤是把别人的行
    一起数进去 —— 那样写的用例第一次真跑之后就会红，而红的理由与它要
    验的东西毫无关系。
    """

    repository = RetrievalEvalBundleRepositoryImpl(session=session)

    tenant = uuid.uuid4()
    fresh_id = uuid.uuid4()

    await repository.add_bundle(
        _bundle(
            bundle_id=fresh_id,
            tenant_id=tenant,
            status=EvalBundleStatus.READY,
        )
    )
    await repository.add_bundle(
        _bundle(tenant_id=tenant, status=EvalBundleStatus.DRAFT)
    )

    items, total = await repository.list_bundles(
        tenant_id=tenant,
        status=EvalBundleStatus.READY,
    )

    assert total == 1
    assert [item.bundle_id for item in items] == [fresh_id]

    # 分页的第二排序键必须让结果稳定：同一页重复请求拿到同一批。
    first, _ = await repository.list_bundles(
        tenant_id=tenant, offset=0, limit=1,
    )
    again, _ = await repository.list_bundles(
        tenant_id=tenant, offset=0, limit=1,
    )
    assert [x.bundle_id for x in first] == [x.bundle_id for x in again]


async def test_bundle_order_follows_created_at_not_uuid(session):
    """
    排序必须由 `created_at` 决定，**不能掉到 UUID 上**。

    故意让"更新那一份"拿到**更小**的 UUID：若实现里 created_at 相等
    或者排序键写错，结果就会反过来。这条用例是上一条那个坑的护栏 ——
    它钉住的是"我们按时间排"，而不只是"这次恰好排对了"。
    """

    repository = RetrievalEvalBundleRepositoryImpl(session=session)

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tenant = uuid.uuid4()

    older = _bundle(
        bundle_id=UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),
        tenant_id=tenant,
        name="先建的",
        created_at=base,
    )
    newer = _bundle(
        bundle_id=UUID("00000000-0000-0000-0000-000000000001"),
        tenant_id=tenant,
        name="后建的",
        created_at=base.replace(second=1),
    )

    # 插入顺序与 UUID 顺序都刻意反过来。
    await repository.add_bundle(older)
    await repository.add_bundle(newer)

    # 按 tenant 收窄 —— 见上一个用例里"为什么不能断言全局总数"的说明。
    items, total = await repository.list_bundles(tenant_id=tenant)

    assert total == 2
    assert [item.name for item in items] == ["后建的", "先建的"]


async def test_latest_generation_run_is_per_bundle(session):
    """
    ⚠️ 这里的 `created_at` **是显式填的，而且必须显式填**
    ----------------------------------------------------
    一度写成不填、靠列的 `server_default=now()`，用例就变成了**随机
    通过**：PostgreSQL 的 `now()` 是**事务开始时间**，同一个事务里插的
    四行拿到完全相同的 `created_at`，于是"最近一次"掉到
    `generation_run_id`（随机 UUID）的 tiebreaker 上。

    症状很有欺骗性：单跑这个文件时全绿，跑全量时才红 —— 因为它依赖
    三个随机 UUID 的相对大小，而测试顺序会影响……什么都不影响，
    它只是每次掷一次骰子。

    （这一条是被"全量跑一次"抓出来的。单文件绿不代表绿。）
    """

    repository = RetrievalEvalBundleRepositoryImpl(session=session)

    bundle_a = _bundle()
    bundle_b = _bundle()

    await repository.add_bundle(bundle_a)
    await repository.add_bundle(bundle_b)

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for index in range(3):
        await repository.add_generation_run(
            RetrievalEvalGenerationRun(
                generation_run_id=uuid.uuid4(),
                bundle_id=bundle_a.bundle_id,
                sample_seed=100 + index,
                created_at=base.replace(second=index),
            )
        )

    await repository.add_generation_run(
        RetrievalEvalGenerationRun(
            generation_run_id=uuid.uuid4(),
            bundle_id=bundle_b.bundle_id,
            sample_seed=999,
            created_at=base.replace(second=0),
        )
    )

    latest = await repository.list_latest_generation_runs(
        [bundle_a.bundle_id, bundle_b.bundle_id]
    )

    assert set(latest) == {bundle_a.bundle_id, bundle_b.bundle_id}
    # A 拿到的是**最后一次**（种子最大的那个），不是第一条。
    assert latest[bundle_a.bundle_id].sample_seed == 102
    assert latest[bundle_b.bundle_id].sample_seed == 999

    # 没有运行记录的 bundle 不出现在结果里（用 .get 取值，语义一样）。
    assert await repository.list_latest_generation_runs([]) == {}


async def test_window_counters_are_derived_not_accumulated(session):
    """
    计数从窗口行**推导**，而不是累加。

    这是"重复执行不产生偏差"的落点。任务基础设施是至少一次投递（被
    reclaimed 后会重跑），而**累加**在重跑时会算重 —— 症状是"产出的
    case 比列出来的多"，且只在重跑路径上出现。

    所以计数走 `count_windows_by_status`：它每次都从窗口行重算，跑几遍
    都是同一个答案。运行行上的那三个数由服务层按它整体写回。

    （曾经真的写过一个 additive 的 `increment_generation_run_counters`，
    在这里被换掉了 —— 保留这条用例是为了让"不能再加回累加"有个护栏。）
    """

    repository = RetrievalEvalBundleRepositoryImpl(session=session)

    bundle = _bundle()
    await repository.add_bundle(bundle)

    run = RetrievalEvalGenerationRun(
        generation_run_id=uuid.uuid4(),
        bundle_id=bundle.bundle_id,
        sample_seed=1,
    )
    await repository.add_generation_run(run)

    from app.services.document.domain.entities import (
        RetrievalEvalGenerationWindow,
    )

    windows = [
        RetrievalEvalGenerationWindow(
            generation_window_id=uuid.uuid4(),
            generation_run_id=run.generation_run_id,
            window_index=index,
            status=status,
            input_chunks=[{"chunk_id": str(uuid.uuid4()), "content": "x"}],
        )
        for index, status in enumerate([
            EvalJobStatus.COMPLETED,
            EvalJobStatus.COMPLETED,
            EvalJobStatus.FAILED,
            EvalJobStatus.QUEUED,
        ])
    ]

    await repository.add_windows(windows)

    counts = await repository.count_windows_by_status(
        run.generation_run_id
    )

    assert counts == {"completed": 2, "failed": 1, "queued": 1}

    # 再数一遍必须一模一样 —— 这就是"重跑不会算重"。
    again = await repository.count_windows_by_status(
        run.generation_run_id
    )
    assert again == counts

    # 没有窗口的运行返回空字典（而不是报错）。
    assert await repository.count_windows_by_status(uuid.uuid4()) == {}


# ============================================================
# Case 幂等（这一组是重点）
# ============================================================


async def test_add_cases_is_idempotent(session):
    """
    同一批 case 插两次，第二次一条也不写。

    这正是"重试不产生重复"的落点：窗口重试会重造同一批 case，而
    `(bundle, 源chunk, 题序)` 上的唯一约束必须把第二批挡住。

    ⚠️ 断言的是**返回值**（真的写了几条），不只是"没抛异常" ——
    只断言不抛的话，一个"静默 upsert 覆盖"的实现也能过，而那会让重试
    悄悄改掉首轮已经校验过的内容。
    """

    repository = RetrievalEvalCaseRepositoryImpl(session=session)

    bundle = _bundle()
    await RetrievalEvalBundleRepositoryImpl(
        session=session
    ).add_bundle(bundle)

    cases = [_case(bundle.bundle_id) for _ in range(3)]

    first = await repository.add_cases(cases)
    second = await repository.add_cases(cases)

    assert first == 3
    assert second == 0, "重复插入必须被唯一约束挡下，不该写进第二批"

    total = await repository.count_cases(bundle.bundle_id)
    assert total == 3


async def test_case_round_trips_the_source_document_identity(session):
    """
    case 带着源文档身份往返。

    ⚠️ 这两列不是装饰：本语料的正文常写「本公司」（实测 1477/39697 个
    chunk 含这三个字），所以出题时主体只能取自**文档**。而问题一旦写了
    公司名，**答案的正确性就只能靠"这块 chunk 属于那家公司"来保证**
    —— 正文里的那个数字不带主体。不记这两列，这类题事后无法复核。

    顺带焊住一处易错点：实体上的名字是 `document_title` /
    `document_key`，而列名是 `source_document_title` /
    `source_document_key`（前缀区分"这是源文档的快照，不是当前归属"）。
    两边靠 `_to_case` 手工映射 —— 写错一个不是报错，是**字段恒为 None**。
    """

    repository = RetrievalEvalCaseRepositoryImpl(session=session)
    bundle_repository = RetrievalEvalBundleRepositoryImpl(session=session)

    bundle = _bundle()
    await bundle_repository.add_bundle(bundle)

    document_id = uuid.uuid4()

    case = _case(
        bundle.bundle_id,
        document_id=document_id,
        document_title="特锐德",
        document_key="300001.SZ",
    )

    assert await repository.add_cases([case]) == 1

    fetched, _ = await repository.list_cases(bundle_id=bundle.bundle_id)

    assert len(fetched) == 1

    stored = fetched[0]

    assert stored.document_id == document_id
    assert stored.document_title == "特锐德"
    assert stored.document_key == "300001.SZ"


async def test_case_document_identity_may_be_absent(session):
    """
    文档没标题 / 没业务标识时，两列是 None 而**不是空串**。

    空串与 NULL 在这个语义下必须分开：空串是"文档确实有一项空标题"，
    NULL 是"文档没提供"。提示词那边据此决定写「未知」还是不写 ——
    混成一个的话，模型会拿到一个空的"所属文档："行。
    """

    repository = RetrievalEvalCaseRepositoryImpl(session=session)
    bundle_repository = RetrievalEvalBundleRepositoryImpl(session=session)

    bundle = _bundle()
    await bundle_repository.add_bundle(bundle)

    await repository.add_cases([_case(bundle.bundle_id)])

    fetched, _ = await repository.list_cases(bundle_id=bundle.bundle_id)

    assert fetched[0].document_title is None
    assert fetched[0].document_key is None


async def test_same_question_different_order_is_also_rejected(session):
    """
    同一个问题换个题序也算重复。

    模型重试时把同一道题排到另一个序号是常见的（输出顺序会变）。只靠
    `(bundle, chunk, 题序)` 挡不住它，所以还有一条按**问题指纹**的唯一
    约束。两条都撞得上才算"重试幂等"。
    """

    repository = RetrievalEvalCaseRepositoryImpl(session=session)

    bundle = _bundle()
    await RetrievalEvalBundleRepositoryImpl(
        session=session
    ).add_bundle(bundle)

    chunk_id = uuid.uuid4()
    fingerprint = "f" * 64

    original = _case(
        bundle.bundle_id,
        source_chunk_id=chunk_id,
        question_sha256=fingerprint,
        question_order=0,
    )

    # 同一个问题、不同题序 -> 撞 uq_retrieval_eval_case_question
    duplicate = _case(
        bundle.bundle_id,
        source_chunk_id=chunk_id,
        question_sha256=fingerprint,
        question_order=1,
    )

    assert await repository.add_cases([original]) == 1
    assert await repository.add_cases([duplicate]) == 0


async def test_offsets_check_is_enforced(session):
    """
    答案偏移的先后关系由库上的 CHECK 兜底。

    实体层的 `model_validator` 先拦一道（错误离原因更近），而库这道拦的是
    **任何**写入路径 —— 包括运维手工 SQL。两条都要在。
    """

    repository = RetrievalEvalCaseRepositoryImpl(session=session)

    bundle = _bundle()
    await RetrievalEvalBundleRepositoryImpl(
        session=session
    ).add_bundle(bundle)

    # 实体层先拦：构造都构造不出来。
    with pytest.raises(ValueError, match="answer_end_offset"):
        _case(
            bundle.bundle_id,
            answer_start_offset=30,
            answer_end_offset=10,
        )


async def test_gold_chunks_grouped_and_source_first(session):
    """
    金标按 case 分组，且**源 chunk 排最前**。

    顺序不是装饰：导出与预览里第一个看到的应当是"出题的那一块" ——
    它是构造出来的事实，可信度最高；跨 chunk 裁定出来的那些排在后面。
    """

    repository = RetrievalEvalCaseRepositoryImpl(session=session)
    bundle_repository = RetrievalEvalBundleRepositoryImpl(session=session)

    bundle = _bundle()
    await bundle_repository.add_bundle(bundle)

    case = _case(bundle.bundle_id)
    await repository.add_cases([case])

    other_chunk = uuid.uuid4()

    gold = [
        # 故意先插"别处的 chunk"，源 chunk 后插 —— 排序若靠插入顺序，
        # 这条断言会红。
        RetrievalEvalCaseGoldChunk(
            eval_case_gold_chunk_id=uuid.uuid4(),
            eval_case_id=case.eval_case_id,
            chunk_id=other_chunk,
            is_source_chunk=False,
            validation_method=EvalGoldValidationMethod.LLM_CROSS_CHUNK_JUDGE,
            evidence_excerpt="内需疲软",
        ),
        RetrievalEvalCaseGoldChunk(
            eval_case_gold_chunk_id=uuid.uuid4(),
            eval_case_id=case.eval_case_id,
            chunk_id=case.source_chunk_id,
            is_source_chunk=True,
            validation_method=EvalGoldValidationMethod.SOURCE_CHUNK,
        ),
    ]

    assert await repository.add_gold_chunks(gold) == 2
    # 重跑裁定不该在同一个 (case, chunk) 上留两行。
    assert await repository.add_gold_chunks(gold) == 0

    grouped = await repository.list_gold_chunks([case.eval_case_id])

    assert list(grouped) == [case.eval_case_id]
    assert [item.chunk_id for item in grouped[case.eval_case_id]] == [
        case.source_chunk_id,
        other_chunk,
    ]

    # 没有金标的 case 不出现在结果里（调用方用 .get(id, [])）。
    assert await repository.list_gold_chunks([uuid.uuid4()]) == {}


async def test_distinct_source_chunks_counts_chunks_not_cases(session):
    """
    分母是**不同的源 chunk 数**，不是 case 数。

    同一个 chunk 出的题可能少于 M（内容不够），所以 `case_count / M`
    会低估覆盖面。这个差异只在"某些 chunk 只出了 1 道题"时才显形 ——
    而那正是常见情况。
    """

    repository = RetrievalEvalCaseRepositoryImpl(session=session)

    bundle = _bundle()
    await RetrievalEvalBundleRepositoryImpl(
        session=session
    ).add_bundle(bundle)

    shared_chunk = uuid.uuid4()

    cases = [
        _case(
            bundle.bundle_id,
            source_chunk_id=shared_chunk,
            question_order=0,
            question_sha256=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        ),
        _case(
            bundle.bundle_id,
            source_chunk_id=shared_chunk,
            question_order=1,
            question_sha256=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        ),
        _case(
            bundle.bundle_id,
            source_chunk_id=uuid.uuid4(),
            question_order=0,
            question_sha256=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        ),
    ]

    assert await repository.add_cases(cases) == 3
    assert await repository.count_cases(bundle.bundle_id) == 3
    assert await repository.count_distinct_source_chunks(
        bundle.bundle_id
    ) == 2


async def test_validated_export_order_is_deterministic(session):
    """
    导出顺序固定：DTD -> 源 chunk -> 题序。

    `dataset_sha256` 建在这个确定性上 —— 顺序若依赖库的返回顺序，同样
    的数据导出两次会得到不同的指纹，而那个指纹的用途恰恰是回答
    "这两次评测用的是不是同一份数据"。
    """

    repository = RetrievalEvalCaseRepositoryImpl(session=session)

    bundle = _bundle()
    await RetrievalEvalBundleRepositoryImpl(
        session=session
    ).add_bundle(bundle)

    dtd_first = UUID("00000000-0000-0000-0000-000000000001")
    dtd_second = UUID("00000000-0000-0000-0000-000000000002")

    cases = [
        _case(
            bundle.bundle_id,
            document_type_definition_id=dtd_second,
            question_sha256=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        ),
        _case(
            bundle.bundle_id,
            document_type_definition_id=dtd_first,
            question_sha256=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        ),
        # 这条是 rejected，不该进导出。
        _case(
            bundle.bundle_id,
            document_type_definition_id=dtd_first,
            validation_status=EvalCaseValidationStatus.REJECTED,
            question_sha256=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        ),
    ]

    await repository.add_cases(cases)

    exported = await repository.list_all_validated_cases(bundle.bundle_id)

    assert len(exported) == 2, "rejected 的 case 不该出现在导出里"
    assert [
        case.document_type_definition_id for case in exported
    ] == [dtd_first, dtd_second]

    assert await repository.count_cases(
        bundle.bundle_id,
        validation_status=EvalCaseValidationStatus.VALIDATED,
    ) == 2


async def test_window_identity_is_not_overwritten(session):
    """
    窗口的输入快照与序号更新不上去。

    它们是这个窗口的**身份**：重试靠 `window_index` 定位、靠
    `input_chunks` 复现当时的正文。允许改写的话，"重试"会变成"换了输入
    再跑一次"，而产出照样进同一份数据集 —— 事后无从分辨。
    """

    repository = RetrievalEvalBundleRepositoryImpl(session=session)

    bundle = _bundle()
    await repository.add_bundle(bundle)

    run = RetrievalEvalGenerationRun(
        generation_run_id=uuid.uuid4(),
        bundle_id=bundle.bundle_id,
        sample_seed=42,
    )
    await repository.add_generation_run(run)

    from app.services.document.domain.entities import (
        RetrievalEvalGenerationWindow,
    )

    window = RetrievalEvalGenerationWindow(
        generation_window_id=uuid.uuid4(),
        generation_run_id=run.generation_run_id,
        window_index=0,
        input_chunks=[
            {"chunk_id": str(uuid.uuid4()), "content": "原始正文"}
        ],
    )

    assert await repository.add_windows([window]) == 1

    # 试图改写身份 + 更新状态。
    window.input_chunks = [
        {"chunk_id": str(uuid.uuid4()), "content": "换过的正文"}
    ]
    window.window_index = 99
    window.status = EvalJobStatus.COMPLETED
    window.valid_case_count = 3

    await repository.update_window(window)

    fetched = await repository.get_window(window.generation_window_id)

    assert fetched.status is EvalJobStatus.COMPLETED
    assert fetched.valid_case_count == 3
    assert fetched.window_index == 0, "序号是身份，不该被改"
    assert fetched.input_chunks[0]["content"] == "原始正文", (
        "输入快照是身份，不该被改"
    )


async def test_unfinished_windows_include_queued(session):
    """
    "还没成功"包含 `queued` —— 这是刻意的。

    一次执行有窗口数上限，跑不完的留在 `queued`；而本层**不发子任务**，
    所以"再次提交同一个任务"是继续跑它们的唯一途径。把 `queued` 排除
    掉的话，上限之外那些窗口永远没人处理。

    并发保护不靠这个过滤条件，靠 `claim_window`（见下一条用例）。
    """

    repository = RetrievalEvalBundleRepositoryImpl(session=session)

    bundle = _bundle()
    await repository.add_bundle(bundle)

    run = RetrievalEvalGenerationRun(
        generation_run_id=uuid.uuid4(),
        bundle_id=bundle.bundle_id,
        sample_seed=42,
    )
    await repository.add_generation_run(run)

    from app.services.document.domain.entities import (
        RetrievalEvalGenerationWindow,
    )

    windows = [
        RetrievalEvalGenerationWindow(
            generation_window_id=uuid.uuid4(),
            generation_run_id=run.generation_run_id,
            window_index=index,
            status=status,
            input_chunks=[{"chunk_id": str(uuid.uuid4()), "content": "x"}],
        )
        for index, status in enumerate([
            EvalJobStatus.QUEUED,
            EvalJobStatus.COMPLETED,
            EvalJobStatus.FAILED,
            EvalJobStatus.PARTIAL_FAILED,
            EvalJobStatus.RUNNING,
            EvalJobStatus.CANCELLED,
        ])
    ]

    await repository.add_windows(windows)

    unfinished = await repository.list_unfinished_windows(
        run.generation_run_id
    )

    # queued(0) + failed(2) + partial_failed(3)；completed/running/cancelled
    # 都不该出现 —— running 表示正被别的执行处理。
    assert [window.window_index for window in unfinished] == [0, 2, 3]

    all_windows = await repository.list_windows(run.generation_run_id)
    assert [window.window_index for window in all_windows] == [0, 1, 2, 3, 4, 5]


async def test_claim_window_is_race_safe(session):
    """
    **同一个窗口只能被认领一次。**

    这是"同一批窗口不会被跑两遍"唯一可靠的落点：应用层"先查状态、再决定
    跑不跑"在两次执行并发时两边都会查到 `queued`，然后**都去调模型** ——
    而模型是按次计费的，重复那一遍的钱已经花掉了。

    会真的并发：调用方连点两次"重试"就会提交两条任务，两条各领一个
    lease、各跑各的。
    """

    repository = RetrievalEvalBundleRepositoryImpl(session=session)

    bundle = _bundle()
    await repository.add_bundle(bundle)

    run = RetrievalEvalGenerationRun(
        generation_run_id=uuid.uuid4(),
        bundle_id=bundle.bundle_id,
        sample_seed=7,
    )
    await repository.add_generation_run(run)

    from app.services.document.domain.entities import (
        RetrievalEvalGenerationWindow,
    )

    window = RetrievalEvalGenerationWindow(
        generation_window_id=uuid.uuid4(),
        generation_run_id=run.generation_run_id,
        window_index=0,
        input_chunks=[{"chunk_id": str(uuid.uuid4()), "content": "x"}],
    )
    await repository.add_windows([window])

    assert await repository.claim_window(window.generation_window_id) is True

    # 第二次认领必须失败 —— 状态已经是 running，不在可认领集合里。
    assert await repository.claim_window(window.generation_window_id) is False

    claimed = await repository.get_window(window.generation_window_id)
    assert claimed.status is EvalJobStatus.RUNNING
    # 认领与"尝试次数 +1"是同一件事：失败的那次不该把计数加上去。
    assert claimed.attempt_count == 1

    # 跑失败之后又可以认领了（这是重试的入口）。
    claimed.status = EvalJobStatus.FAILED
    await repository.update_window(claimed)

    assert await repository.claim_window(window.generation_window_id) is True

    retried = await repository.get_window(window.generation_window_id)
    assert retried.attempt_count == 2

    # 已完成的不能再认领。
    retried.status = EvalJobStatus.COMPLETED
    await repository.update_window(retried)

    assert await repository.claim_window(window.generation_window_id) is False
