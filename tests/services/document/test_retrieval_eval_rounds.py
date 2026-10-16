"""
逐轮（`attempt_no`）读路径的真库回归：单条痕迹与 case 列表。

要焊住的那条缝
------------
`retrieval_eval_case_result` 的唯一键是
`(eval_run_id, eval_case_id, attempt_no)` —— 逐轮放宽让**同一条 case 有
多行**（第 N 轮只跑第 N-1 轮没命中的，第 3 轮才开精排）。而两条读路径
早先都按「(run, case) 就是一行」写：

    痕迹页   按 (run, case) 取一行 + `scalar_one_or_none()` —— 这条 case
             跑到过第 2 轮就 **MultipleResultsFound 直接 500**
    列表     按行分页、按行筛 hit —— 同一条 case 出现 2~3 次且响应里没有
             轮次字段可分，`total` 是行数（实测一轮 40 条 case 报 84）

为什么必须连真库
----------------
第二种错法（`MultipleResultsFound`）是**真实数据库会话**才抛的：假仓储
返回「排好的返回值」，取一行永远取得到一行。这与
`test_retrieval_eval_repositories.py` 是同一条理由 —— 抄一遍假约束再断言
它等于自己，真正的约束错配照样全绿。

数据形状照着真实那一轮来（库里实测：40 条 case、84 行、轮次分布
40/24/20）：

    case A   三轮都没命中      -> 代表行取**第 3 轮**（同级取更晚的）
    case B   第 1 轮没中、第 2 轮命中 -> 代表行是第 2 轮
    case C   第 1 轮就命中      -> 只有第 1 轮

⚠️ 用例自己造数据、跑在**会回滚的事务**里（DDL 与插入一并消失）。
"""

from __future__ import annotations

import uuid
from pathlib import Path
from uuid import UUID

import pytest
import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.common.models import Base
from app.services.document.application.retrieval_eval_service_impl import (
    RetrievalEvalServiceImpl,
)
from app.services.document.domain.entities import (
    RetrievalEvalCaseResult,
    RetrievalEvalRun,
)
from app.services.document.repository.document_chunk_impl import (
    DocumentChunkRepositoryImpl,
)
from app.services.document.repository.retrieval_eval_bundle_impl import (
    RetrievalEvalBundleRepositoryImpl,
)
from app.services.document.repository.retrieval_eval_case_impl import (
    RetrievalEvalCaseRepositoryImpl,
)
from app.services.document.repository.retrieval_eval_run_impl import (
    RetrievalEvalRunRepositoryImpl,
)
from app.services.document.repository.type_definition_impl import (
    DocumentTypeDefinitionRepositoryImpl,
)

POSTGRE_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "app" / "config" / "datasource" / "postgre.yaml"
)

#: 建表只建评测这九张（`create_all` 对已存在的表是 no-op）。
_TABLE_PREFIX = "retrieval_eval_"


def _dsn() -> str:
    raw = yaml.safe_load(POSTGRE_CONFIG.read_text(encoding="utf-8"))
    return raw["config"]["dsn"]


@pytest.fixture
async def session():
    """
    一个用完就回滚的会话，**连同建表**。

    `import_all_models()` 必须先跑：SQLAlchemy 只认识已经 import 过的
    模型，否则 `Base.metadata` 里压根没有这几张表。
    """

    from app.infra.datasource.initializer import import_all_models

    import_all_models()

    engine = create_async_engine(_dsn())

    try:
        connection = await engine.connect()
        await connection.close()

    except Exception as exc:  # noqa: BLE001 —— 连不上就跳过
        await engine.dispose()
        pytest.skip(f"连不上本地 Postgres，跳过逐轮读路径回归：{exc}")

    session = AsyncSession(engine, expire_on_commit=False)

    try:
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
        await session.rollback()
        await session.close()
        await engine.dispose()


# ============================================================
# 造数据：三条 case，三种轮次形状
# ============================================================


class _Fixture:
    """一次用例要用的 id 与行，攒在一起免得来回传参。"""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


async def _seed(session: AsyncSession) -> _Fixture:
    """
    直接写库造一轮评测（不走出题那一步：那要钱，也与本轮要验的东西无关）。

    用**表的 Core 接口**而不是手写 INSERT：列名跟着表走，表改名时手写的
    那份会静默指到别处（见 evaluate 冒烟的同一段说明）。
    """

    bundle_id = uuid.uuid4()
    run_id = uuid.uuid4()
    dtd_id = uuid.uuid4()
    other_dtd_id = uuid.uuid4()

    case_a, case_b, case_c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    await session.execute(
        Base.metadata.tables["retrieval_eval_bundle"].insert().values(
            bundle_id=bundle_id,
            name="逐轮回归数据集",
            status="ready",
            schema_version="1.0",
            selection_config={},
            generation_config={},
            source_chunk_count=3,
            case_count=3,
            dataset_sha256="0" * 64,
        )
    )

    case_model = Base.metadata.tables["retrieval_eval_case"]

    for order, (case_id, case_dtd) in enumerate(
        [(case_c, dtd_id), (case_b, dtd_id), (case_a, other_dtd_id)]
    ):
        await session.execute(
            case_model.insert().values(
                eval_case_id=case_id,
                bundle_id=bundle_id,
                document_type_definition_id=case_dtd,
                source_chunk_id=uuid.uuid4(),
                source_content_hash="a" * 64,
                question=f"第 {order} 题？",
                answer_original="答案",
                answer_start_offset=0,
                answer_end_offset=2,
                question_order=order,
                question_sha256=uuid.uuid4().hex * 2,
                validation_status="validated",
            )
        )

    runs = RetrievalEvalRunRepositoryImpl(session=session)

    await runs.add_run(
        RetrievalEvalRun(
            eval_run_id=run_id,
            bundle_id=bundle_id,
            total_cases=3,
            max_k=10,
        )
    )

    def _result(
            case_id: UUID,
            attempt_no: int,
            hit: bool | None,
            rank: int | None,
    ) -> RetrievalEvalCaseResult:

        # ⚠️ `retrieved_chunk_ids` **非空**是刻意的：折叠只读标量
        # （`list_case_result_briefs`），整行要按 `(case, 轮次)` 再取一次。
        # 这里若留空，两步串错（拿 brief 当整行用）也照样绿 —— 而那正是
        # 「列表里没有候选与分数」这个 bug 的样子。
        return RetrievalEvalCaseResult(
            eval_case_result_id=uuid.uuid4(),
            eval_run_id=run_id,
            eval_case_id=case_id,
            attempt_no=attempt_no,
            # 每轮的"实际发出的文本"刻意与轮次号绑在一起：读路径若把某一
            # 轮的 query_text 安到另一轮头上，只有这种可辨认的值看得出来。
            query_text=f"第 {attempt_no} 轮实际发出的问题 2007-04-1{attempt_no}",
            gold_hit=hit,
            gold_rank=rank,
            retrieved_chunk_ids=[
                {
                    "chunk_id": str(uuid.uuid4()),
                    "rank": 1 + attempt_no,
                    "score": 0.5 + attempt_no / 100,
                    "score_name": "rrf",
                    "source_key": "",
                    "metadata": {"attempt": attempt_no},
                }
            ],
            latency_ms=10 * attempt_no,
        )

    # A：三轮都没命中（同级 -> 代表行取最晚那轮）
    # B：第 1 轮没中、第 2 轮命中（代表行是第 2 轮，而库里有一行 hit=False）
    # C：第 1 轮就命中
    await runs.add_case_results(
        [
            _result(case_a, 1, False, None),
            _result(case_a, 2, False, None),
            _result(case_a, 3, False, None),
            _result(case_b, 1, False, None),
            _result(case_b, 2, True, 4),
            _result(case_c, 1, True, 2),
        ]
    )

    return _Fixture(
        bundle_id=bundle_id,
        run_id=run_id,
        dtd_id=dtd_id,
        other_dtd_id=other_dtd_id,
        case_a=case_a,
        case_b=case_b,
        case_c=case_c,
        runs=runs,
    )


def _service(session: AsyncSession) -> RetrievalEvalServiceImpl:
    """
    真实现的读路径装配。

    ⚠️ 用**真仓储**而不是替身：本文件要验的正是「真库上一条 case 有多行
    时这两条读路径怎么答」，替身一替就把被验的东西替掉了。
    """

    return RetrievalEvalServiceImpl(
        # 这三条读路径只用得到 case 与 run 两个仓储；另两个只为装配齐，
        # 也一律用**真仓储**构造 —— 替身会让"读路径哪天多查一次表"静默
        # 绿掉。
        bundle_repository=RetrievalEvalBundleRepositoryImpl(session=session),
        case_repository=RetrievalEvalCaseRepositoryImpl(session=session),
        run_repository=RetrievalEvalRunRepositoryImpl(session=session),
        chunk_repository=DocumentChunkRepositoryImpl(session=session),
        type_definition_repository=DocumentTypeDefinitionRepositoryImpl(
            session=session
        ),
    )


# ============================================================
# 仓储：一条 case 的多行必须**原样**取回来
# ============================================================


async def test_trace_lookup_returns_every_round(session):
    """
    这是本次 500 的直接回归：两条行在库里时取一行 = MultipleResultsFound。

    顺序也必须按轮次升序 —— 调用方按它取「第 N 轮」与「代表行」，
    靠库的返回顺序就成了随机行为。
    """

    data = await _seed(session)

    rows = await data.runs.list_results_for_trace(
        data.run_id, data.case_a,
    )

    assert [row.attempt_no for row in rows] == [1, 2, 3]
    assert [row.gold_hit for row in rows] == [False, False, False]

    single = await data.runs.list_results_for_trace(data.run_id, data.case_c)

    assert [row.attempt_no for row in single] == [1]

    missing = await data.runs.list_results_for_trace(
        data.run_id, uuid.uuid4(),
    )

    assert missing == []


async def test_briefs_carry_no_heavy_columns(session):
    """
    折叠用的那一读**不能**把候选带回来 —— 它存在的理由就是这个。

    实测每行的 `retrieved_chunk_ids` 均值 7.8 KB（最大 20 KB），而折叠只看
    命中与名次。1000 条 case 三轮就是二十多 MB，为的只是丢掉其中绝大部分。
    哪天有人把这里改回整行读取，这条会红。
    """

    data = await _seed(session)

    briefs = await data.runs.list_case_result_briefs(data.run_id)

    assert len(briefs) == 6
    assert all(brief.retrieved_chunk_ids == [] for brief in briefs)

    # 折叠要用的那几列必须在。
    assert {brief.attempt_no for brief in briefs} == {1, 2, 3}
    assert sum(1 for brief in briefs if brief.gold_hit) == 2


async def test_list_case_results_counts_rows_not_cases(session):
    """
    `list_case_results` 是**行**级接口（总数 = 行数），折叠是调用方的事。

    这条把口径钉死：谁要「一条 case 一行」，谁就必须自己折叠 ——
    而不是指望这个接口顺手去重。
    """

    data = await _seed(session)

    rows, total = await data.runs.list_case_results(
        eval_run_id=data.run_id, offset=0, limit=100,
    )

    assert total == 6
    assert len(rows) == 6

    only_first, first_total = await data.runs.list_case_results(
        eval_run_id=data.run_id, attempt_no=1, offset=0, limit=100,
    )

    assert first_total == 3
    assert {row.eval_case_id for row in only_first} == {
        data.case_a, data.case_b, data.case_c,
    }


# ============================================================
# 服务：单条痕迹
# ============================================================


async def test_trace_defaults_to_the_representative_round(session):
    """
    不指定轮次 = 代表行那一轮，与列表、表头同一条。

    A 三类都没中（同级取更晚的第 3 轮）、B 第 2 轮命中（取第 2 轮）、
    C 第 1 轮就命中（只有第 1 轮）。
    """

    data = await _seed(session)
    service = _service(session)

    trace_a = await service.get_case_trace(data.run_id, data.case_a)
    trace_b = await service.get_case_trace(data.run_id, data.case_b)
    trace_c = await service.get_case_trace(data.run_id, data.case_c)

    assert trace_a.attempt_no == 3
    assert trace_a.available_attempts == [1, 2, 3]

    assert trace_b.attempt_no == 2
    assert trace_b.available_attempts == [1, 2]

    assert trace_c.attempt_no == 1
    assert trace_c.available_attempts == [1]


async def test_trace_can_pin_a_round(session):
    """指定轮次就看那一轮；没有那一轮是 404，不是空痕迹。"""

    from app.common.exceptions import NotFoundError

    data = await _seed(session)
    service = _service(session)

    first = await service.get_case_trace(
        data.run_id, data.case_a, attempt_no=1,
    )

    assert first.attempt_no == 1
    # 上一行是第 1 轮的结果行，看一眼它挂的观察/候选是那一行的
    # （这里是空痕迹，但读的必须是第 1 行而不是第 3 行）。
    assert first.available_attempts == [1, 2, 3]

    with pytest.raises(NotFoundError):
        await service.get_case_trace(
            data.run_id, data.case_b, attempt_no=3,
        )

    with pytest.raises(NotFoundError):
        await service.get_case_trace(data.run_id, uuid.uuid4())


# ============================================================
# 服务：case 列表
# ============================================================


async def test_case_list_is_one_row_per_case(session):
    """
    列表一条 case 一行，`total` 是 case 数 —— 不是结果行数（6）。
    """

    data = await _seed(session)
    service = _service(session)

    response = await service.list_run_cases(data.run_id)

    assert response.total == 3
    assert len(response.items) == 3

    ids = [item.eval_case_id for item in response.items]

    assert len(set(ids)) == 3
    assert set(ids) == {data.case_a, data.case_b, data.case_c}

    by_case = {item.eval_case_id: item for item in response.items}

    assert by_case[data.case_a].attempt_no == 3
    assert by_case[data.case_b].attempt_no == 2
    assert by_case[data.case_c].attempt_no == 1

    # 命中数是按 **case** 数的（B、C），不是按行的。
    assert [item.gold_hit for item in response.items].count(True) == 2

    # 候选与分数来自**整行**（折叠时读的 brief 里没有这些列）。少这步，
    # 列表页会一排排地"没有 top 候选"，而每个页面单看都像没召回到。
    assert all(len(item.top_candidates) == 1 for item in response.items)
    assert all(item.retrieved_chunk_ids for item in response.items)

    # 而且取的是**代表行那一轮**的候选，不是随便哪一轮的：B 的代表行是
    # 第 2 轮，它的候选名次是 1+2=3（见 `_result` 的造法）。
    assert by_case[data.case_b].top_candidates[0].rank == 3

    # 实际发出的文本跟着**那一行**走（B 是第 2 轮，A 是第 3 轮）—— 它是
    # "这条命中的是哪句话"的唯一答案，安错轮次等于把证据指错。
    assert by_case[data.case_b].query_text == "第 2 轮实际发出的问题 2007-04-12"
    assert by_case[data.case_a].query_text == "第 3 轮实际发出的问题 2007-04-13"


async def test_trace_carries_the_rounds_query_text(session):
    """
    痕迹页也要带**那一轮**的 query_text。

    这份痕迹是"这一轮怎么跑的"，而不说清"这一轮问的是什么"，第 1 轮与
    第 2 轮的痕迹只能靠猜（两轮的 query 不同）。
    """

    data = await _seed(session)
    service = _service(session)

    second = await service.get_case_trace(data.run_id, data.case_b, attempt_no=2)
    first = await service.get_case_trace(data.run_id, data.case_b, attempt_no=1)

    assert second.query_text == "第 2 轮实际发出的问题 2007-04-12"
    assert first.query_text == "第 1 轮实际发出的问题 2007-04-11"


async def test_case_list_filters_after_collapsing(session):
    """
    `hit` 筛的是**折叠后**的结论。

    B 在库里有一行 `gold_hit=false`（第 1 轮没中），而它的结论是命中 ——
    按行筛的话它会在 `hit=true` 里缺席、又在 `hit=false` 里冒出来，与列表
    自己显示的那一行自相矛盾。
    """

    data = await _seed(session)
    service = _service(session)

    hits = await service.list_run_cases(data.run_id, hit=True)
    misses = await service.list_run_cases(data.run_id, hit=False)

    assert {item.eval_case_id for item in hits.items} == {
        data.case_b, data.case_c,
    }
    assert hits.total == 2

    assert {item.eval_case_id for item in misses.items} == {data.case_a}
    assert misses.total == 1
    assert all(item.gold_hit is False for item in misses.items)


async def test_case_list_filters_by_document_type(session):
    """DTD 过滤同样在折叠之后（每案只有一行，不会漏案也不会重复）。"""

    data = await _seed(session)
    service = _service(session)

    mine = await service.list_run_cases(
        data.run_id, document_type_definition_id=data.dtd_id,
    )

    assert {item.eval_case_id for item in mine.items} == {
        data.case_b, data.case_c,
    }
    assert mine.total == 2

    other = await service.list_run_cases(
        data.run_id, document_type_definition_id=data.other_dtd_id,
    )

    assert {item.eval_case_id for item in other.items} == {data.case_a}
    assert other.total == 1


async def test_case_list_pages_over_cases(session):
    """
    分页翻的是 **case**：`total` 若是行数（6），翻到底也看不到最后一条 case。
    """

    data = await _seed(session)
    service = _service(session)

    first = await service.list_run_cases(data.run_id, page=1, page_size=2)
    second = await service.list_run_cases(data.run_id, page=2, page_size=2)

    assert first.total == second.total == 3
    assert len(first.items) == 2
    assert len(second.items) == 1

    seen = {item.eval_case_id for item in first.items} | {
        item.eval_case_id for item in second.items
    }

    assert seen == {data.case_a, data.case_b, data.case_c}


async def test_case_list_skips_rows_whose_case_is_gone(session):
    """
    结果行指向一条读不出来的 case（不该发生）：跳过它，**而且不计进 total**
    —— 记进去的话末页会是空的，看起来像"分页坏了"。

    直接删 case 行来造这个形状（结果行对 case 是 RESTRICT，所以先删结果行；
    那正是"结果还在、case 没了"这件事在真实世界里的到来方式）。
    """

    data = await _seed(session)
    service = _service(session)

    await session.execute(
        text(
            "DELETE FROM retrieval_eval_case_result "
            "WHERE eval_case_id = :case_id"
        ),
        {"case_id": str(data.case_c)},
    )
    await session.execute(
        text("DELETE FROM retrieval_eval_case WHERE eval_case_id = :case_id"),
        {"case_id": str(data.case_c)},
    )

    response = await service.list_run_cases(data.run_id)

    assert response.total == 2
    assert {item.eval_case_id for item in response.items} == {
        data.case_a, data.case_b,
    }
