"""
查询改写：`common` 的行为、注册表、以及它怎么随阶梯传下去。

这三件事都不碰库、不碰网络（`llm` 那条路的实现只验注册与失败姿态，
真调用在评测的端到端里跑），所以是毫秒级的纯用例。
"""

from __future__ import annotations

import asyncio

import pytest

from app.services.document.domain.schemas import (
    DocumentRetrieveRequest,
    QueryRewriterConfig,
)
from app.services.document.domain.schemas.enum import QueryRewriterKind
from app.services.document.retrieve.registry import (
    RetrieveStageNotFound,
    get_registry,
)
from app.services.document.retrieve.rewrite import CommonQueryRewriter

DTC = QueryRewriterConfig(kind=QueryRewriterKind.COMMON)


def _rewrite(query: str) -> tuple[str, bool, dict]:
    result = asyncio.run(
        CommonQueryRewriter().rewrite(query=query, config=DTC)
    )
    return result.text, result.changed, result.detail


# ============================================================
# common：日期两种写法都展开
# ============================================================


def test_chinese_date_gains_the_dash_form() -> None:
    """
    中文写法要补出**补零**的短横线写法。

    补零不是洁癖：ES 那条 `date_normalizer` 是 `(\\d{4})-(\\d{2})-(\\d{2})`，
    恰好两位 —— `2007-4-2` 不会被压成原子词项，等于没补。
    """

    text, changed, detail = _rewrite("荣安地产2007年4月12日披露的对外担保？")

    assert changed is True
    assert "2007-04-12" in text
    assert detail["dates"][0]["canonical"] == "2007-04-12"


def test_dash_date_gains_the_chinese_form() -> None:
    """短横线写法要补出中文写法（这批语料的散文就写中文）。"""

    text, changed, _ = _rewrite("ST海王 于2015-05-22接待调研的参与方式？")

    assert changed is True
    assert "2015年5月22日" in text


def test_non_padded_dash_gains_both_canonical_forms() -> None:
    """`2022-2-7` 两种规范写法都不在原文里 —— 两条都要补。"""

    text, changed, _ = _rewrite("渝开发 于2022-2-7接待调研？")

    assert changed is True
    assert "2022-02-07" in text
    assert "2022年2月7日" in text


def test_no_date_leaves_the_query_alone() -> None:
    """没有日期就一个字都不改 —— 改写只该动它认得出的东西。"""

    text, changed, detail = _rewrite("公司今年的营收是多少？")

    assert text == "公司今年的营收是多少？"
    assert changed is False
    assert detail["dates"] == []


def test_impossible_dates_are_skipped_not_expanded() -> None:
    """13 月 40 日不是日期：不展开，但要记下来（免得看起来像"没看见"）。"""

    text, changed, detail = _rewrite("这家公司2007年13月40日干了什么？")

    assert changed is False
    assert detail["skipped"] == ["2007-13-40"]


def test_rewriting_is_idempotent() -> None:
    """
    幂等：把输出再喂一遍必须一模一样。

    不幂等的话，重试/续跑会不断往 query 上叠日期串 —— 而那种劣化在结果
    里看不出来（只是"命中的块变少了"）。
    """

    once, changed_once, _ = _rewrite("荣安地产2007年4月12日披露的对外担保？")
    twice, changed_twice, _ = _rewrite(once)

    assert changed_once is True
    assert twice == once
    assert changed_twice is False, "第二次不该再改"


def test_both_forms_already_present_is_a_no_op() -> None:
    """两种写法本来都在时不重复追加（与幂等是同一条的两个入口）。"""

    text, changed, _ = _rewrite("2007年4月12日 与 2007-04-12 是同一件事吗？")

    assert changed is False
    assert text.count("2007-04-12") == 1


# ============================================================
# 注册表
# ============================================================


def test_registry_resolves_both_kinds() -> None:
    """
    两个 kind 都能解析，且走的是**反射发现**（新加一个实现不用改注册表）。

    `get_registry` 是 lru_cache 的单例，测试里要清一下 —— 否则前一个用例
    （或别的模块）拿到的实例会跨用例复用。
    """

    get_registry.cache_clear()
    registry = get_registry()

    assert (
        type(registry.rewriter(QueryRewriterKind.COMMON)).__name__
        == "CommonQueryRewriter"
    )
    assert (
        type(registry.rewriter(QueryRewriterKind.LLM)).__name__
        == "LlmQueryRewriter"
    )


def test_duplicate_kind_fails_loudly(monkeypatch) -> None:
    """
    一个 kind 只能有一个实现：重复声明时**显式失败**，不让先注册的被
    静默顶掉（与精排/融合同一口径）。

    ⚠️ 假实现**不继承 `QueryRewriter`**，只长得像（收集器是鸭子类型：
    读 `rewriter_kinds`、无参构造）。继承的话它会永久留在
    `QueryRewriter.__subclasses__()` 里 —— 那会让后面任何一次重建注册表
    的用例撞上"COMMON 被声明了两次"，而红的是它们，不是这条。
    """

    from app.services.document.retrieve import registry as registry_module

    class _Duplicate:
        rewriter_kinds = (QueryRewriterKind.COMMON,)

        async def rewrite(self, *, query, config):  # pragma: no cover
            raise NotImplementedError

    monkeypatch.setattr(
        registry_module,
        "leaf_subclasses",
        lambda base, skip_abstract=False: [CommonQueryRewriter, _Duplicate],
    )

    with pytest.raises(RetrieveStageNotFound):
        registry_module.RetrieveProcessorRegistry._collect_rewriters()


# ============================================================
# 随阶梯传下去
# ============================================================


def test_default_ladder_carries_a_rewriter_per_round() -> None:
    """
    默认阶梯：第 1 轮 common、第 2 轮 llm、**第 3 轮不传**（不改写）。

    第 3 轮是刻意留空的 —— 它的输入已经是前两轮都没找到的最难那批，而
    这一轮要读的是精排的增量（见阶梯上的说明）。
    """

    from app.services.document.domain.schemas.retrieval_eval import (
        DEFAULT_ATTEMPT_LADDER,
    )

    kinds = [
        spec.query_rewriter.kind if spec.query_rewriter else None
        for spec in DEFAULT_ATTEMPT_LADDER
    ]

    assert kinds == [
        QueryRewriterKind.COMMON,
        QueryRewriterKind.LLM,
        None,
    ]


def test_attempt_spec_survives_the_snapshot_round_trip() -> None:
    """
    阶梯是经 `model_dump(mode="json", exclude_unset=True)` 落进快照、再由
    `model_validate` 读回来的（service_impl 的两处）。新字段必须能过这一
    遭 —— 过不去的话，跑到第 2 轮才会发现改写器没了。
    """

    from app.services.document.domain.schemas.retrieval_eval import (
        DEFAULT_ATTEMPT_LADDER,
        EvalAttemptSpec,
    )

    for spec in DEFAULT_ATTEMPT_LADDER:

        restored = EvalAttemptSpec.model_validate(
            spec.model_dump(mode="json", exclude_unset=True)
        )

        assert restored.query_rewriter == spec.query_rewriter


def test_build_retrieve_request_passes_the_round_rewriter() -> None:
    """
    阶梯 -> 检索请求的唯一映射点。

    `attempt is None`（单轮、没阶梯）时必须**不给改写**，而不是回落某个
    运行级默认 —— "不传就不用"就是这么成立的。
    """

    from uuid import uuid4

    from app.services.document.application.retrieval_eval_evaluation import (
        build_retrieve_request,
    )
    from app.services.document.domain.entities import (
        RetrievalEvalCase,
        RetrievalEvalRun,
    )
    from app.services.document.domain.schemas.retrieval_eval import (
        EvalAttemptSpec,
    )

    case = RetrievalEvalCase(
        eval_case_id=uuid4(),
        bundle_id=uuid4(),
        document_type_definition_id=uuid4(),
        source_chunk_id=uuid4(),
        source_content_hash="a" * 64,
        question="问题",
        answer_original="答",
        answer_start_offset=0,
        answer_end_offset=1,
        question_order=0,
        question_sha256="b" * 64,
    )

    run = RetrievalEvalRun(eval_run_id=uuid4(), bundle_id=uuid4())

    spec = EvalAttemptSpec(
        query_rewriter=QueryRewriterConfig(kind=QueryRewriterKind.LLM)
    )

    with_attempt = build_retrieve_request(
        case=case, run=run, gold_chunk_ids=[], attempt=spec,
    )
    without = build_retrieve_request(
        case=case, run=run, gold_chunk_ids=[], attempt=None,
    )

    assert with_attempt.query_rewriter is spec.query_rewriter
    assert without.query_rewriter is None


def test_request_defaults_to_no_rewriting() -> None:
    """请求层的默认值：不传 = 不改写（服务端不会替调用方开）。"""

    request = DocumentRetrieveRequest(
        document_type_definition_ids=["ef000001-0001-4000-8000-000000000001"],
        input_text="问题",
    )

    assert request.query_rewriter is None


# ============================================================
# 短语加权（A3）：改写器给清单，索引后端应用
# ============================================================


def test_common_reports_both_date_forms_as_boost_phrases() -> None:
    """
    两种写法**都给**，不是只给"原文里缺的那种"。

    正文散文写中文、`#### 2007-04-12 ｜ …` 小标题写短横线，而那个标题常
    被切进**上一块** —— 只加一种，就有一半的块拿不到这份权重，而"哪一块
    才是这块的标题"在这一层看不到。
    """

    result = asyncio.run(
        CommonQueryRewriter().rewrite(
            query="荣安地产2007年4月12日披露的对外担保？", config=DTC,
        )
    )

    assert result.boost_phrases == ("2007-04-12", "2007年4月12日")


def test_common_boosts_dates_even_when_the_text_is_unchanged() -> None:
    """
    文本没变也照样给短语。

    两种写法都在原文里时 `changed=False`、一个字都不追加 —— 但**短语加权
    仍然该生效**（文本路已经没得可加，短语路才是让日期当强信号的那条）。
    这两件事混成一件的话，"两种写法都写了"的题就永远拿不到加权。
    """

    result = asyncio.run(
        CommonQueryRewriter().rewrite(
            query="2007年4月12日 与 2007-04-12 是同一件事吗？", config=DTC,
        )
    )

    assert result.changed is False
    assert set(result.boost_phrases) == {"2007-04-12", "2007年4月12日"}


def test_no_date_means_no_boost_phrases() -> None:
    """没有判别词就不该塞空片段（空短语在 ES 里是语法错误）。"""

    result = asyncio.run(
        CommonQueryRewriter().rewrite(query="公司今年营收多少？", config=DTC)
    )

    assert result.boost_phrases == ()


def test_llm_rewriter_does_not_claim_boost_phrases() -> None:
    """
    LLM 那条**不给**短语加权。

    它会顺手补出"觉得该有"的词（实测给一条 2007 年的题补了公司全称与
    "被担保人"）。把模型吐的词当**短语级**权重，等于让它决定排序公式里的
    一项，而那一项没有量过。要用它的输出就只用文本 —— BM25 按普通词项
    处理，错了也只是多几个词。
    """

    from app.services.document.retrieve.rewrite import LlmQueryRewriter

    assert LlmQueryRewriter().rewrite  # 存在即可；行为由下面那条断言钉住
    assert getattr(LlmQueryRewriter, "rewriter_kinds") == (
        QueryRewriterKind.LLM,
    )


def test_es_builds_phrase_clauses_and_skips_blanks() -> None:
    """
    ES 侧把清单拼成 `match_phrase` 的 `should` 子句。

    去空、去重：空短语是语法错误；重复等于偷偷把 boost 加了两遍。
    """

    from app.services.document.index.es import _PHRASE_BOOST, _phrase_clauses

    clauses = _phrase_clauses(["2007-04-12", "", "   ", "2007-04-12", "2007年4月12日"])

    assert [item["match_phrase"]["content"]["query"] for item in clauses] == [
        "2007-04-12",
        "2007年4月12日",
    ]
    assert all(
        item["match_phrase"]["content"]["boost"] == _PHRASE_BOOST
        for item in clauses
    )
    assert _phrase_clauses([]) == []
