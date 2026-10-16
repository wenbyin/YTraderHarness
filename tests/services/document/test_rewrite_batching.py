"""
批量改写的**串号防线**与它的接线（不碰网络）。

为什么单独一组用例
------------------
批量最危险的失败不是"出错"，是**把甲的改写安到乙头上**：返回的每一条都
读起来合理，只有对照原文才知道错了。所以这里盯的是三件事：

    模型按编号回了吗        回包校验（条数 / 编号 / 重复）
    校验不过怎么办          回落逐条，而不是按错位的位置硬配
    关思考有没有带上        这条调用的输出形状由提示词给定，
                            `disable_thinking` 是它不被截断的前提

真调用不在这里（要钱、要网、还可能因为对端改版在别人机器上红）——
一次真实调用验证过：3 条 → 一次调用，prompt 699 / completion 127 token，
`completion_tokens_details` 为 None（推理确实关了）。
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services.document.application.retrieval_eval_service_impl import (
    _precompute_rewrites,
)
from app.services.document.domain.schemas import (
    QueryRewriteInput,
    QueryRewriterConfig,
)
from app.services.document.domain.schemas.enum import QueryRewriterKind
from app.services.document.retrieve.rewrite import (
    CommonQueryRewriter,
    LlmQueryRewriter,
)

LLM = QueryRewriterConfig(kind=QueryRewriterKind.LLM)


def _reply(items: list[dict]) -> SimpleNamespace:
    """一条假的 ModelResponse —— 只带解析要看的那部分。"""

    message = SimpleNamespace(
        content=json.dumps({"items": items}, ensure_ascii=False),
    )

    return SimpleNamespace(message=message)


def _patch_invoke(monkeypatch, reply) -> None:
    """把"调模型"换成一段固定的回包，并**记下请求**。"""

    seen: dict = {}

    async def _invoke(self, *, queries, config):
        seen["queries"] = list(queries)
        seen["request"] = self
        return reply

    monkeypatch.setattr(LlmQueryRewriter, "_invoke", _invoke)

    return seen


# ============================================================
# 编号校验：批量最危险的那一类失败
# ============================================================


def test_batch_maps_by_index_not_by_position(monkeypatch) -> None:
    """
    按 `index` 配对，**不是**按列表顺序。

    模型换了个顺序回（这里故意把 2 放在最前）时，按位置取就会串号 ——
    而两条改写读起来都合理，没有任何报错。
    """

    _patch_invoke(
        monkeypatch,
        _reply([
            {"index": 2, "rewritten_query": "乙改", "changed": True},
            {"index": 1, "rewritten_query": "甲改", "changed": True},
        ]),
    )

    results = asyncio.run(
        LlmQueryRewriter().rewrite_many(queries=["甲", "乙"], config=LLM)
    )

    assert [item.text for item in results] == ["甲改", "乙改"]


def test_missing_index_falls_back_to_per_query(monkeypatch) -> None:
    """
    少回一条（编号不齐）→ **整批不用**，回落到逐条。

    按位置硬配的后果是把后面每一条都错位一位 —— 那比"这次没改写"严重得多。
    """

    calls: list[int] = []

    async def _invoke(self, *, queries, config):
        calls.append(len(queries))
        # 第一次（批量）少回一条；逐条之后每次只问一条，正常回。
        if len(queries) > 1:
            return _reply([
                {"index": 1, "rewritten_query": "甲改", "changed": True},
            ])
        return _reply([
            {
                "index": 1,
                "rewritten_query": queries[0] + "改",
                "changed": True,
            },
        ])

    monkeypatch.setattr(LlmQueryRewriter, "_invoke", _invoke)

    results = asyncio.run(
        LlmQueryRewriter().rewrite_many(queries=["甲", "乙"], config=LLM)
    )

    assert [item.text for item in results] == ["甲改", "乙改"]
    assert calls == [2, 1, 1], "批量失败后应当逐条重试（而不是整批放弃）"


def test_duplicate_index_falls_back(monkeypatch) -> None:
    """重复编号同样分不清哪条是哪条 → 回落逐条。"""

    async def _invoke(self, *, queries, config):
        if len(queries) > 1:
            return _reply([
                {"index": 1, "rewritten_query": "甲改"},
                {"index": 1, "rewritten_query": "还是甲改"},
            ])
        return _reply([
            {"index": 1, "rewritten_query": queries[0] + "改"},
        ])

    monkeypatch.setattr(LlmQueryRewriter, "_invoke", _invoke)

    results = asyncio.run(
        LlmQueryRewriter().rewrite_many(queries=["甲", "乙"], config=LLM)
    )

    assert [item.text for item in results] == ["甲改", "乙改"]


def test_total_failure_degrades_to_the_original(monkeypatch) -> None:
    """
    连逐条都失败 → 每条**退回原文 + degraded**（绝不抛）。

    契约就是"改写失败不该让检索失败"；而 `degraded=True` 让调用方分得清
    "没改写"与"想改没改成"。
    """

    async def _invoke(self, *, queries, config):
        raise RuntimeError("模型挂了")

    monkeypatch.setattr(LlmQueryRewriter, "_invoke", _invoke)

    results = asyncio.run(
        LlmQueryRewriter().rewrite_many(queries=["甲", "乙"], config=LLM)
    )

    assert [item.text for item in results] == ["甲", "乙"]
    assert all(item.degraded for item in results)
    assert all(item.changed is False for item in results)


def test_batch_asks_for_thinking_off(monkeypatch) -> None:
    """
    这条调用必须带 `disable_thinking=True`。

    它的输出形状完全由提示词给定（只回一个 JSON），而思考与正文**共用**
    `max_tokens`：不关的话推理占约九成，批量时条数一多正文就写不完
    （`Unterminated string`）。实测关掉后同一条提示词的输出是 9 token。
    """

    captured: dict = {}

    # ⚠️ **不能**把 `_invoke` 也替换掉 —— 请求就是在它里面构造的，替掉它
    # 等于把要验的东西替掉了。这里只替 runtime 与 prompt provider，
    # 让真的 `_invoke` 跑一遍。
    class _Runtime:
        async def ainvoke(self, request):
            captured["request"] = request
            return _reply([{"index": 1, "rewritten_query": "甲改"}])

    monkeypatch.setattr(
        "app.services.document.retrieve.rewrite.get_model_registry",
        lambda: SimpleNamespace(
            get_runtime=lambda **_kwargs: _Runtime(),
        ),
    )
    monkeypatch.setattr(
        "app.services.document.retrieve.rewrite.get_prompt_provider",
        lambda: SimpleNamespace(node_prompt=lambda _role: "系统提示"),
    )

    asyncio.run(LlmQueryRewriter().rewrite(query="甲", config=LLM))

    request = captured["request"]

    assert request.disable_thinking is True
    assert request.metadata["batch_size"] == 1


# ============================================================
# 评测侧：什么时候批量、批量结果怎么配对
# ============================================================


def _case(question: str):
    return SimpleNamespace(eval_case_id=uuid4(), question=question)


def test_precompute_is_skipped_for_non_model_kinds() -> None:
    """
    `common` 不调模型 → **不做**预处理（交给检索侧现算）。

    两处都做的话，"预处理过的 common"与"服务里现算的 common"就成了两份
    实现，而它们只在日期形态这种细节上分叉 —— 表现是"某些轮次的结果
    对不上"，很难查。
    """

    cases = [_case("甲"), _case("乙")]

    assert asyncio.run(_precompute_rewrites(
        cases, QueryRewriterConfig(kind=QueryRewriterKind.COMMON),
    )) == {}
    assert asyncio.run(_precompute_rewrites(cases, None)) == {}


def test_precompute_pairs_results_with_cases(monkeypatch) -> None:
    """批量结果按**位置**配回 case（顺序由 `rewrite_many` 保证）。"""

    async def _rewrite_many(self, *, queries, config):  # noqa: ARG001
        from app.services.document.retrieve.base import RewriteResult

        return [
            RewriteResult(
                text=query + "改",
                changed=True,
                boost_phrases=("p",),
                date_keys=("20070412",),
            )
            for query in queries
        ]

    monkeypatch.setattr(LlmQueryRewriter, "rewrite_many", _rewrite_many)
    monkeypatch.setattr(
        "app.services.document.retrieve.registry.get_registry",
        lambda: SimpleNamespace(
            rewriter=lambda _kind: LlmQueryRewriter(),
        ),
    )

    first, second = _case("甲"), _case("乙")

    precomputed = asyncio.run(
        _precompute_rewrites([first, second], LLM)
    )

    assert precomputed[first.eval_case_id].text == "甲改"
    assert precomputed[second.eval_case_id].text == "乙改"
    assert precomputed[first.eval_case_id].date_keys == ["20070412"]


def test_precompute_gives_up_when_lengths_disagree(monkeypatch) -> None:
    """
    条数对不上 → **整体放弃预处理**（返回空 dict），绝不按错位配。

    放弃之后这些 case 会走服务自己那条路（逐条），而不是带着错位的改写
    去检索 —— 后者会让"这条 case 的结果"对应到别人的 query 上。
    """

    async def _rewrite_many(self, *, queries, config):  # noqa: ARG001
        from app.services.document.retrieve.base import RewriteResult

        return [RewriteResult(text="只回了一条")]

    monkeypatch.setattr(LlmQueryRewriter, "rewrite_many", _rewrite_many)
    monkeypatch.setattr(
        "app.services.document.retrieve.registry.get_registry",
        lambda: SimpleNamespace(
            rewriter=lambda _kind: LlmQueryRewriter(),
        ),
    )

    assert asyncio.run(_precompute_rewrites([_case("甲"), _case("乙")], LLM)) == {}


def test_request_carries_either_rewriter_or_precomputed() -> None:
    """
    请求里**只带一个**：给了预处理结果就不再带改写器。

    两个都带时服务只会用一个（另一个静默失效）—— 契约上直接避免这种组合。
    """

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

    precomputed = QueryRewriteInput(text="问题 2007-04-12")

    request = build_retrieve_request(
        case=case, run=run, gold_chunk_ids=[],
        attempt=spec, query_rewrite=precomputed,
    )

    assert request.query_rewrite is precomputed
    assert request.query_rewriter is None, "两者不能同时带"


def test_common_rewriter_batch_matches_single(monkeypatch) -> None:
    """默认的批量实现就是逐条 —— 对不调模型的实现，这就是最优解。"""

    queries = ["甲 2007年4月12日", "乙没有日期"]

    batched = asyncio.run(
        CommonQueryRewriter().rewrite_many(
            queries=queries,
            config=QueryRewriterConfig(kind=QueryRewriterKind.COMMON),
        )
    )

    singles = [
        asyncio.run(CommonQueryRewriter().rewrite(
            query=query,
            config=QueryRewriterConfig(kind=QueryRewriterKind.COMMON),
        ))
        for query in queries
    ]

    assert [item.text for item in batched] == [item.text for item in singles]
