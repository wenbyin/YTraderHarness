"""
`llm` 精排的映射与失败姿态（不碰网络）。

盯的三件事，与批量改写那组同源：

    编号怎么映射        按 `index` 配，不是按列表顺序（串号 = 分数挂错 chunk）
    对不上怎么办        降级（原序返回 + degraded），绝不按位置硬配
    请求形状            关思考（输出形状由提示词给定）+ 候选条数

真调用验证过一次：把正确答案从第 3 位提到第 1，同模板的其它期次排在后面，
无关块 0.0 —— 那是"能排对"的证据；这里管的是"排不出错"。
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services.document.domain.schemas.retrieve import RerankerConfig
from app.services.document.domain.schemas.enum import RerankerType
from app.services.document.retrieve.base import RetrieveCandidate
from app.services.document.retrieve import rerank_llm
from app.services.document.retrieve.rerank_llm import (
    LlmReranker,
    _window,
)

CONFIG = RerankerConfig(type=RerankerType.LLM)


def _candidate(text: str, score: float = 0.5) -> RetrieveCandidate:
    return RetrieveCandidate(
        document_chunk_id=uuid4(),
        score=score,
        retrieval_source="keyword",
        content=text,
    )


def _patch_reply(monkeypatch, reply: str, captured: dict | None = None):
    """把"调模型"换成一段固定回包（并可选地记下请求）。"""

    class _Runtime:
        async def ainvoke(self, request):
            if captured is not None:
                captured["request"] = request
            return SimpleNamespace(
                message=SimpleNamespace(content=reply),
            )

    monkeypatch.setattr(
        rerank_llm,
        "get_model_registry",
        lambda: SimpleNamespace(get_runtime=lambda **_: _Runtime()),
    )
    monkeypatch.setattr(
        rerank_llm,
        "get_prompt_provider",
        lambda: SimpleNamespace(node_prompt=lambda _role: "系统提示"),
    )


def _reply(items: list[dict]) -> str:
    return json.dumps({"items": items}, ensure_ascii=False)


# ============================================================
# 编号映射
# ============================================================


def test_scores_map_by_index_not_by_position(monkeypatch) -> None:
    """
    按 `index` 配对。模型换序回时，按位置取就会把分数挂到别的候选上 ——
    而那种错看起来"只是一次排序不好"。
    """

    _patch_reply(monkeypatch, _reply([
        {"index": 3, "relevance": 9},
        {"index": 1, "relevance": 1},
        {"index": 2, "relevance": 5},
    ]))

    candidates = [_candidate("甲"), _candidate("乙"), _candidate("丙")]

    out = asyncio.run(LlmReranker().rerank(
        query="问题", candidates=candidates, top_n=3, config=CONFIG,
    ))

    assert [item.content for item in out] == ["丙", "乙", "甲"]
    assert out[0].score == pytest.approx(1.0)
    assert out[-1].score == pytest.approx(1 / 9)


def test_missing_index_degrades(monkeypatch) -> None:
    """少回一条 → 整批降级（原序），不猜哪条缺了。"""

    _patch_reply(monkeypatch, _reply([
        {"index": 1, "relevance": 9},
    ]))

    candidates = [_candidate("甲", 0.9), _candidate("乙", 0.5)]
    degraded: dict = {}

    out = asyncio.run(LlmReranker().rerank(
        query="问题", candidates=candidates, top_n=2,
        config=CONFIG, degraded=degraded,
    ))

    assert [item.content for item in out] == ["甲", "乙"], "原序"
    assert [item.score for item in out] == [0.9, 0.5], "分数不动"
    assert degraded, "降级必须说出来"


def test_duplicate_index_degrades(monkeypatch) -> None:
    """重复编号分不清哪条是哪条 → 降级。"""

    _patch_reply(monkeypatch, _reply([
        {"index": 1, "relevance": 9},
        {"index": 1, "relevance": 2},
    ]))

    degraded: dict = {}

    asyncio.run(LlmReranker().rerank(
        query="问题", candidates=[_candidate("甲"), _candidate("乙")],
        top_n=2, config=CONFIG, degraded=degraded,
    ))

    assert degraded


def test_unparseable_reply_degrades(monkeypatch) -> None:
    """回包不是 JSON → 降级（不抛 —— 精排失败不该让检索失败）。"""

    _patch_reply(monkeypatch, "我觉得第一条最好。")

    degraded: dict = {}

    out = asyncio.run(LlmReranker().rerank(
        query="问题", candidates=[_candidate("甲")], top_n=1,
        config=CONFIG, degraded=degraded,
    ))

    assert len(out) == 1
    assert degraded and "reason" in degraded


# ============================================================
# 窗口与长度
# ============================================================


def test_window_is_capped_by_count(monkeypatch) -> None:
    """`top_n` 再大也被条数上限收窄（它同时是输入规模的闸）。"""

    assert _window([_candidate("x")] * 100, top_n=100) == (
        rerank_llm._MAX_CANDIDATES
    )
    assert _window([_candidate("x")] * 3, top_n=100) == 3
    assert _window([_candidate("x")] * 100, top_n=5) == 5
    assert _window([_candidate("x")], top_n=0) == 1, "至少送一条"


def test_unscored_rest_is_appended_unchanged(monkeypatch) -> None:
    """窗口之外的候选原序接在后面、**不丢**（最终条数归 cutter）。"""

    _patch_reply(monkeypatch, _reply([
        {"index": 2, "relevance": 9},
        {"index": 1, "relevance": 1},
    ]))

    candidates = [_candidate("甲"), _candidate("乙"), _candidate("丙")]

    out = asyncio.run(LlmReranker().rerank(
        query="问题", candidates=candidates, top_n=2, config=CONFIG,
    ))

    assert len(out) == 3, "长度不变"
    assert [item.content for item in out] == ["乙", "甲", "丙"]


def test_empty_candidates_short_circuits() -> None:
    """空输入不进模型。"""

    out = asyncio.run(LlmReranker().rerank(
        query="问题", candidates=[], top_n=5, config=CONFIG,
    ))

    assert out == []


# ============================================================
# 请求形状
# ============================================================


def test_request_asks_for_thinking_off(monkeypatch) -> None:
    """
    这条调用必须带 `disable_thinking=True`。

    输出形状完全由提示词给定（只回一个 JSON），而思考与正文**共用**
    `max_tokens`：十几条候选的输入下不关思考，很容易推理写满、JSON 没写完。
    """

    captured: dict = {}

    _patch_reply(monkeypatch, _reply([{"index": 1, "relevance": 5}]), captured)

    asyncio.run(LlmReranker().rerank(
        query="问题", candidates=[_candidate("甲")], top_n=1, config=CONFIG,
    ))

    request = captured["request"]

    assert request.disable_thinking is True
    assert request.metadata["candidate_count"] == 1
    assert request.metadata["model_role"] == "retrieve.rerank"


def test_prompt_carries_the_candidates_numbered(monkeypatch) -> None:
    """候选是按编号给模型的 —— 编号对不上就没法校验串号。"""

    captured: dict = {}

    _patch_reply(monkeypatch, _reply([
        {"index": 1, "relevance": 5},
        {"index": 2, "relevance": 6},
    ]), captured)

    asyncio.run(LlmReranker().rerank(
        query="某个问题",
        candidates=[_candidate("第一段正文"), _candidate("第二段正文")],
        top_n=2, config=CONFIG,
    ))

    human = captured["request"].messages[1].content

    assert "某个问题" in human
    assert "--- 1 ---" in human and "--- 2 ---" in human
    assert "第一段正文" in human and "第二段正文" in human
