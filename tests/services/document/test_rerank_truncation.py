"""
精排**只改变顺序，不决定条数**。

这条不变量此前是破的：`rerank.py` 的末行是
`normalize_scores((ranked + rest)[:top_n])` —— 于是一个 `cutter.top_k=30`
的轮次里，`top_n=10` 会把结果压到 10 条，排在 11~30 的候选**再也回不来**。
它很难被发现：结果页上只是"条目变少了"，看起来像裁剪配置的功劳。

这里用打桩的打分函数验（不加载任何模型、不碰网络）。
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.services.document.domain.schemas import RerankerConfig
from app.services.document.domain.schemas.enum import RerankerType
from app.services.document.retrieve.base import RetrieveCandidate
from app.services.document.retrieve.rerank import (
    FlagEmbeddingReranker,
    PassthroughReranker,
    _MAX_CANDIDATES,
)

CONFIG = RerankerConfig(
    type=RerankerType.CROSS_ENCODER, model="stub", top_n=None,
)


class _StubReranker(FlagEmbeddingReranker):
    """
    把打分换成一段固定分数（不加载模型）。

    ⚠️ `reranker_types = ()` 是必须的：父类声明了 CROSS_ENCODER，而注册表
    是按 `leaf_subclasses` 反射收集的 —— 不显式清空的话，这个测试替身会
    被当成一个真的精排实现注册进去，于是**后面任何重建注册表的用例**都会
    撞上"cross_encoder 被声明了两次"，而红的是它们，不是这条。
    """

    reranker_types = ()

    def __init__(self, scores: list[float] | None) -> None:
        self._scores = scores
        self.scored_count = 0

    async def _score(self, query, candidates, config):  # noqa: ANN001
        self.scored_count = len(candidates)
        return self._scores


def _candidates(count: int) -> list[RetrieveCandidate]:
    return [
        RetrieveCandidate(
            document_chunk_id=uuid4(),
            score=1.0 - index / (count + 1),
            retrieval_source="keyword",
            content=f"第 {index} 段",
        )
        for index in range(count)
    ]


async def test_output_keeps_every_candidate() -> None:
    """输入 30 条、`top_n=5` —— 输出还是 30 条（旧版会只剩 5 条）。"""

    reranker = _StubReranker([0.9, 0.8, 0.7, 0.6, 0.5])
    candidates = _candidates(30)

    out = await reranker.rerank(
        query="问题", candidates=candidates, top_n=5, config=CONFIG,
    )

    assert len(out) == 30
    assert reranker.scored_count == 5, "只给送进来的那 5 段打分"


async def test_unscored_tail_keeps_fusion_order() -> None:
    """没被打分的那截接在后面，且保持融合顺序（它们没被比较过）。"""

    reranker = _StubReranker([0.1, 0.2])
    candidates = _candidates(5)

    out = await reranker.rerank(
        query="问题", candidates=candidates, top_n=2, config=CONFIG,
    )

    assert [item.content for item in out[2:]] == ["第 2 段", "第 3 段", "第 4 段"]


async def test_scoring_window_is_capped_by_cost_constant() -> None:
    """
    送一个大 `top_n` 不能把成本顶穿 —— 窗口收在 `_MAX_CANDIDATES`。

    （`top_n` 来自配置或融合的 top_k，两者的上界都比 50 大。）
    """

    reranker = _StubReranker([0.5] * _MAX_CANDIDATES)
    candidates = _candidates(_MAX_CANDIDATES + 20)

    out = await reranker.rerank(
        query="问题", candidates=candidates, top_n=5000, config=CONFIG,
    )

    assert reranker.scored_count == _MAX_CANDIDATES
    assert len(out) == _MAX_CANDIDATES + 20


async def test_degrade_returns_the_full_list_and_reports_it() -> None:
    """
    打分失败（`_score` 回 None）：原序、**全长**，并且**报 degraded**。

    不报的话调用方会把它当成"精排跑了但没改动结果"，而真实原因是模型
    根本没跑起来 —— 那两件事在宏观统计里必须分得开。
    """

    reranker = _StubReranker(None)
    candidates = _candidates(7)
    degraded: dict[str, str] = {}

    out = await reranker.rerank(
        query="问题", candidates=candidates, top_n=3,
        config=CONFIG, degraded=degraded,
    )

    assert len(out) == 7
    assert [item.content for item in out] == [
        item.content for item in candidates
    ]
    assert degraded, "降级必须说出来"


async def test_passthrough_stub_does_not_truncate_either() -> None:
    """
    占位实现同样不许截断。

    它服务 llm / custom 两个没人实现的类型，配了它的轮次本来就没精排 ——
    一个"什么都没做"的精排偷偷把结果裁短，比它什么都不做更难发现。
    """

    reranker = PassthroughReranker()
    candidates = _candidates(9)
    degraded: dict[str, str] = {}

    out = await reranker.rerank(
        query="问题", candidates=candidates, top_n=4,
        config=RerankerConfig(type=RerankerType.CUSTOM),
        degraded=degraded,
    )

    assert len(out) == 9
    assert degraded, "没实现的精排也要报'这一段没真跑'"


async def test_empty_candidates_short_circuits() -> None:
    """空输入不触发任何调用（连打分都不该进）。"""

    reranker = _StubReranker([0.1])

    out = await reranker.rerank(
        query="问题", candidates=[], top_n=5, config=CONFIG,
    )

    assert out == []
    assert reranker.scored_count == 0
