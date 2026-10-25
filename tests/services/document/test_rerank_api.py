"""
第三方精排（`RerankerType.API`）的**离线**用例。

为什么一条都不出网
------------------
这条实现的风险不在"能不能连通"（那由一次真调用验证，见 rerank_api.py
的模块说明），而在**回包映射**：服务端回的是"第几段得了多少分"，映射错
一位就是把别的块的分数安在了它头上 —— 而那看起来只是"精排效果不太
好"，没有任何报错。所以这里把 `_post` 换成一个返回构造回包的假函数，
专测映射、归一化与失败姿态。

（真调用不能进测试：它要钱、要网、还可能因为对端改版而在别人的机器上
红。这与 `test_retrieval_eval_repositories.py` 里"连不上库就跳过"是同
一条取舍。）
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.services.document.domain.schemas import RerankerConfig
from app.services.document.domain.schemas.enum import RerankerType
from app.services.document.retrieve.base import RetrieveCandidate
from app.services.document.retrieve.rerank_api import (
    ApiReranker,
    ApiRerankerSettings,
    _origin,
)

CONFIG = RerankerConfig(type=RerankerType.API, model="qwen3.7-text-rerank")


def _candidate(text: str, score: float) -> RetrieveCandidate:
    return RetrieveCandidate(
        document_chunk_id=uuid4(),
        score=score,
        retrieval_source="keyword",
        content=text,
    )


def _settings(**overrides) -> ApiRerankerSettings:
    payload = {
        "base_url": "https://maas.qianwenaiapi.com/compatible-mode/v1",
        "api_key": "sk-test",
        "protocol": "dashscope",
        "model": "qwen3.7-text-rerank",
    }
    payload.update(overrides)
    return ApiRerankerSettings.model_validate(payload)


def _patch_post(monkeypatch, handler) -> None:
    """把出网那一步换掉；handler 收 (url, payload) 返回回包 dict。"""

    async def _fake_post(self, *, url, headers, payload, settings):
        return handler(url, payload)

    monkeypatch.setattr(ApiReranker, "_post", _fake_post)
    monkeypatch.setattr(
        "app.services.document.retrieve.rerank_api.get_reranker_settings",
        lambda: _settings(),
    )


# ============================================================
# URL / 请求体
# ============================================================


@pytest.mark.parametrize(
    "base,expected",
    [
        (
            "https://maas.qianwenaiapi.com/compatible-mode/v1",
            "https://maas.qianwenaiapi.com",
        ),
        ("https://maas.qianwenaiapi.com/", "https://maas.qianwenaiapi.com"),
        ("https://dashscope.aliyuncs.com", "https://dashscope.aliyuncs.com"),
        ("https://x.example.com/v1", "https://x.example.com"),
    ],
)
def test_origin_strips_the_compatible_mode_suffix(base, expected) -> None:
    """
    原生精排路径挂在**站点根**上，而 llm 段给的 base_url 带
    `/compatible-mode/v1`（那是 chat 的端点）。

    不还原就会拼成 `.../compatible-mode/v1/api/v1/services/...` —— 实测
    返回**不带任何提示的 404**，看起来像"服务没开"。
    """

    assert _origin(base) == expected


async def test_request_targets_the_native_endpoint(monkeypatch) -> None:
    """请求打到原生端点，且带上模型与候选正文（服务端要正文才能打分）。"""

    seen: dict = {}

    def handler(url, payload):
        seen["url"] = url
        seen["payload"] = payload
        return {"output": {"results": [{"index": 0, "relevance_score": 0.9}]}}

    _patch_post(monkeypatch, handler)

    candidates = [_candidate("甲", 0.5), _candidate("乙", 0.4)]

    await ApiReranker().rerank(
        query="问题", candidates=candidates, top_n=2, config=CONFIG,
    )

    assert seen["url"].endswith(
        "/api/v1/services/rerank/text-rerank/text-rerank"
    )
    assert seen["url"].startswith("https://maas.qianwenaiapi.com/api/")

    payload = seen["payload"]

    assert payload["model"] == "qwen3.7-text-rerank"
    assert payload["input"]["query"] == "问题"
    assert payload["input"]["documents"] == ["甲", "乙"]


# ============================================================
# 分数映射
# ============================================================


async def test_scores_map_back_by_index(monkeypatch) -> None:
    """
    `index` 指的是**送进去的顺序**，不是名次 —— 映射错了就是把别人的
    分数安在这条候选头上，而结果看起来只是"精排排得怪"。
    """

    _patch_post(
        monkeypatch,
        lambda url, payload: {
            "output": {
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.5},
                ]
            }
        },
    )

    candidates = [
        _candidate("甲", 1.0),
        _candidate("乙", 0.8),
        _candidate("丙", 0.6),
    ]

    out = await ApiReranker().rerank(
        query="问题", candidates=candidates, top_n=3, config=CONFIG,
    )

    assert [item.content for item in out] == ["丙", "乙", "甲"]
    # 最高分被归一化成 1.0（与本地那条路同一个不变量）。
    assert out[0].score == pytest.approx(1.0)
    assert out[-1].score == pytest.approx(0.1 / 0.9)


async def test_unscored_candidates_are_kept_at_the_tail(monkeypatch) -> None:
    """服务端只回它打过分的那几段；缺的接在后面，**不能丢**。"""

    _patch_post(
        monkeypatch,
        lambda url, payload: {
            "output": {"results": [{"index": 1, "relevance_score": 0.7}]}
        },
    )

    candidates = [_candidate("甲", 1.0), _candidate("乙", 0.8)]

    out = await ApiReranker().rerank(
        query="问题", candidates=candidates, top_n=2, config=CONFIG,
    )

    assert [item.content for item in out] == ["乙", "甲"]


async def test_output_is_not_truncated_to_top_n(monkeypatch) -> None:
    """
    `top_n` 是**送进**精排的条数，不是返回条数。

    这里送 4 段、`top_n=2`：只给前 2 段打分，但返回仍是 4 段 ——
    最终条数归 `cutter.top_k`。旧版会截成 2 条，让精排变成第二次裁剪。
    """

    def handler(url, payload):
        assert len(payload["input"]["documents"]) == 2, "只该送 top_n 段"
        return {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            }
        }

    _patch_post(monkeypatch, handler)

    candidates = [_candidate(t, 0.9 - i * 0.1) for i, t in enumerate("甲乙丙丁")]

    out = await ApiReranker().rerank(
        query="问题", candidates=candidates, top_n=2, config=CONFIG,
    )

    assert len(out) == len(candidates)
    assert [item.content for item in out][:2] == ["乙", "甲"]


# ============================================================
# 失败姿态：降级，但要说出来
# ============================================================


async def test_api_failure_degrades_instead_of_raising(monkeypatch) -> None:
    """
    接口挂掉时**原序返回 + 报 degraded**。

    不报的话，调用方看到的只是一个"已排好序"的列表 —— 与成功一模一样，
    而宏观统计会把它读成"精排跑了但没改善排序"（检索质量问题），
    真实原因是这次调用根本没成功（能力缺口）。
    """

    async def _explode(self, *, url, headers, payload, settings):
        raise RuntimeError("精排服务 502：bad gateway")

    monkeypatch.setattr(ApiReranker, "_post", _explode)
    monkeypatch.setattr(
        "app.services.document.retrieve.rerank_api.get_reranker_settings",
        lambda: _settings(),
    )

    candidates = [_candidate("甲", 1.0), _candidate("乙", 0.8)]
    degraded: dict[str, str] = {}

    out = await ApiReranker().rerank(
        query="问题", candidates=candidates, top_n=2,
        config=CONFIG, degraded=degraded,
    )

    assert [item.content for item in out] == ["甲", "乙"], "保持融合顺序"
    assert len(out) == 2
    assert "reason" in degraded and "502" in degraded["reason"]


async def test_unknown_response_shape_degrades(monkeypatch) -> None:
    """
    认不出回包 = 整批失败（降级），**不能**当成 0 分。

    当 0 分的话，一次接口改版会表现为"精排把最相关的排到了最后" ——
    读起来像检索质量问题。
    """

    _patch_post(monkeypatch, lambda url, payload: {"result": "格式变了"})

    degraded: dict[str, str] = {}

    out = await ApiReranker().rerank(
        query="问题", candidates=[_candidate("甲", 1.0)], top_n=1,
        config=CONFIG, degraded=degraded,
    )

    assert len(out) == 1
    assert "results" in degraded["reason"]


async def test_out_of_range_index_degrades(monkeypatch) -> None:
    """`index` 越界说明对端把候选搞混了 —— 宁可降级也不猜。"""

    _patch_post(
        monkeypatch,
        lambda url, payload: {
            "output": {"results": [{"index": 7, "relevance_score": 0.9}]}
        },
    )

    degraded: dict[str, str] = {}

    await ApiReranker().rerank(
        query="问题", candidates=[_candidate("甲", 1.0)], top_n=1,
        config=CONFIG, degraded=degraded,
    )

    assert degraded, "越界的 index 必须降级"


async def test_missing_model_degrades(monkeypatch) -> None:
    """没配置模型时降级（而不是发一个没有 model 的请求）。"""

    monkeypatch.setattr(
        "app.services.document.retrieve.rerank_api.get_reranker_settings",
        lambda: _settings(model=None),
    )

    degraded: dict[str, str] = {}

    await ApiReranker().rerank(
        query="问题", candidates=[_candidate("甲", 1.0)], top_n=1,
        config=RerankerConfig(type=RerankerType.API), degraded=degraded,
    )

    assert "模型" in degraded["reason"]
