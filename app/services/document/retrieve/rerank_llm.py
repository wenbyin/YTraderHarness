"""
用大模型精排（`RerankerType.LLM`）—— 占位实现的真身。

它要解决的问题（与 cross-encoder 同一个）
----------------------------------------
融合（RRF）有**结构性偏差**：只被一路命中的 chunk，无论在那一路里排多高，
都赢不过任何一条被两路命中的。而"这条 case 的金标只被一路召回到"恰恰是
常见情形（实测：7 条最深 case 的 dense 一路全都没召回）。

精排是**另一路信号**：它把 query 与候选正文放一起看，与"被几路命中"无关。
`cross_encoder` 那条走本地小模型；这一条走大模型 —— 代价是慢与贵，换来的
是"能带上领域约定"（见提示词里"这批语料的特点"那一节）。

⚠️ 两个实现是**并列**的，不是替代：选哪个由 `RerankerConfig.type` 决定
（`cross_encoder` / `llm` / `api`）。注册表按类型索引，一个类型一个实现。

为什么单独一个文件
------------------
与 `rerank_api.py` 同一条理由：`rerank.py` 顶上挂着 FlagEmbedding / torch
那条 import 链，而这条实现一行都不需要它。注册表是按模块反射收集的
（`import_package_modules(recursive=False)`），多一个兄弟文件不花钱。

代价的形状：**窗口就是成本**
---------------------------
一次调用要读 N 条候选正文（每条截到 400 字），所以 `top_n` 不只是"排几条"，
它是这次调用的**输入规模**。因此这里的封顶（`_MAX_CANDIDATES`）比
cross-encoder 那条小得多 —— 50 条 × 400 字 ≈ 2 万字，一次调用读这么多
既贵又慢，而且模型的上下文窗口未必接得住。

窗口之外的候选**原序跟在后面**（不打分、不丢）—— 与 cross-encoder 同一套
语义：`top_n` 是"送进精排的条数"，最终条数归 `cutter.top_k`。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.agent.context.processor.messages import (
    message_text as llm_message_text,
)
from app.agent.model import ModelRequest, ModelRole, get_model_registry
from app.agent.prompt import get_prompt_provider

# 截断长度只有**一份**，住在提示词模块里：给多少正文直接决定模型读得懂
# 什么，那是提示词的一部分；而这里要按同一个数做成本预算。抄一份过来
# 迟早会漂，漂的表现是"预算按 400 算、实际送 800"。
from app.agent.prompt.prompts.retrieve_rerank import (
    MAX_CHARS_PER_CANDIDATE,
    render_user_message,
)

from ..domain.schemas import RerankerConfig
from ..domain.schemas.enum import RerankerType
from .base import Reranker, RetrieveCandidate, normalize_scores

__all__ = ["LlmReranker"]

logger = logging.getLogger(__name__)


#: 后台/在线调用用的 agent_type。与 model.yaml 里的 binding 必须一致。
_RERANK_AGENT_TYPE = "document_retrieve_rerank"

#: 送进模型的候选条数上限。
#:
#: ⚠️ 比 `rerank.py` 的 `_MAX_CANDIDATES`（cross-encoder 那条）小一个
#: 量级，因为成本形状不同：本地 cross-encoder 是 0.3s/条的前向，而这条
#: 一次调用要塞 N 条正文进去（每条截到 `MAX_CHARS_PER_CANDIDATE`）。
#:
#: **它同时就是输入规模的闸**：输入上限 = `_MAX_CANDIDATES` ×
#: `MAX_CHARS_PER_CANDIDATE` ≈ 8000 字（≈6K token），所以不需要再来一个
#: "总字数额度"—— 那种闸在这两个数下永远不会触发，留着只会让人以为能靠
#: 它控成本。要控成本就调这两个数之一。
#:
#: 调大它之前先量两件事：单次调用的耗时、以及**分数是否还有区分度**
#: （候选一多，模型容易给一堆同分 —— 同分等于没排）。
_MAX_CANDIDATES = 20

#: 从回包里剔 ``` 围栏（提示词要求只输出一个 JSON 对象）。
_FENCE = re.compile(
    r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$",
    re.DOTALL,
)


class _ScoreItem(BaseModel):
    """回包里的**一条**打分。`index` 必填 —— 见下面的串号说明。"""

    index: int
    relevance: float


class _ScorePayload(BaseModel):
    items: list[_ScoreItem]


def _load_object(text: str) -> dict[str, Any]:
    """模型输出 -> JSON 对象（去 ``` 围栏、顶层必须是对象）。"""

    raw = (text or "").strip()

    if not raw:
        raise ValueError("模型返回了空输出（提示词要求一个 JSON 对象）。")

    fenced = _FENCE.match(raw)

    if fenced is not None:
        raw = fenced.group("body").strip()

    payload = json.loads(raw)

    if not isinstance(payload, dict):
        raise ValueError(
            "模型输出的顶层不是对象（提示词要求一个 JSON 对象）。"
            f"实际类型：{type(payload).__name__}。"
        )

    return payload


class LlmReranker(Reranker):
    """
    调模型给候选打分，按分数重排。

    **失败一律降级**（保持融合顺序原样返回 + 写 `degraded`），不抛异常：
    精排挂掉不该让一次检索失败。绝不"按位置硬配"—— 那会把甲的分数安到
    乙头上，而那种错**看起来只是一次排序不好**（见下面 `_score` 的说明）。

    runtime **懒取**：注册表是无参构造（`reranker_cls()`），不能有
    `__init__` 依赖；顺带的好处是没配这个 role 的环境不会在装配期就炸。
    """

    reranker_types = (RerankerType.LLM,)

    async def rerank(
            self,
            *,
            query: str,
            candidates: list[RetrieveCandidate],
            top_n: int,
            config: RerankerConfig,
            degraded: dict[str, str] | None = None,
    ) -> list[RetrieveCandidate]:

        if not candidates:
            return []

        window = _window(candidates, top_n)

        selected = candidates[:window]

        # 超出部分的候选**没被打分**，但也不能丢 —— 原序接在后面。
        rest = candidates[window:]

        try:
            scores = await self._score(
                query=query, candidates=selected, config=config,
            )

        except Exception as exc:  # noqa: BLE001 —— 契约：失败即降级

            logger.warning(
                "LLM 精排不可用（%s: %s），本轮按融合顺序返回。",
                type(exc).__name__,
                exc,
            )

            if degraded is not None:
                degraded["reason"] = f"{type(exc).__name__}: {exc}"

            return list(candidates)

        scored = list(zip(selected, scores))

        # 稳定排序：同分的保持融合原序（`sort` 是稳定的）。
        scored.sort(key=lambda pair: pair[1], reverse=True)

        ranked = [
            candidate.normalized(score / _MAX_RELEVANCE)
            for candidate, score in scored
        ]

        # 与其它两条实现一致：归一化到 [0,1]、**不截断**（最终条数归 cutter）。
        return normalize_scores(ranked + list(rest))

    async def _score(
            self,
            *,
            query: str,
            candidates: list[RetrieveCandidate],
            config: RerankerConfig,
    ) -> list[float]:
        """
        返回与 `candidates` **等长等序**的原始分（0~10）。

        Raises:
            ValueError / ValidationError: 回包条数不对、编号对不上、或不成形。
                调用方据此降级 —— 宁可"这次没精排"，也不接受可能串了号的
                分数（串号不会报错，只会让分数挂到别的 chunk 上，那是比
                没精排更糟的结果）。
        """

        request = ModelRequest(
            messages=(
                SystemMessage(
                    content=get_prompt_provider().node_prompt(
                        ModelRole.RERANK
                    )
                ),
                HumanMessage(
                    content=render_user_message(
                        query=query,
                        candidates=[
                            (candidate.content or "")[
                                :MAX_CHARS_PER_CANDIDATE
                            ]
                            for candidate in candidates
                        ],
                    )
                ),
            ),
            # ⚠️ 关思考：这条调用的输出形状完全由提示词给定（只回一个
            # JSON），而思考与正文**共用** `max_tokens` —— 不关的话，
            # 20 条候选的输入下很容易"推理写满、JSON 没写完"（实测过
            # 同类症状：`Unterminated string`）。它与模型级那个关思考开关
            # 不是一回事，只对当场这次调用生效。
            disable_thinking=True,
            metadata={
                "purpose": "retrieve.rerank",
                "agent_type": _RERANK_AGENT_TYPE,
                "model_role": ModelRole.RERANK,
                "candidate_count": len(candidates),
                "model": config.model,
            },
        )

        runtime = get_model_registry().get_runtime(
            agent_type=_RERANK_AGENT_TYPE,
            model_role=ModelRole.RERANK,
        )

        response = await runtime.ainvoke(request)

        payload = _ScorePayload.model_validate(
            _load_object(llm_message_text(response.message))
        )

        by_index = {item.index: item.relevance for item in payload.items}

        expected = set(range(1, len(candidates) + 1))

        if len(by_index) != len(payload.items):
            raise ValueError(
                f"回包里有重复编号（共 {len(payload.items)} 条）—— 分不清"
                "哪条分数对应哪条候选。"
            )

        if set(by_index) != expected:
            raise ValueError(
                f"回包的编号与候选对不上：期望 1..{len(candidates)}，"
                f"实际 {sorted(by_index)}。"
            )

        return [float(by_index[index]) for index in range(1, len(candidates) + 1)]


#: 提示词里约定的满分。归一化时除以它 —— 让它与 `normalize_scores`
#: （按最大值缩放）各管一段：这里把 0~10 落进 [0,1]，那里保证头名是 1.0。
_MAX_RELEVANCE = 10.0


def _window(candidates: list[RetrieveCandidate], top_n: int) -> int:
    """
    这一轮实际送几条进模型：`min(top_n, 条数上限, 候选总数)`。

    至少 1 条：`top_n` 传 0 或负数是调用方的错，但"一条都不送"会让
    `_score` 拿到空列表、回包必然对不上号，于是**每一轮都降级** ——
    而那种故障看起来只是"精排从来不生效"。
    """

    return max(1, min(top_n, _MAX_CANDIDATES, len(candidates)))
