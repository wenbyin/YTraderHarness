"""
跨 chunk 判金标：这道题别的 chunk 是不是也能答。

为什么需要这一步
----------------
题目是**逆向**造出来的（从 chunk 出题，而不是先有问题再找答案）。于是
很自然的情况是：同一份档案里，另一段也把同一个事实写了一遍。

如果只把出题的那一块算作金标，检索返回了另一块时会被判 miss —— 那是
**冤枉**：它确实答得上来。这类假 miss 会让召回率偏低，而且偏低的部分
完全无法从数据上看出来（它看起来就是"没召回到"）。

为什么候选只限**已抽样的池子**
------------------------------
全语料两两比对在成本和噪音上都不成立（一个 DTD 下可能几万块）。所以
候选只取本 bundle 已抽中的 chunk。

⚠️ **因此 gold 集合是下界，不是全集。** 这条限制必须写进 bundle 元数据
—— 不写的话，下游会把"没命中"读成"语料里没有别的答案"，而那个结论
不成立。

为什么判据要严、而且要求**逐字证据**
----------------------------------
"相关"和"能答"是两件事，而模型非常容易把它们混起来（同一个实体、同
一个主题就够了）。宽松判据的后果是金标集合虚胖 → 召回率虚高 → 而且
虚高的部分全是错的，无法察觉。

所以加了一道机械校验：模型给出的 `evidence_excerpt` 必须在那个候选
chunk 的正文里**逐字**找到。找不到说明这条判定没有落到原文上，作废。
这一步把"模型随口说能答"挡住了很大一部分 —— 它逼模型的判定必须指向
一段真实存在的原文。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.context.processor.messages import (
    message_text as llm_message_text,
)
from app.agent.model import ModelRequest, ModelRole
from app.agent.prompt import get_prompt_provider
from app.agent.prompt.prompts.eval_cross_chunk_judge import (
    render_user_message as render_judge_request,
)

from ..qa.locate import Span, locate_quotes
from .output import ModelOutputError, parse_adjudication_verdict

__all__ = [
    "ADJUDICATE_AGENT_TYPE",
    "AdjudicationOutcome",
    "GoldCandidate",
    "adjudicate_question",
    "build_model_request",
]

#: 发起裁定调用的 agent_type。必须与 model.yaml 的 bindings 逐字一致。
ADJUDICATE_AGENT_TYPE = "retrieval_eval_adjudicate"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GoldCandidate:
    """一个被判定为「也能回答这道题」的 chunk。"""

    chunk_id: Any
    evidence_excerpt: str
    #: 裁定过程的原始记录（模型返回了什么），写进 gold_chunk.judgment。
    judgment: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdjudicationOutcome:
    """
    一次裁定的结果。

    `verdicts` 里每条都能追溯到原文证据；`rejected` 记下被驳回的判定
    与原因 —— 驳回率本身是个有用的信号（它高说明模型的判据在漂）。
    """

    accepted: list[GoldCandidate] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        return self.error_message is None


def build_model_request(
        *,
        question: str,
        answer_original: str,
        candidates: list[dict[str, Any]],
) -> ModelRequest:
    """
    一次裁定请求。单独成方法是为了可测（与出题那份同一理由）。

    分工与出题一致：系统提示词按 model_role 查，用户消息（问题 + 已知
    答案 + 候选正文）交给提示词模块的 `render_user_message`。
    """

    by_id = {
        str(chunk.get("chunk_id")): chunk for chunk in candidates
    }

    human = render_judge_request(
        question=question,
        answer_original=answer_original,
        candidates=candidates,
    )

    return ModelRequest(
        messages=(
            SystemMessage(
                content=get_prompt_provider().node_prompt(
                    ModelRole.EVAL_CROSS_CHUNK_JUDGE
                )
            ),
            HumanMessage(content=human),
        ),
        metadata={
            "purpose": "retrieval_eval.adjudicate",
            "agent_type": ADJUDICATE_AGENT_TYPE,
            "model_role": ModelRole.EVAL_CROSS_CHUNK_JUDGE,
            "candidate_count": len(by_id),
        },
    )


async def adjudicate_question(
        *,
        question: str,
        answer_original: str,
        candidates: list[dict[str, Any]],
        runtime: Any,
) -> AdjudicationOutcome:
    """
    判定 `candidates` 里哪些 chunk 也能独立回答这道题。

    `candidates` 的元素至少要有 `chunk_id` 与 `content`，且**必须已经
    排除出题用的那个源 chunk**（源 chunk 是金标这件事不需要判 —— 它
    是构造出来的事实，不是模型判断）。

    调用方负责把候选池限制在"本 bundle 已抽样的 chunk"内，见模块说明。

    `runtime` 是 Model Runtime。与 `generate_for_window` 一样**不抛异常**：
    裁定失败只该让这条题的金标集合退化成"只有源 chunk"（那是安全的默认
    值），不该让整批失败。
    """

    if not candidates:

        return AdjudicationOutcome()

    by_id = {
        str(chunk.get("chunk_id")): chunk for chunk in candidates
    }

    try:

        response = await runtime.ainvoke(
            build_model_request(
                question=question,
                answer_original=answer_original,
                candidates=candidates,
            )
        )

        verdict = parse_adjudication_verdict(
            llm_message_text(response.message)
        )

    except ModelOutputError as exc:

        logger.warning("金标裁定输出无法解析：%s", exc)

        return AdjudicationOutcome(
            error_message=f"parse_error: {exc}",
        )

    except Exception as exc:

        logger.warning(
            "金标裁定失败：%s: %s", type(exc).__name__, exc,
        )

        return AdjudicationOutcome(
            error_message=f"{type(exc).__name__}: {exc}",
        )

    accepted: list[GoldCandidate] = []
    rejected: list[str] = []

    for raw_id in verdict.answerable_chunk_ids or []:

        candidate = by_id.get(str(raw_id))

        if candidate is None:
            rejected.append(
                f"裁定返回了候选之外的 chunk_id：{raw_id!r}"
            )
            continue

        # 候选正文只取一次：下面的逐字校验与证据切片都要用它，
        # 而 `str(chunk.get(...) or "")` 每次都会重算一遍。
        content = str(candidate.get("content") or "")

        excerpt = (
            verdict.evidence_excerpts or {}
        ).get(str(raw_id), "") or ""

        if not excerpt.strip():
            rejected.append(
                f"{raw_id}：判为能回答但没给 evidence_excerpt。"
                "没有原文证据的判定不予采信。"
            )
            continue

        # ⚠️ 机械校验：证据必须是这个候选正文里的**逐字**片段。
        #
        # 这是唯一能自动验证"模型的判定落到了原文上"的手段。没有它，
        # 模型只要说一句"能回答"，那个 chunk 就进金标集合 —— 而它可能
        # 根本没看过正文，或者看串了段。
        located, failures = locate_quotes(
            [excerpt],
            content,
            Span(start=0, end=len(content)),
        )

        if not located:
            rejected.append(
                f"{raw_id}：evidence_excerpt 在该候选正文里找不到"
                f"（{failures[0] if failures else excerpt[:40]!r}）——"
                "判定未落到原文，作废。"
            )
            continue

        accepted.append(
            GoldCandidate(
                chunk_id=raw_id,
                # 同样存原文切出来的那一段，而不是模型给的那串 ——
                # 理由见 generator.validate_items 里同一处。
                evidence_excerpt=(
                    content[located[0].span.start:located[0].span.end]
                ),
                judgment={
                    "question": question,
                    "answer_original": answer_original,
                    "evidence_exact": located[0].exact,
                },
            )
        )

    return AdjudicationOutcome(
        accepted=accepted,
        rejected=rejected,
    )
