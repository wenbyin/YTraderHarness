"""
单个窗口：调 LLM -> 解析 -> **服务端校验** -> 产出 case。

三条贯穿的原则（与 qa/generator.py 同源，理由也一样）
--------------------------------------------------
**一、单个窗口失败不影响整批。** LLM 会超时、会限流、会返回不合格式
的东西。这些只该让**那一个窗口**的产出作废并记下原因，而不是让跑了
几小时的批次整体失败。所以 `generate_for_window` 把异常吞成
`WindowOutcome`，绝不向上抛。

**二、编造的引用必须被丢弃，而不是「尽量用」。** 一条在源正文里找不到
的 `answer_original` 意味着这道题的答案位置无法确定 —— 保留它等于往
数据集里塞错误的 ground truth，而错误的目标标注比没有标注更有害：
它让召回率莫名偏低，且无法归因。

**三、JSON 合法不等于内容合法。** 模型返回结构完全正确的 JSON，里面
照样可以有编造的 chunk_id、根本不是原文的「答案」、以及换个说法重复
两遍的同一道题。**绝不因为 JSON 解析成功就信任内容** —— 这是这个模块
存在的主要理由。

校验能拦住什么、拦不住什么
--------------------------
    拦得住（机械可判）    未知 chunk_id、答案非逐字子串、空问题、
                          同一 chunk 内重复的问题、超出题数上限
    拦不住（只能靠提示词）跨 chunk 推断、外部知识、计算、逻辑跳跃

第二类是**已知的、写在明面上的**局限。它不会报错，只会让数据集质量
下降 —— 所以提示词（prompt.py）才是边界的第一道防线，而不是这里。
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.context.processor.messages import (
    message_text as llm_message_text,
)
from app.agent.model import ModelRequest, ModelRole
from app.agent.prompt import get_prompt_provider
from app.agent.prompt.prompts.eval_qa_generation import (
    render_user_message as render_qa_request,
)

from ..qa.locate import Span, locate_quotes
from .output import ModelOutputError, parse_window_questions

__all__ = [
    "ERROR_CODES",
    "EVAL_AGENT_TYPE",
    "GeneratedCase",
    "RejectedItem",
    "WindowOutcome",
    "build_model_request",
    "generate_for_window",
    "normalize_question",
    "validate_items",
]

logger = logging.getLogger(__name__)


#: 窗口失败/产出异常的分类码。
#:
#: 受控词表而不是自由文本：调用方要**按码**判断"这个窗口值不值得重试"
#: （限流值得、窗口超预算不值得），而自由文本做不到这一点。
ERROR_CODES = (
    #: 调模型本身失败（超时、限流、连接、400）。
    "llm_error",
    #: 模型返回了，但结构化解析失败（不是合法 JSON / 字段对不上）。
    "parse_error",
    #: 模型返回了，但一条合法 case 都没有。
    "no_valid_case",
    #: 窗口里一个 chunk 都没有（切分阶段的边界情况）。
    "empty_window",
)


@dataclass(frozen=True)
class GeneratedCase:
    """一条通过校验的 QA，已经可以直接写进 retrieval_eval_case。"""

    chunk_id: Any
    question: str
    answer_original: str
    question_order: int
    question_sha256: str
    answer_start_offset: int | None = None
    answer_end_offset: int | None = None
    #: 逐字精确匹配为 True；白空格差异导致的折叠匹配为 False。
    exact: bool = True
    #: 该答案片段在源 chunk 内出现了几次。>1 表示它在块内不唯一 ——
    #: 不影响判定（金标是整块），但记下来便于复核。
    occurrences: int = 1


@dataclass(frozen=True)
class RejectedItem:
    """一条被丢弃的模型产出，以及为什么。"""

    chunk_id: str | None
    question: str
    reason: str


@dataclass
class WindowOutcome:
    """
    一个窗口的结果。

    **成功与失败都存在这个对象里**，而不是用异常区分 —— 单窗口失败
    是预期内的日常事件（限流、偶发格式错误），把它做成异常会让调用方
    的循环里到处是 try。
    """

    cases: list[GeneratedCase] = field(default_factory=list)
    rejected: list[RejectedItem] = field(default_factory=list)
    insufficient_fact_chunk_ids: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """
        这个窗口算不算成功。

        注意判据是 `error_code is None`，而不是"产出了 case"。一个
        真正不含可问事实的窗口（模型正确地返回了空列表并标注了
        insufficient_fact）**不算失败** —— 把它记成失败会让 bundle
        永远进不了 ready，而那条路是提示词明确允许的。
        """

        return self.error_code is None


def normalize_question(question: str) -> str:
    """
    问题文本的归一化形式，用于查重。

    折叠全部白空格并去掉首尾 —— 只差一个空格的同一道题就是同一道题。
    不折叠的话，模型换个空格就能绕过去重，而两道几乎相同的查询词在
    检索里会互相争抢排名，对评测没有额外价值。
    """

    return " ".join(question.split())


def validate_items(
        items: list[Any],
        window_chunks: list[dict[str, Any]],
        *,
        questions_per_chunk: int,
) -> tuple[list[GeneratedCase], list[RejectedItem]]:
    """
    校验一个窗口的模型产出。返回 `(合法 case, 被丢弃的条目)`。

    纯函数，不碰网络与数据库 —— 校验规则是这个功能里最需要被反复验证
    的部分，它必须能脱离 LLM 单独测。
    """

    by_id = {
        str(chunk.get("chunk_id")): chunk
        for chunk in window_chunks
    }

    cases: list[GeneratedCase] = []
    rejected: list[RejectedItem] = []

    #: str(chunk_id) -> 该 chunk 已产出的题数
    per_chunk_count: dict[str, int] = {}

    #: str(chunk_id) -> 该 chunk 已见过的问题指纹
    seen_questions: dict[str, set[str]] = {}

    for item in items:

        raw_chunk_id = str(getattr(item, "chunk_id", "") or "").strip()
        question = (getattr(item, "question", "") or "").strip()
        answer = getattr(item, "answer_original", "") or ""

        # ---- 1. chunk_id 必须是本窗口里给过的 ----
        #
        # 模型编造 id 是常见失败，而且**必须**在这里拦住：不拦的话，
        # 这道题的源 chunk 就不存在，金标集合是空的，而空金标会让
        # 每一条检索都判 miss —— 一个凭空多出来的 0% 召回。
        chunk = by_id.get(raw_chunk_id)

        if chunk is None:
            rejected.append(
                RejectedItem(
                    chunk_id=raw_chunk_id or None,
                    question=question,
                    reason=(
                        f"chunk_id 不在本窗口内：{raw_chunk_id!r}。"
                        "模型只能使用输入里给出的 id。"
                    ),
                )
            )
            continue

        # ---- 2. 问题与答案不能是空的 ----
        if not question:
            rejected.append(
                RejectedItem(
                    chunk_id=raw_chunk_id,
                    question="",
                    reason="问题为空。",
                )
            )
            continue

        if not answer.strip():
            rejected.append(
                RejectedItem(
                    chunk_id=raw_chunk_id,
                    question=question,
                    reason="answer_original 为空。",
                )
            )
            continue

        # ---- 3. 每块题数上限 ----
        #
        # 上限由**服务端**强制，不靠模型自觉：模型偶尔会一口气给十几条
        # （尤其内容密集的表格段）。放行会让 token 与耗时不受控，也会
        # 让案例数远超配置。
        if per_chunk_count.get(raw_chunk_id, 0) >= questions_per_chunk:
            rejected.append(
                RejectedItem(
                    chunk_id=raw_chunk_id,
                    question=question,
                    reason=(
                        f"超出每块题数上限（{questions_per_chunk}），"
                        "本块已经取满。"
                    ),
                )
            )
            continue

        # ---- 4. 同一块内不能重复问 ----
        fingerprint = hashlib.sha256(
            normalize_question(question).encode("utf-8")
        ).hexdigest()

        bucket = seen_questions.setdefault(raw_chunk_id, set())

        if fingerprint in bucket:
            rejected.append(
                RejectedItem(
                    chunk_id=raw_chunk_id,
                    question=question,
                    reason="同一源 chunk 内的问题重复（归一化后相同）。",
                )
            )
            continue

        # ---- 5. 答案必须是源正文里的**逐字**片段 ----
        #
        # 复用 qa/locate.py 的定位逻辑：精确匹配失败时会退化成白空格
        # 折叠匹配（LLM 的换行/多空格常与原文有出入）。退化的匹配仍然
        # 可信 —— 定位到的是**原文真实的字符区间**。
        content = str(chunk.get("content") or "")

        located, failures = locate_quotes(
            [answer],
            content,
            Span(start=0, end=len(content)),
        )

        if not located:
            rejected.append(
                RejectedItem(
                    chunk_id=raw_chunk_id,
                    question=question,
                    reason=(
                        "answer_original 在该 chunk 正文里找不到："
                        f"{failures[0] if failures else answer[:40]!r}。"
                        "模型改写/拼接了原文。"
                    ),
                )
            )
            continue

        hit = located[0]

        # ⚠️ **存原文切出来的那一段，而不是模型给的那串。**
        #
        # 白空格折叠匹配时两者并不逐字相同（模型把换行写成了空格之类）。
        # 直接存模型那串，就等于在"逐字片段"这一列里放了一个非逐字的
        # 字符串 —— 而下游（导出、人工复核）相信这一列。用 content 切片
        # 则天然满足"逐字 + 与 offset 一致"。
        answer_original = content[hit.span.start:hit.span.end]

        cases.append(
            GeneratedCase(
                chunk_id=chunk.get("chunk_id"),
                question=question,
                answer_original=answer_original,
                question_order=per_chunk_count.get(raw_chunk_id, 0),
                question_sha256=fingerprint,
                answer_start_offset=hit.span.start,
                answer_end_offset=hit.span.end,
                exact=hit.exact,
                occurrences=hit.occurrences,
            )
        )

        per_chunk_count[raw_chunk_id] = (
            per_chunk_count.get(raw_chunk_id, 0) + 1
        )
        bucket.add(fingerprint)

    return cases, rejected


#: 发起出题调用的 agent_type。
#:
#: ⚠️ 它必须与 `app/config/model.yaml` 的 bindings 里那一条**逐字一致** ——
#: Resolver 只做等值匹配，对不上就是"没有这个绑定"的报错。这里的值取
#: 服务/产物名（`retrieval_eval_generate`），与 memory 那几条同一口径。
EVAL_AGENT_TYPE = "retrieval_eval_generate"


def _truncation_evidence(response: Any) -> str | None:
    """
    这次输出是不是被**输出额度**截断的？是就给一句证据，否则 None。

    为什么要单独判：被截断的 JSON 与"模型真的写坏了 JSON"在 `json.loads`
    眼里完全一样（`Unterminated string starting at …`），而两者的处置
    **相反**：

        截断    一次要的东西太多，或额度被推理吃掉
                → 抬 max_tokens / 减少一次要的 chunk 数；
                  重试是同一条路，大概率在同一个地方再截一次（白花钱）
        写坏    提示词或模型的输出纪律问题 → 重试有意义

    此前两类在日志里都写成"模型输出不是合法 JSON"，指向的是**解析器** ——
    照着那个方向查会去改提示词，而真正要动的是额度或批量大小。

    判定依据是 provider 的 `finish_reason == "length"`（**权威信号**，
    不是猜的）；拿不到它时回 None —— 宁可退回"无法解析"，也不要把
    "看起来像被截断"当成截断去误导下一个读日志的人。
    """

    message = getattr(response, "message", None)

    metadata = dict(getattr(message, "response_metadata", None) or {})

    if metadata.get("finish_reason") != "length":
        return None

    usage = dict(metadata.get("token_usage") or {})

    parts = ["finish_reason=length"]

    if usage.get("completion_tokens") is not None:
        parts.append(f"输出 {usage['completion_tokens']} tokens")

    reasoning = (
        (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        )
    )

    if reasoning is not None:
        parts.append(f"其中推理 {reasoning}")

    return "，".join(parts)


def build_model_request(
        *,
        window_chunks: list[dict[str, Any]],
        questions_per_chunk: int,
        document_context: str = "",
) -> ModelRequest:
    """
    一个窗口 -> 一次模型请求。

    单独成方法是为了**可测**：断言"喂进去的长什么样"不必真的调模型
    （与 memory 的 `_build_extract_request` 同一条理由）。

    职责分工与 memory 那边一致：
        系统提示词   按 model_role 查（配置驱动，所以走 prompt provider）
        用户消息     交给提示词模块的 `render_user_message`
                      —— 那一半的**排版**是提示词的一部分

    `document_context` 是**文档级背景**（通常取正文最开头的文档头，里面
    有公司名与证券代码）。它拼在系统提示词的**末尾**而不是用户消息里：
    它对同一份档案的所有窗口都一样，放在系统段更贴近"长期设定"，而且
    系统段在多数 provider 那边是可缓存的前缀。

    ⚠️ 它**必须给**（有的话）：事件正文里通常只有事件本身、不含公司名，
    不给的话模型只能写「公司」这种占位词 —— 产出的问题没有主体，检索时
    会把各家公司的同类事件全捞回来。
    """

    system_prompt = get_prompt_provider().node_prompt(
        ModelRole.EVAL_QA_GENERATION
    )

    if document_context.strip():
        system_prompt += (
            "\n\n## 文档背景\n\n"
            "这份档案的整体信息（问题里的**主体名称**要按它来写）：\n\n"
            f"{document_context.strip()}\n"
        )

    human = render_qa_request(window_chunks, questions_per_chunk)

    return ModelRequest(
        messages=(
            SystemMessage(content=system_prompt),
            HumanMessage(content=human),
        ),
        metadata={
            "purpose": "retrieval_eval.generate",
            "agent_type": EVAL_AGENT_TYPE,
            "model_role": ModelRole.EVAL_QA_GENERATION,
            "chunk_count": len(window_chunks),
            "questions_per_chunk": questions_per_chunk,
        },
    )


async def generate_for_window(
        *,
        window_chunks: list[dict[str, Any]],
        runtime: Any,
        questions_per_chunk: int,
        document_context: str = "",
) -> WindowOutcome:
    """
    跑一个窗口。**不抛异常** —— 任何失败都变成 `WindowOutcome`。

    `runtime` 是 Model Runtime（`ModelRegistry.get_runtime(...)` 拿到的
    那个）。本函数不关心模型是谁、怎么连的，也不自己做重试 —— timeout /
    retry / fallback 全在 Runtime 里（一次 `ainvoke` 内部可能试了几家，
    对外只是一次调用）。

    ⚠️ **不设 `response_format`**，与 memory 那几处一致
    -------------------------------------------------
    JSON 形状写在提示词里，回应由 `parse_window_questions` 用 pydantic
    校验。理由是这条路径在仓库里是被验证过的，而 `response_format` 传
    pydantic 模型时各家 provider 的行为不一致（见 adapters.py 的说明）。
    """

    started = time.monotonic()

    if not window_chunks:

        return WindowOutcome(
            error_code="empty_window",
            error_message="窗口内没有 chunk。",
            elapsed_seconds=time.monotonic() - started,
        )

    request = build_model_request(
        window_chunks=window_chunks,
        questions_per_chunk=questions_per_chunk,
        document_context=document_context,
    )

    try:

        response = await runtime.ainvoke(request)

        parsed = parse_window_questions(
            llm_message_text(response.message)
        )

    except ModelOutputError as exc:

        # 先判"是不是被输出额度截断的" —— 它与"真写坏了 JSON"在
        # `json.loads` 眼里一模一样（都是 Unterminated string），但处置相反。
        truncated = _truncation_evidence(response)

        if truncated is not None:

            logger.warning(
                "窗口输出**被截断**（%s）。这不是提示词或解析的问题："
                "一次要 %d 个 chunk × %d 条题，每条答案都是长引文，"
                "输出额度被推理吃掉之后就写不完了。处置方向是"
                "①抬该模型的 max_tokens ②减少一次要的 chunk 数 ——"
                "重试大概率再截一次，只是白花钱。原文：%s",
                truncated,
                len(window_chunks),
                questions_per_chunk,
                exc,
            )

            return WindowOutcome(
                error_code="truncated",
                error_message=f"输出被截断（{truncated}）：{exc}",
                elapsed_seconds=time.monotonic() - started,
            )

        # 模型答了、但形状不对。与 llm_error 分开：这一类**值得重试**
        # （这一次没按格式答不代表下一次也是），而 llm_error 里混着
        # 400/401 那种重试也没用的。
        logger.warning("窗口输出无法解析：%s", exc)

        return WindowOutcome(
            error_code="parse_error",
            error_message=str(exc),
            elapsed_seconds=time.monotonic() - started,
        )

    except Exception as exc:

        logger.warning(
            "窗口生成失败：%s: %s", type(exc).__name__, exc,
        )

        return WindowOutcome(
            error_code="llm_error",
            error_message=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.monotonic() - started,
        )

    raw = parsed.model_dump()

    cases, rejected = validate_items(
        parsed.items or [],
        window_chunks,
        questions_per_chunk=questions_per_chunk,
    )

    elapsed = time.monotonic() - started

    if not cases:

        # 一条都没留下。注意这与"窗口里本来就没事实"不同 —— 后者模型
        # 会返回空 items **且**标注 insufficient_fact；而这里指的是
        # 模型给了条目、但全部没通过校验。那是"模型在编"，值得重试。
        #
        # （真正空窗口的情形由调用方看 insufficient_fact_chunk_ids 判断，
        # 不在这里当失败。）
        if parsed.items:

            return WindowOutcome(
                rejected=rejected,
                insufficient_fact_chunk_ids=(
                    parsed.insufficient_fact_chunk_ids or []
                ),
                raw_response=raw,
                error_code="no_valid_case",
                error_message=(
                    f"模型给了 {len(parsed.items)} 条，全部未通过校验；"
                    f"首条原因：{rejected[0].reason if rejected else '（无）'}"
                ),
                elapsed_seconds=elapsed,
            )

        return WindowOutcome(
            insufficient_fact_chunk_ids=(
                parsed.insufficient_fact_chunk_ids or []
            ),
            raw_response=raw,
            elapsed_seconds=elapsed,
        )

    logger.info(
        "窗口产出 %d 条（丢弃 %d 条），耗时 %.1fs",
        len(cases),
        len(rejected),
        elapsed,
    )

    return WindowOutcome(
        cases=cases,
        rejected=rejected,
        insufficient_fact_chunk_ids=(
            parsed.insufficient_fact_chunk_ids or []
        ),
        raw_response=raw,
        elapsed_seconds=elapsed,
    )
