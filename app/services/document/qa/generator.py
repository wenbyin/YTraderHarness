"""
QA 数据集生成 —— 把窗口、LLM、定位、求交串起来。

流程
----
    clean 正文
      -> 切窗口（window.split_sections）
      -> 每窗口问一次 LLM（并发 + 限流）
      -> 摘录定位（locate.locate_quotes）
      -> 与 chunk 区间求交（overlap.chunks_overlapping）
      -> 组装成 EvaluationCase 形状的 dict

三条贯穿始终的原则
------------------
**一、单个窗口失败不影响整批。** LLM 会超时、会返回不合格式的东西、
会编造引用。这些都应该只让**那一个窗口**的样本作废并记下原因，而不是
让跑了几个小时的批次整体失败。所以 `_generate_for_window` 把异常吞成
`WindowFailure`，绝不向上抛。

**二、编造的引用必须被丢弃，而不是"尽量用"。** 一条搜不到的摘录意味
着这道题的答案位置无法确定 —— 保留它等于往数据集里塞错误的 ground
truth，而错误的目标标注比没有标注更有害（它会让召回率看起来莫名偏低，
且无法归因）。

**三、日志要说得出"卡在哪"。** 批量任务最怕的是跑了很久、最后只得到
一个空结果。所以每个窗口的耗时、产出条数、失败原因都要留痕。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.common.utils import Section

from .locate import Span, line_of, locate_quotes
from .overlap import ChunkRange, ChunkOverlap, chunks_overlapping

__all__ = [
    "DatasetStats",
    "GeneratedQuestion",
    "WindowFailure",
    "generate_dataset",
]

logger = logging.getLogger(__name__)


class GeneratedQuestion(BaseModel):
    """
    LLM 对**一个窗口**该产出的东西。

    字段名刻意直白：`with_structured_output` 会把它们变成 JSON schema
    发给模型，名字越接近自然语言，模型填得越准。
    """

    question: str = Field(
        description="一个能用窗口内原文回答的问题。",
    )

    answer: str = Field(
        description="该问题的答案，具体到数字或事实。",
    )

    evidence_quotes: list[str] = Field(
        description=(
            "支撑该答案的原文片段，**必须逐字复制自给定正文**。"
            "不得改写、不得拼接、不得补充标点。"
        ),
    )


class WindowQuestions(BaseModel):
    """一次调用的返回。空列表是**合法**结果（窗口里没有可问的事实）。"""

    questions: list[GeneratedQuestion] = Field(
        default_factory=list,
        description="最多 N 条；窗口信息量不足时返回空列表。",
    )


#: 结构化输出用哪种方式。
#:
#: ⚠️ **必须用 json_mode，不能用默认的 function_calling。**
#:
#: langchain 的 `with_structured_output` 默认走**强制工具调用**
#: （`tool_choice='specified'`），而思考型模型拒绝这个组合 ——
#: 实测 `kimi-k2.6`：
#:
#:     400 tool_choice 'specified' is incompatible with thinking enabled
#:
#: 后果是**每个窗口都失败**，而单窗口失败被设计成隔离处理，所以表面上
#: 只是"产出为空"，不会看到这条 400，极难归因。
#:
#: json_mode 走的是 `response_format={"type": "json_object"}`，完全不碰
#: tool_choice，思考型模型照常工作（实测 42.5s 出题成功）。
#:
#: 代价：json_mode **不会**把 schema 自动告诉模型，必须由调用方写进
#: 提示词 —— 见 _json_instruction。
STRUCTURED_METHOD = "json_mode"


#: 生成用的系统提示。
#:
#: 写在模块里而不是配置文件：它与 `GeneratedQuestion` 的字段含义、
#: 与 locate 的"逐字匹配"契约**强耦合**，改一处必须改另一处。放进
#: 配置会让它们有机会各自漂移，而漂移的后果（引用不再逐字 -> 全部被
#: 丢弃 -> 产出为空）很难归因到"有人改了一句提示词"。
_SYSTEM_PROMPT = """\
你在为一份公司公告档案构建**检索**测试集。给定一段正文，产出若干
「问题 + 答案 + 原文依据」。

这些问题的用途是**拿去检索**：系统会把它当成用户的查询词，去索引里
找答案所在的那段正文。所以问题的写法必须像**真人会输入的那样**，
而不是像阅读理解题的题干。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
最重要的要求：问题必须**自包含**、**可检索**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

把问题单独拿出来（不看这段正文、不看这份文档），它必须仍然能被理解，
并且能凭它自己在索引里找到那段正文。

具体地：

1. **必须写出主体名称。** 主体（公司名）在下面的「文档背景」里给 ——
   正文本身通常不含公司名，但**问题里必须有**，否则各家公司的同类
   事件在索引里长得一模一样，检索会捞回一堆别家的。
   ❌ 「公司披露业绩预告的日期是？」
   ❌ 「归属净利润同比上升幅度是多少？」
   ✅ 「东方财富2023年7月15日披露的业绩预告涉及哪个会计期间？」

2. **必须写出时间锚点。** 哪一年、哪一月、哪个报告期。没有时间的问题
   会同时命中这家公司的几十条同类事件，等于没检索。
   ❌ 「归属净利润的下限是多少？」
   ✅ 「东方财富2023年1-6月预计归属净利润的下限是多少？」

3. **禁止一切元指代。** 索引里没有这些词，检索时它们是纯噪音：
   「本段」「该事件」「上述事项」「事件小节」「这段正文」「文中」
   「公司披露的事项类型」……
   ❌ 「事件小节中的日期是什么？」
   ❌ 「公司披露的事项类型是什么？」
   ✅ 「东方财富2023-07-15 披露的事项属于哪一类？」

4. **一个问题问一件事，不要把同一个数字拆成多题。**
   把「范围是多少」拆成「上限是多少」+「下限是多少」，会产出两条几乎
   相同的查询词，在检索里互相争抢排名，对评测没有额外价值。
   ❌ 上限题 + 下限题
   ✅ 「东方财富2023年1-6月预计归属净利润的范围是多少？」

5. **答案必须是具体的事实或数字** —— 日期、股数、金额、比例、机构名。
   不要出「这段话讲了什么」这类概括性问题。

6. **`evidence_quotes` 必须是从给定正文里逐字复制的片段。** 不得改写、
   不得合并多句、不得增删标点或空白。系统会拿它去原文里做精确匹配以
   定位答案位置 —— 不是逐字的引用会被直接丢弃，那道题就白出了。

7. ⚠️ **一条答案的证据可以有多条 `evidence_quotes`，请充分利用。**
   这点很重要：**优先出那种答案需要同时看好几处才能确定的问题**，
   把**每一处**的依据都列进 evidence_quotes。例如：

   「比亚迪 2024 年哪一次股东户数下降幅度最大？」
   —— 这需要横向比较全年的多次披露，依据分散在多条事件里，
   evidence_quotes 应当把**参与比较的那几处**都列上。

   这类问题是**最有价值**的：它们模拟的是真人真的会问的问题，而且
   答案分散在多处 —— 恰恰是检索最容易失败、最需要评测的那一类。
   相比之下，只从一条事件里抠一个数字的问题太容易命中，测不出东西。

8. 如果这段正文太短、或没有可问的具体事实，**返回空列表**。宁可少出
   题，不要硬凑 —— 凑出来的题会成为数据集里的噪音。
"""


@dataclass
class WindowFailure:
    """一个窗口没能贡献样本的原因。留在产物里，便于判断数据集可信度。"""

    start: int
    end: int
    title: str
    reason: str


@dataclass
class DatasetStats:
    """整批的统计。"""

    windows_total: int = 0
    windows_ok: int = 0
    windows_failed: int = 0

    questions_generated: int = 0
    questions_kept: int = 0
    questions_dropped: int = 0

    elapsed_seconds: float = 0.0

    failures: list[WindowFailure] = field(default_factory=list)

    #: 保留样本被丢弃的具体原因（计数），便于看"为什么少了这么多"。
    drop_reasons: dict[str, int] = field(default_factory=dict)

    def to_meta(self) -> dict[str, Any]:
        return {
            "windows_total": self.windows_total,
            "windows_ok": self.windows_ok,
            "windows_failed": self.windows_failed,
            "questions_generated": self.questions_generated,
            "questions_kept": self.questions_kept,
            "questions_dropped": self.questions_dropped,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "drop_reasons": self.drop_reasons,
            "failures": [
                {
                    "section": f.title,
                    "span": [f.start, f.end],
                    "reason": f.reason,
                }
                for f in self.failures
            ],
        }


async def generate_dataset(
        *,
        clean_text: str,
        sections: list[Section],
        chunks: list[ChunkRange],
        model: Any,
        document_version_id: UUID,
        document_context: str = "",
        max_questions_per_window: int = 3,
        concurrency: int = 4,
) -> tuple[list[dict[str, Any]], DatasetStats]:
    """
    跑完整批，返回 `(cases, stats)`。

    `model` 是**已经配好的** chat 客户端（见 app/common/llm.py）；本函数
    不关心它是哪一家。调用方负责选 provider。

    `document_context` 是**文档级背景**（通常取正文最开头的文档头，
    里面有公司名与证券代码）。它必须给 —— 窗口正文里只有事件本身、
    不含公司名，不给的话模型只能写"公司"这种占位词，产出的问题没有
    主体，检索时会把各家公司的同类事件全捞回来。理由详见 _SYSTEM_PROMPT。

    并发用 `Semaphore` 限流而不是 `gather` 全放：云端 LLM 有 QPS 限制，
    一次性打几百个请求会触发限流，重试反而更慢。
    """

    stats = DatasetStats(windows_total=len(sections))

    started = time.monotonic()

    if not sections:
        stats.elapsed_seconds = time.monotonic() - started
        return [], stats

    structured = model.with_structured_output(
        WindowQuestions,
        method=STRUCTURED_METHOD,
    )

    # json_mode 不会把 schema 告诉模型，得我们自己写进提示词。
    # 拼一次复用给所有窗口 —— 它是常量，没必要每窗口重算。
    #
    # 文档背景放进 system 而不是 user：它对**所有窗口都一样**，
    # 而 system 部分在多数 provider 那边是可缓存的前缀。
    system_prompt = _SYSTEM_PROMPT

    if document_context.strip():
        system_prompt += (
            "\n\n## 文档背景\n\n"
            "这份档案的整体信息（问题里的**主体名称**要按它来写）：\n\n"
            f"{document_context.strip()}\n"
        )

    system_prompt += "\n\n" + _json_instruction()

    gate = asyncio.Semaphore(max(1, concurrency))

    async def run_one(section: Section) -> list[dict[str, Any]]:
        async with gate:
            return await _one_window(
                section=section,
                clean_text=clean_text,
                chunks=chunks,
                structured=structured,
                system_prompt=system_prompt,
                document_version_id=document_version_id,
                limit=max_questions_per_window,
                stats=stats,
            )

    # return_exceptions=True 是双保险：_one_window 内部已经吞了异常，
    # 但万一有漏网的（比如 CancelledError 之外的 BaseException），
    # 也不该让整批陪葬。
    results = await asyncio.gather(
        *(run_one(s) for s in sections),
        return_exceptions=True,
    )

    cases: list[dict[str, Any]] = []

    for section, result in zip(sections, results):

        if isinstance(result, BaseException):
            stats.windows_failed += 1
            stats.failures.append(
                WindowFailure(
                    start=section.start,
                    end=section.end,
                    title=section.title,
                    reason=f"{type(result).__name__}: {result}",
                )
            )
            continue

        cases.extend(result)

    stats.elapsed_seconds = time.monotonic() - started

    return cases, stats


def _json_instruction() -> str:
    """
    json_mode 下必须由提示词交代输出结构。

    `with_structured_output(method="json_mode")` 只负责设
    `response_format` 并在返回时按 schema 解析 —— 它**不会**把 schema
    发给模型。不写这段，模型就自由发挥，解析失败率会高得离谱。

    用 `model_json_schema()` 而不是手写一份：手写的那份会与模型定义
    漂移，而漂移的表现是"字段时有时无"这种难查的解析错误。
    """

    schema = json.dumps(
        WindowQuestions.model_json_schema(),
        ensure_ascii=False,
        indent=None,
    )

    return (
        "## 输出格式\n\n"
        "**只输出一个 JSON 对象**，不要任何解释文字，不要 markdown 代码块"
        "（不要 ``` 包裹）。它必须严格符合下面这个 JSON Schema：\n\n"
        f"{schema}\n"
    )


async def _one_window(
        *,
        section: Section,
        clean_text: str,
        chunks: list[ChunkRange],
        structured: Any,
        system_prompt: str,
        document_version_id: UUID,
        limit: int,
        stats: DatasetStats,
) -> list[dict[str, Any]]:
    """
    单个窗口：问 LLM -> 定位 -> 求交。

    **这里不抛异常** —— 任何失败都记进 stats 并返回空列表。
    """

    window = Span(start=section.start, end=section.end)

    user_prompt = (
        f"事件小节：{section.title or '（文档头部，无标题）'}\n\n"
        f"正文：\n\"\"\"\n{section.text}\n\"\"\"\n\n"
        f"请生成最多 {limit} 个问题。"
    )

    started = time.monotonic()

    try:
        parsed: WindowQuestions = await structured.ainvoke(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )

    except Exception as exc:

        stats.windows_failed += 1
        stats.failures.append(
            WindowFailure(
                start=section.start,
                end=section.end,
                title=section.title,
                reason=f"LLM 调用失败：{type(exc).__name__}: {exc}",
            )
        )

        logger.warning(
            "窗口 %r 生成失败：%s: %s",
            section.title,
            type(exc).__name__,
            exc,
        )

        return []

    questions = (parsed.questions or [])[:limit]
    stats.questions_generated += len(questions)

    cases: list[dict[str, Any]] = []

    for item in questions:

        located, failures = locate_quotes(
            item.evidence_quotes, clean_text, window,
        )

        if not located:
            # 一条都没定位到 —— 多半是模型改写了原文而不是逐字引用。
            stats.questions_dropped += 1
            _bump(stats, "引用无法在原文中定位")
            continue

        # ⚠️ **多区间**，不是一条从 min 到 max 的大区间。
        #
        # 一个问题的答案可能分散在正文好几处（跨 section 的问题尤其
        # 如此）。取 min..max 会得到一个**横跨两者之间全部内容**的巨型
        # 区间 —— 它技术上仍落在 chunk 边界内，但毫无意义：中间那些
        # 与答案无关的 chunk 会被一并算成"目标 chunk"，数据集里的
        # ground truth 就此失真，而且不报错。
        spans = _merge_spans([loc.span for loc in located])

        overlaps = _overlaps_of(spans, chunks)

        if not overlaps:
            # 定位到了，但没有 chunk 覆盖任何区间 —— 说明该段落没有
            # 进入索引（切分时被丢弃、或偏移没记录）。这种题在检索
            # 评测里没有意义：无论检索多好都命中不了。
            stats.questions_dropped += 1
            _bump(stats, "区间不落在任何 chunk 内")
            continue

        stats.questions_kept += 1

        cases.append(
            {
                # ---- 对齐 evaluation.yaml 的 EvaluationCase ----
                "user_input": item.question,
                "reference": item.answer,
                "reference_text": section.text,
                "reference_spans": [
                    {
                        "start_char_index": s.start,
                        "end_char_index": s.end,
                    }
                    for s in spans
                ],
                "reference_chunk_ids": [
                    str(o.chunk_id) for o in overlaps
                ],
                # ---- 溯源 ----
                # 行号覆盖**全部**区间：从第一段的开头到最后一段的结尾。
                # 答案分散在好几处时，中间那些行并不都是答案，但行号只
                # 是给人看的溯源信息 —— 精确位置在 reference_spans 里。
                "start_line": line_of(clean_text, spans[0].start),
                "end_line": line_of(clean_text, spans[-1].end),
                "section": section.title,
                "document_version_id": str(document_version_id),
                "evidence_quotes": [loc.quote for loc in located],
                "quote_locations": [
                    {
                        "start_char_index": loc.span.start,
                        "end_char_index": loc.span.end,
                        "exact": loc.exact,
                        "occurrences": loc.occurrences,
                    }
                    for loc in located
                ],
            }
        )

    # ⚠️ ok 计数放在**最后**，不放 LLM 调用之后。
    #
    # 放在前面会让一个窗口同时被记成 ok 和 failed：LLM 成功了（ok+1），
    # 但后面的定位/求交抛出未预期的异常，异常冒到 generate_dataset 那边
    # 又记一次 failed。实测出现过 `windows_ok=1, windows_failed=1` 这种
    # 自相矛盾的统计。
    stats.windows_ok += 1

    logger.info(
        "窗口 %r 耗时 %.1fs，出题 %d 条，保留 %d 条",
        section.title,
        time.monotonic() - started,
        len(questions),
        len(cases),
    )

    return cases


def _bump(stats: DatasetStats, reason: str) -> None:
    stats.drop_reasons[reason] = stats.drop_reasons.get(reason, 0) + 1


def _merge_spans(spans: list[Span]) -> list[Span]:
    """
    把同一道题的多个摘录区间合并成**最少的若干段**。

    只合并**相交或相邻**的：两条摘录挨在一起（比如一句被拆成前后两半）
    合成一段，比报两段更贴近"答案就在这儿"的语义；而相隔很远的两段
    必须保持分开 —— 中间那段正文与答案无关。

    这是 `min..max` 那个做法的替代：那个会把中间全吞进来。
    """

    if not spans:
        return []

    ordered = sorted(spans, key=lambda s: (s.start, s.end))

    merged: list[Span] = [ordered[0]]

    for current in ordered[1:]:

        last = merged[-1]

        # `<=` 而不是 `<`：首尾相接（last.end == current.start）也算
        # 相邻，合并掉。留一段零长度的缝没有意义。
        if current.start <= last.end:
            merged[-1] = Span(
                start=last.start,
                end=max(last.end, current.end),
            )
            continue

        merged.append(current)

    return merged


def _overlaps_of(
        spans: list[Span],
        chunks: list[ChunkRange],
) -> list[ChunkOverlap]:
    """
    所有区间覆盖到的 chunk 的**并集**。

    一道跨 section 的题会命中多个区间，同一个 chunk 可能同时与其中
    两个重叠 —— 所以按 chunk_id 去重。重叠字符数取**最大值**而不是
    累加：累加会得出"这个 chunk 与答案重叠了 300 个字符"这种读数，
    而 chunk 总共可能只有 200 字符，那样会误导人以为它覆盖得更多。
    """

    best: dict[Any, ChunkOverlap] = {}

    for span in spans:

        for overlap in chunks_overlapping(span, chunks):

            existing = best.get(overlap.chunk_id)

            if existing is None or overlap.overlap_chars > existing.overlap_chars:
                best[overlap.chunk_id] = overlap

    # 稳定排序：先按重叠字符数降序，再按 chunk_index —— 与
    # chunks_overlapping 的排序口径一致，便于两侧对照。
    return sorted(
        best.values(),
        key=lambda o: (-o.overlap_chars, o.chunk_index),
    )
