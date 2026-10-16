"""
逐阶段痕迹的收集器。

它回答两个问题，且**只回答这两个**：

    ① 被盯住的 chunk（目标 / 金标）**在哪一段被丢的**？
    ② 每一段进出的 top-N 是什么、各多少分？

①是评测要的（金标死在召回还是死在裁剪，改法完全不同）；②是判断"别的
chunk 里也有没有正确答案"以及"是阈值砍的还是排序本来就错"的材料。

⚠️ 它是**纯观察者**，这一条是硬约束
----------------------------------
它拿到的是各阶段**已经算完的**列表，只读、不改、返回值一律不用。
`RetrieveProcessor.run` 里它的调用点都在"这一步的结果已经产生之后"，
而不是插在中间。理由很实在：`run()` 是正在工作的主链路，一次"顺手改了
顺序"或"顺手去个重"就会**改变检索结果**，而那不报错 —— 只会让评测的
数字与线上不一致。

配套用例：`tests/services/document/test_retrieve_trace_is_readonly.py`
焊死"带 trace 与不带 trace，results 逐字节相同"。

⚠️ 不记全部候选，只记 top-N ∪ 被盯住的
------------------------------------
一次检索的候选池可能是几百条（每路 top_k 200 × 2 路）。全记下来是每案
几百行、只为回答"哪几条活到了最后"—— 而那个问题 `results` 已经答了。
被丢的那些里最有信息量的是**卡在门槛上**的（第 11–20 名），所以只取
"按死时排名最靠前的 N 个"。加上被盯住的无条件保留，每段 ≤ 2N + |盯住的| 行。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from uuid import UUID

from ..domain.schemas.retrieve import (
    RetrieveStageTrace,
    RetrieveTrace,
    TracedChunkCandidate,
)
from .base import RetrieveCandidate

__all__ = [
    "DEFAULT_TOP_N",
    "INDEX_LAYER_NOTE",
    "RetrievalTraceCollector",
]


#: 每段保留几条候选（存活侧与丢弃侧各这么多）。
DEFAULT_TOP_N = 10

#: 采集的已知边界，写进 `RetrieveTrace.notes`。
#:
#: ⚠️ 放在**响应里**而不是只写在注释里：不写的话，使用者会把「某个
#: chunk 没出现在痕迹的任何一行里」读成「这轮没有东西被过滤掉」，而
#: 真实原因是它根本没被索引层交出来。
INDEX_LAYER_NOTE = (
    "被 `retrieval_config.min_score`（下推到索引层的过滤，ES 的 "
    "min_score / Milvus 的 radius）砍掉的候选**从没见过**，因此不会出现"
    "在任何一行里 —— 它们在痕迹看来与「没召回到」无法区分。"
    "要做到精确归因需要索引后端回传被过滤的候选。"
)


class RetrievalTraceCollector:
    """
    收集一次检索的逐阶段痕迹。

    用法（在 `RetrieveProcessor.run` 与召回那里各调一次）：

        trace = RetrievalTraceCollector(target_chunk_ids=gold_ids)
        trace.observe_recall(route_sets)              # 每路一次
        trace.observe_stage(stage_key="fusioner", before=..., after=...)
        ...
        response.trace = trace.trace
    """

    def __init__(
            self,
            *,
            target_chunk_ids: Iterable[UUID] | None = None,
            top_n: int = DEFAULT_TOP_N,
    ) -> None:

        #: 被盯住的 chunk。判定用 set，顺序用 list（报告里要保持输入顺序）。
        self._target_order: list[UUID] = [
            UUID(str(x)) for x in (target_chunk_ids or [])
        ]
        self._target: set[UUID] = set(self._target_order)

        self._top_n = max(1, top_n)

        self._stages: list[RetrieveStageTrace] = []

        #: 额外的"已知边界"。与 stages 分开：stages 是这一轮**数据**的
        #: 流向，note 是**这次采集本身**的边界或异常（例如查询改写失败
        #: 退回了原文 —— 它不属于任何一段）。
        self._notes: list[str] = []

    # ========================================================
    # 采集
    # ========================================================

    def observe_recall(
            self,
            route_sets: Sequence[Sequence[RetrieveCandidate]],
    ) -> None:
        """
        召回边界：**每路一次**。

        `route_sets` 是"每路一个列表"（与 `RetrieveProcessor` 收到的
        形状一致）。空的那几路不在里面 —— 调用方本来就把空结果滤掉了。

        ⚠️ 这里**只报每一路自己**，不判"金标被召回丢了"。因为召回是
        **多路并行**的：一个 chunk 没出现在 dense 那一路、但出现在
        keyword 那一路，**不是被丢弃**，只是某一支没中。把单路的缺失
        当成丢弃，会让几乎每个 chunk 都显示"死在召回"，宏观统计直接废掉。

        真正"一路都没碰到"的那些由本方法末尾补一条合成记录 —— 而它必须
        在**所有路的输出都拿到之后**才能判。
        """

        seen: set[UUID] = set()

        for candidates in route_sets:

            route = _route_of(candidates)

            self._stages.append(
                self._build(
                    stage_key=f"recall.{route}" if route else "recall",
                    source_key=route,
                    before=candidates,
                    after=candidates,
                    executed=True,
                )
            )

            seen.update(
                candidate.document_chunk_id for candidate in candidates
            )

        # ---- 一路都没撞到的被盯 chunk ----
        #
        # 它不满足任何 `before - after`（两边都没有它），但它恰恰是最该
        # 看到的情况：目标 chunk 连召回的边都没碰到。
        missed = [
            chunk_id for chunk_id in self._target_order
            if chunk_id not in seen
        ]

        if missed:

            self._stages.append(
                RetrieveStageTrace(
                    stage_key="recall",
                    source_key="",
                    executed=True,
                    input_count=0,
                    output_count=0,
                    candidates=[
                        TracedChunkCandidate(
                            chunk_id=chunk_id,
                            rank=1,
                            score=None,
                            source_key=None,
                            selected=False,
                            drop_reason="not_recalled",
                            is_target=True,
                        )
                        for chunk_id in missed
                    ],
                )
            )

    def observe_stage(
            self,
            *,
            stage_key: str,
            before: Sequence[RetrieveCandidate],
            after: Sequence[RetrieveCandidate],
            source_key: str = "",
            reason: str | None = None,
            reasons: dict[UUID, str] | None = None,
            executed: bool = True,
    ) -> None:
        """
        一个通用阶段边界（融合 / 过滤 / 精排 / 裁剪）。

        `before` / `after` 的**顺序即排名**（1-based）—— 候选对象上
        没有 rank 字段，排名就是它在列表里的位置。

        `reason` 是这一段的**统一**丢弃原因（融合/精排/裁剪各只有一种）。
        `reasons` 是**逐个**原因（过滤有四种，只有它自己分得出来）。
        两者都给时 `reasons` 优先。
        """

        self._stages.append(
            self._build(
                stage_key=stage_key,
                source_key=source_key,
                before=before,
                after=after,
                reason=reason,
                reasons=reasons,
                executed=executed,
            )
        )

    def note(self, text: str) -> None:
        """
        补一条已知边界（形制见 `INDEX_LAYER_NOTE`）。

        去重且保序：同一条边界被报两次（例如多 DTD 各失败一次）时，
        notes 变成一长串重复行，读的人会以为是两件不同的事。
        """

        if text and text not in self._notes:
            self._notes.append(text)

    # ========================================================
    # 产出
    # ========================================================

    @property
    def trace(self) -> RetrieveTrace:
        """把收集到的痕迹变成响应里的那个对象。"""

        return RetrieveTrace(
            target_chunk_ids=list(self._target_order),
            stages=list(self._stages),
            notes=[INDEX_LAYER_NOTE, *self._notes],
        )

    # ========================================================
    # internals
    # ========================================================

    def _build(
            self,
            *,
            stage_key: str,
            source_key: str,
            before: Sequence[RetrieveCandidate],
            after: Sequence[RetrieveCandidate],
            reason: str | None = None,
            reasons: dict[UUID, str] | None = None,
            executed: bool = True,
    ) -> RetrieveStageTrace:

        # 进阶段时的排名与分数。
        #
        # ⚠️ 重复的 chunk_id 只取**第一次**出现（也就是最好的名次）：
        # 融合的输入是多路拼起来的，同一块被两路命中就会出现两次。
        # 取最后一次会让"它当时排第几"这个数变小（列表越后越靠后），
        # 而那个数正是用来判断"是不是卡在门槛上"的。
        rank_before: dict[UUID, int] = {}
        score_before: dict[UUID, float] = {}
        by_id: dict[UUID, RetrieveCandidate] = {}

        for index, candidate in enumerate(before):

            chunk_id = candidate.document_chunk_id

            if chunk_id in rank_before:
                continue

            rank_before[chunk_id] = index + 1
            score_before[chunk_id] = candidate.score
            by_id[chunk_id] = candidate

        after_ids = {
            candidate.document_chunk_id for candidate in after
        }

        rows: list[TracedChunkCandidate] = []
        emitted: set[UUID] = set()

        # ---- ① 存活的 top-N（按 after 的顺序 = 排名）----
        for index, candidate in enumerate(after[:self._top_n]):

            chunk_id = candidate.document_chunk_id

            rows.append(
                TracedChunkCandidate(
                    chunk_id=chunk_id,
                    rank=index + 1,
                    score=candidate.score,
                    source_key=candidate.pipeline_key,
                    selected=True,
                    is_target=chunk_id in self._target,
                )
            )

            emitted.add(chunk_id)

        # ---- ② 被丢的 top-N（按**死时**的排名挑最靠前的）----
        #
        # 卡在门槛上的那几条最有信息量：看到 rank=11 / score=0.29 而阈值
        # 是 0.3，一眼看出"是分数砍的"而不是"排序错了"。
        dropped: list[tuple[UUID, RetrieveCandidate]] = []

        for chunk_id, candidate in by_id.items():

            if chunk_id in after_ids:
                continue

            dropped.append((chunk_id, candidate))

        dropped.sort(key=lambda item: rank_before[item[0]])

        for chunk_id, candidate in dropped[:self._top_n]:

            rows.append(
                TracedChunkCandidate(
                    chunk_id=chunk_id,
                    rank=rank_before[chunk_id],
                    score=score_before[chunk_id],
                    source_key=candidate.pipeline_key,
                    selected=False,
                    drop_reason=_reason_for(
                        chunk_id, reason=reason, reasons=reasons,
                    ),
                    is_target=chunk_id in self._target,
                )
            )

            emitted.add(chunk_id)

        # ---- ③ 被盯住的：无条件保留 ----
        #
        # 即使它排在 200 名、或者在 ①② 的 N 之外，也要有一行 ——
        # 「这颗目标 chunk 在哪被丢」全靠它。
        for chunk_id in self._target_order:

            if chunk_id in emitted:
                continue

            if chunk_id not in by_id and chunk_id not in after_ids:
                # 这一段从没见过它（还没被召回，或者早就在上一段没了）。
                # 不在这里补行 —— 它"死在别处"这件事由**别处**那一段回答，
                # 在这儿补一行会让人以为它死在这一段。
                continue

            candidate = by_id.get(chunk_id)

            if candidate is None:
                continue

            selected = chunk_id in after_ids

            rows.append(
                TracedChunkCandidate(
                    chunk_id=chunk_id,
                    rank=(
                        _rank_in(after, chunk_id)
                        if selected
                        else rank_before[chunk_id]
                    ),
                    score=(
                        candidate.score
                        if selected
                        else score_before[chunk_id]
                    ),
                    source_key=candidate.pipeline_key,
                    selected=selected,
                    drop_reason=(
                        None
                        if selected
                        else _reason_for(
                            chunk_id, reason=reason, reasons=reasons,
                        )
                    ),
                    is_target=True,
                )
            )

            emitted.add(chunk_id)

        return RetrieveStageTrace(
            stage_key=stage_key,
            source_key=source_key,
            executed=executed,
            input_count=len(before),
            output_count=len(after),
            candidates=rows,
        )


# ============================================================
# helpers
# ============================================================


def _route_of(candidates: Sequence[RetrieveCandidate]) -> str:
    """
    这一路叫什么。

    取第一个带 `pipeline_key` 的候选 —— 它是
    `DocumentRetrievalDefinition.retrieval_type`（见
    `document_retrieve_service_impl.py` 里构造候选的那一处）。

    ⚠️ 不为此改 `_recall` 的签名（它现在返回"每路一个列表"、丢了路由名）。
    改它要动三个调用点，而候选本身就带着这个信息 —— 少改一处主链路就
    少一分回归风险。
    """

    for candidate in candidates:

        if candidate.pipeline_key:
            return candidate.pipeline_key

    return ""


def _rank_in(
        candidates: Sequence[RetrieveCandidate],
        chunk_id: UUID,
) -> int:
    """在 after 里找它的名次。找不到返回 1（不该发生，兜底不抛）。"""

    for index, candidate in enumerate(candidates):

        if candidate.document_chunk_id == chunk_id:
            return index + 1

    return 1


def _reason_for(
        chunk_id: UUID,
        *,
        reason: str | None,
        reasons: dict[UUID, str] | None,
) -> str | None:
    """
    这一条为什么被丢。

    逐个原因（`reasons`）优先于统一原因（`reason`）：过滤那一段有四种
    原因，只有它自己分得出来；而融合/精排/裁剪各只有一种，按阶段写死
    即可。

    两个都没给时返回 `"unknown"` —— **不返回 None**。None 在这个模型里
    的含义是"没被丢"，而这一条明明被丢了；用它会让"我们不知道原因"
    伪装成"它没死"。
    """

    if reasons is not None:

        found = reasons.get(chunk_id)

        if found:
            return found

    return reason or "unknown"
