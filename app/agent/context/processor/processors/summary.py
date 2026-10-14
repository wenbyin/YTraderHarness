"""
阶段 3：SUMMARY_LOADER —— 生成并注入对话摘要（规格 §十六）。

一个阶段，两件事，一条判据
------------------------
    1. 按轮次窗口判"该不该生成新摘要"（`turns.plan_window`）
    2. 无论生成与否，把当前有效的摘要注入成一段 system 片段

判据只有一条（见 turns.py 那条不变量）：

    未覆盖的轮数 u > 这一轮的窗口上限 limit(T)   ->  生成新摘要

再不动手，下一次窗口往前一滑，cursor 就落在窗口外了——那几轮会**既不
在摘要、也不在窗口**里。这条判据的两侧都是**轮次数**，与 token 无关：
"会话已经长到该重新摘要了"是一个关于对话结构的事实，不是关于预算的。

节奏（`turns.window_limit`）
-------------------------
    起始 3 轮，每满 10 轮 +1，封顶 5

即对话越长、折得越懒。理由有两面：早期轮次少，短间隔能让摘要跟上现场；
后期摘要本身已承载大部分语义、raw 窗口也够用，拉长间隔省下的是重写
成本（每次重写 = revision+1 → context_hash 变 + 一次模型调用）。

⚠️ 生成**必须**发生在注入之前，这是本阶段与旧实现的实质差别
--------------------------------------------------
旧实现里"注入"在阶段 3、"生成"在阶段 10（CONTEXT_COMPACTOR），于是新
摘要写回了图 state，却**没有**进这一次发给模型的 prompt —— 快照说
revision=5，模型看到的是 revision=4。现在两件事同处一个阶段、按
"先生成后注入"排列，因此不存在这一格错位。

⚠️ 摘要为空**不是**"没有摘要"
---------------------------
模型返回空文本时必须抛错：折进去的那一段历史已经不在窗口里了，而
cursor 一旦推进，历史处理器就会按它把那些消息排除 —— 于是那段对话
彻底消失，且没有任何迹象。这一点与旧实现一致（同一句理由）。

⚠️ 模型从哪来
-----------
`dependencies.resolved_compactor`（缺省按 ModelRegistry + agent_type
现建）—— 本模块**不** import 任何 provider SDK（规格 §十八），
实现住在 `summarization.py`。

⚠️ `summary` 是**可选**输入
------------------------
"这段会话还没有摘要"是完全正常的状态。把它声明成必需会让每一段新
会话的第一次调用直接失败 —— 而那种失败对用户毫无意义。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.common.context import StageId

from ..base import ContextProcessor
from ..context import ContextBuildContext
from ..dependencies import dependencies_of
from ..errors import ContextBuildError
from ..messages import message_identifier
from ..spec import ContextProcessorSpec, StateBinding
from ..turns import TurnWindow, plan_window
from ..values import SummarySnapshot

__all__ = ["SummaryProcessor"]

#: 注入进 system 的分节标题。
LABEL = "历史摘要"


class SummaryProcessor(ContextProcessor):
    """按轮次窗口决定要不要生成新摘要，然后注入当前那一段。"""

    spec = ContextProcessorSpec(
        name="summary",
        order=30,
        stage=StageId.SUMMARY_LOADER,
        when="context_policy.include_summary",
        inputs={
            "summary": StateBinding("summary", required=False),
            # 轮次窗口的**唯一**依据。摘要的判据、折叠切片、历史窗口
            # 宽度全部从它算出来（见 turns.py）。
            "messages": StateBinding("messages"),
        },
        description=(
            "注入对话摘要；按轮次（起始 3 轮、每满 10 轮 +1、封顶 5）"
            "在未覆盖轮数超出窗口上限时重新生成。"
        ),
    )

    async def process(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
    ) -> ContextBuildContext:

        policy = context.policy

        messages = list(inputs.get("messages") or ())

        previous = SummarySnapshot.from_state(inputs.get("summary"))

        summary = previous

        if policy is None:

            # 走到这里说明 resolver 阶段没跑（接线错误）。没有策略就
            # 没有节奏参数，**不生成**——但摘要该注入还是要注入，它是
            # 上几轮的事实，与这一次的策略无关。
            context.set_summary(summary)

            return self._inject(context, summary)

        plan = plan_window(
            messages,
            previous.summarized_until_message_id if previous else None,
            min_turns=policy.min_recent_turns,
            max_turns=policy.max_recent_turns,
        )

        detail: dict[str, Any] = plan.as_metadata()

        if plan.should_summarize and policy.allow_compaction:

            summary, draft_detail = await self._summarize(
                context, plan, previous, messages=messages
            )

            detail["generated"] = True
            detail["revision"] = summary.revision if summary else None
            detail.update(draft_detail)

        else:

            detail["generated"] = False
            detail["reason"] = (
                "below_threshold"
                if not plan.should_summarize
                else "compaction_disabled"
            )

        # 约定的键名就是 spec.name（见 pipeline.py 的 `_shape_delta`）：
        # 它成为这一次执行的 `detail["outcome"]`。这里**只写 metadata**，
        # 不写 overlay —— overlay 上的 `summary` 是给后续阶段按路径绑定的
        # 摘要正文（`_inject` 里的 `set_summary` 负责），两者不是一回事。
        context.metadata["summary"] = detail

        return self._inject(context, summary)

    # --------------------------------------------------------

    async def _summarize(
            self,
            context: ContextBuildContext,
            plan: TurnWindow,
            previous: SummarySnapshot | None,
            *,
            messages: Sequence[Any],
    ) -> tuple[SummarySnapshot | None, dict[str, Any]]:
        """真的调一次摘要模型，并产出**新的完整摘要**。"""

        policy = context.policy

        to_fold = plan.fold_messages()

        if not to_fold:
            return previous, {"folded_turns": 0, "folded_messages": 0}

        deps = dependencies_of(context)

        compactor = deps.resolved_compactor(agent_type=context.agent_type)

        draft = await compactor.compact(
            previous=previous,
            messages=to_fold,
            max_output_tokens=policy.summary_output_tokens,
        )

        if not draft.text.strip():

            raise ContextBuildError(
                "摘要模型返回了空摘要。\n"
                "这不能当成「没有摘要」：被折进去的那段历史已经落在窗口"
                "之外（未覆盖轮数超过了窗口上限），而 cursor 一旦推进，"
                "历史处理器就会按它把那些消息排除 —— 于是那段对话彻底"
                "消失，且没有任何迹象。"
            )

        context.note_usage(draft.usage)

        # 只有**真的生成过**才置这一位：节点靠它决定要不要把摘要回写进图
        # state（见 context.py 的 summary_dirty 与 agent_node 的
        # `_context_state_updates`）。没生成却置位，会让每一轮都把同一份
        # 摘要重写一遍，state 的 channel 看起来一直在变。
        context.summary_dirty = True

        return (
            SummarySnapshot(
                text=draft.text.strip(),
                revision=(previous.revision if previous else 0) + 1,
                summarized_until_message_id=message_identifier(
                    to_fold[-1], _index_of(messages, to_fold[-1])
                ),
            ),
            draft.as_metadata(),
        )

    # --------------------------------------------------------

    def _inject(
            self,
            context: ContextBuildContext,
            summary: SummarySnapshot | None,
    ) -> ContextBuildContext:
        """
        把当前有效的摘要注入成一段 system 片段。

        无论有没有内容都要同步一次工作集上的 summary：后续阶段（history
        的窗口宽度、compaction 的切点、快照、校验）读的都是那一份。
        """

        context.set_summary(summary)

        if summary is None or summary.empty:
            return context

        deps = dependencies_of(context)

        counter = deps.resolved_token_counter()

        context.add_section(
            name="summary",
            label=LABEL,
            text=summary.text.strip(),
            source_ids=(f"summary@{summary.revision}",),
            token_estimate=counter.count_text(summary.text),
        )

        return context


# ============================================================
# internals
# ============================================================


def _index_of(messages: Sequence[Any], target: Any) -> int:
    """
    `target` 在 `messages` 里的位置（用于没有 id 时生成 `#序号` 标识）。

    认**对象身份**而不是相等：同一条消息可能被 langchain 的
    `add_messages` 换过实例，而这里要的是"就是那一条"。
    """

    for index, message in enumerate(messages):

        if message is target:
            return index

    return len(messages) - 1
