"""
阶段 9：TOKEN_BUDGETER —— 用**实测**修正预算并裁剪（规格 §二十一）。

为什么预算要算两次
----------------
    resolver 阶段     只知道窗口与各段的**上限** -> 保守的 history_budget
    budget 阶段       组装之后各段的**实际**大小才已知 -> 实测重算

实测重算之后如果历史超了，就**再跑一次同一个选择器**（换一个更紧的
budget）。复用选择器而不是在这里另写一段裁剪逻辑，是为了让三条不变量
只有一处实现：

    整轮裁剪（工具调用与结果不被拆开）
    最后一轮永不裁（当前用户消息在最后一条 Human 之后）
    已摘要的部分不再进来（summarized_until 已经在 history 里生效）

⚠️ 这里**不**判"裁完还是超"
------------------------
预算处理器负责"尽量装下"，**校验器**负责"装不下就失败"（规格 §二十八：
'上下文 Token 超限且无法裁剪' -> Fatal）。两件事分开，报错就只有一个
来源，而那个来源是流水线的最后一步（信息最全）。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.common.context import StageId

from ..base import ContextProcessor
from ..budget import TokenBudget
from ..context import ContextBuildContext
from ..dependencies import dependencies_of
from ..spec import ContextProcessorSpec

__all__ = ["TokenBudgetProcessor"]


def _summary_cursor(context: ContextBuildContext) -> str | None:
    """当前摘要覆盖到哪一条消息（没有摘要就 None）。"""

    summary = context.summary

    return summary.summarized_until_message_id if summary else None


class TokenBudgetProcessor(ContextProcessor):
    """按实测占用重算预算，并把历史裁到预算内。"""

    spec = ContextProcessorSpec(
        name="token_budget",
        order=90,
        stage=StageId.TOKEN_BUDGETER,
        when="context_policy.enforce_token_budget",
        inputs={},
        description=(
            "按 system / 工具 / 历史的实测占用重算预算，必要时整轮裁剪历史。"
        ),
    )

    async def process(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
    ) -> ContextBuildContext:

        deps = dependencies_of(context)

        counter = deps.resolved_token_counter()

        policy = context.policy
        current = context.budget

        if policy is None or current is None:

            # 走到这里说明 resolver 阶段没跑（接线错误）。不猜、不兜底：
            # 让校验器去报"没有预算"这件事。
            context.warn(
                "token_budget 阶段没有有效预算（resolver 阶段可能没跑），"
                "本次不做预算裁剪。"
            )

            return context

        system_message = context.system_message()

        measured_system = (
            counter.count_messages([system_message])
            if system_message is not None
            else 0
        )

        measured_tools = counter.count_tools(context.tools)

        refined = TokenBudget.allocate(
            context_window=current.context_window,
            policy=policy,
            system_budget=measured_system,
            tool_budget=measured_tools,
            window_source=current.window_source,
            model_id=current.model_id,
        )

        before_chars = sum(
            counter.count_text(getattr(m, "content", "") or "")
            for m in context.history
            if isinstance(getattr(m, "content", None), str)
        )

        history_before = counter.count_messages(context.history)

        if history_before > refined.history_budget and len(context.history) > 1:

            selector = deps.resolved_history_selector()

            trimmed = selector.select(
                context.history,
                policy=policy,
                token_counter=counter,
                history_budget=refined.history_budget,
                current_message_id=None,
                # ⚠️ 摘要覆盖到哪要**照传**，不能传 None
                # ------------------------------------
                # 窗口宽度是 `clamp(未覆盖轮数, min, 上限)`，而"未覆盖"
                # 要减去摘要已覆盖的那几轮。传 None 等于宣称"这一段
                # 没有任何摘要覆盖"，选择器就会按"全程未覆盖"重算窗口 ——
                # 于是这次裁剪对摘要一无所知，压不掉本该压的重叠。
                #
                # `context.history` 里**可能**还留着被摘要覆盖的轮（窗口与
                # 摘要允许重叠，见 history.py），因此 cursor 在这里是有
                # 意义的，不是搬一个用不上的字段。
                summarized_until_message_id=_summary_cursor(context),
            )

            context.add_history(trimmed.messages)

        history_after = counter.count_messages(context.history)

        context.budget = refined
        context.set("context_budget", refined.as_state())

        detail = {
            "window": refined.context_window,
            "history_budget": refined.history_budget,
            "measured_system": measured_system,
            "measured_tools": measured_tools,
            "history_before": history_before,
            "history_after": history_after,
            "trimmed": history_after != history_before,
            "chars_before": before_chars,
        }

        context.set("token_budget", detail)
        context.metadata["token_budget"] = detail

        # draft 已经存在（prompt 阶段在前）—— 用裁剪之后的消息重建它。
        # 用 replace 而不是就地改：ModelRequest 是 frozen 的，
        # 而"改请求"这件事必须留下痕迹（这里就是痕迹）。
        if context.draft is not None:

            context.draft = replace(
                context.draft,
                messages=tuple(context.assembled_messages()),
            )

        return context
