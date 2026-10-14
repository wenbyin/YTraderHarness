"""
阶段 11：CONTEXT_VALIDATOR —— 流水线的最后一道闸（规格 §二十八）。

它只做一件事：**在请求发出去之前，确认它不会失败**。

校验清单（每一条都对应一次真实的、代价很高的失败）
----------------------------------------------
    messages 非空                 空请求
    当前用户消息在里面             "没有问题却在回答"
    工具调用与结果成对             provider 400（而报错指向 provider）
    总 token 不超窗口 - 输出 - 余量  provider 截断 / 400，且**整个窗口**失败

⚠️ 校验失败 = 本轮 Run 失败
------------------------
规格 §二十八 把"上下文 Token 超限且无法裁剪"列为 Fatal，理由很直接：
一份被静默截断的上下文会让模型**自信地答错**，而那种错误没有任何
迹象。宁可让这一轮明确失败。

⚠️ 校验器本身可以关（`context_policy.validate`）
--------------------------------------------
关掉它的唯一正当理由是"我知道这次上下文不需要校验"（比如窄输入的
分类调用）。它是 gate 化的，因此这个决定是**显式**的、进快照的。
"""

from __future__ import annotations

from typing import Any

from app.common.context import StageId

from ..base import ContextProcessor
from ..context import ContextBuildContext
from ..dependencies import dependencies_of
from ..errors import ContextBudgetError, ContextBuildError
from ..messages import (
    is_human_message,
    message_identity,
    repair_tool_pairs,
)
from ..spec import ContextProcessorSpec, StateBinding

__all__ = ["ValidationProcessor"]


class ValidationProcessor(ContextProcessor):
    """在请求发出去之前校验它不会失败。"""

    spec = ContextProcessorSpec(
        name="validation",
        order=99,
        stage=StageId.CONTEXT_VALIDATOR,
        when="context_policy.validate",
        inputs={
            "current_message": StateBinding(
                "current_user_message", required=False
            ),
        },
        description=(
            "校验最终上下文：非空、当前用户消息在、工具配对完整、预算不超。"
        ),
    )

    async def process(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
    ) -> ContextBuildContext:

        deps = dependencies_of(context)

        counter = deps.resolved_token_counter()

        draft = context.draft

        if draft is None:

            raise ContextBuildError(
                "校验阶段看不到 draft（ModelRequest）。"
                "prompt_assembler 阶段没有产出请求 —— 检查它是否被跳过。"
            )

        messages = list(draft.messages)

        if not messages:

            raise ContextBuildError("最终上下文为空（没有任何消息）。")

        # ---- 当前用户消息必须在 ---------------------------------

        current = inputs.get("current_message")

        if current is not None:

            key = message_identity(current)

            if not any(message_identity(item) == key for item in messages):

                raise ContextBuildError(
                    "最终上下文里没有当前用户消息。\n"
                    "这次调用会变成「没有问题却在回答」—— 宁可失败也不要"
                    "让模型凭空答一句。\n"
                    "常见原因：history 处理器把最后一轮裁掉了，或者图 state 的"
                    "messages 与 current_user_message 不是同一批对象。"
                )

        elif not any(is_human_message(item) for item in messages):

            raise ContextBuildError(
                "最终上下文里没有任何用户消息（human）。"
            )

        # ---- 工具配对完整性 ------------------------------------

        repaired, dropped = repair_tool_pairs(messages)

        if dropped:

            # 到这里还有不成对的工具往返，说明上游给的历史本身是断的。
            # 直接发出去 provider 会 400，而报错指向 provider ——
            # 因此在**这里**失败，把指向修正到我们这一侧。
            raise ContextBuildError(
                f"最终上下文里有 {dropped} 条不成对的工具调用 / 结果。\n"
                "provider 会以 400 拒绝这种请求，而报错会指向 provider —— "
                "所以在这里拦下。\n"
                "常见原因：上游给的历史从一次工具往返的中间开始。",
                details={"dropped": dropped},
            )

        # ---- 预算 -------------------------------------------------

        if context.budget is not None and context.policy is not None:

            window = context.budget.context_window
            output = context.policy.max_output_tokens
            reserve = context.policy.reserve_tokens

            total = counter.count_messages(messages)
            total += counter.count_tools(context.tools)

            limit = window - output - reserve

            if total > limit:

                raise ContextBudgetError(
                    f"上下文超限且无法再裁剪：{total} > {limit}"
                    f"（窗口 {window} - 输出 {output} - 余量 {reserve}）。\n"
                    "已经裁到只剩当前这一轮仍然装不下 —— 说明这一次的输入"
                    "本身就超出了模型窗口。两条路：换窗口更大的模型，或"
                    "把输入拆小（比如让检索返回更短的片段、或先对当前"
                    "消息做摘要）。",
                    details={
                        "total": total,
                        "limit": limit,
                        "window": window,
                        "output": output,
                        "reserve": reserve,
                    },
                )

            context.set(
                "validation",
                {
                    "messages": len(messages),
                    "tools": len(context.tools),
                    "input_tokens": total,
                    "limit": limit,
                    "headroom": limit - total,
                },
            )

        return context
