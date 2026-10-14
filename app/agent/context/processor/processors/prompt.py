"""
阶段 8：PROMPT_ASSEMBLER —— 流水线的**出口**（规格 §十九）。

它把前面各阶段累积的东西组装成 **Model 模块的 `ModelRequest`**：

    system  （策略里的 system_prompt + 画像 / 记忆 / 资料 / 摘要分节）
    + 历史（history 处理器选出来的）
    + 必须在的消息（当前用户消息，若不在历史里）
    → ModelRequest(messages=…, tools=…, metadata={…})

⚠️ 这个请求直接交给 Model Runtime
-------------------------------
中间**没有**第二套请求类型、也没有转换层（规格 §二十五：所有模型调用
统一经过 ModelRegistry）。节点拿到它只做一件事：

    runtime = registry.get_runtime(...)
    response = await runtime.ainvoke(model_request)

⚠️ `when=None`（无条件）
----------------------
它是 required 阶段：少它就没有产出，既有 `BaseContextBuilder.build()`
会以"没有产出 ModelRequest"告终。因此它不许被 gate（见 pipeline 的
双向校验）。

draft 去哪了
-----------
工作集里的 `draft` 由 Builder 在每个阶段之后同步进既有的
`ContextBuildState.draft`（那是既有契约的产出槽）。装配器只负责**产出**，
不负责往 state 里写 —— 那是桥（Builder）的活。
"""

from __future__ import annotations

from typing import Any

from app.agent.model import ModelRequest
from app.common.context import StageId

from ..base import ContextProcessor
from ..context import ContextBuildContext
from ..errors import ContextBuildError
from ..spec import ContextProcessorSpec, StateBinding

__all__ = ["PromptAssemblerProcessor"]


class PromptAssemblerProcessor(ContextProcessor):
    """把累积的上下文组装成 ModelRequest。"""

    spec = ContextProcessorSpec(
        name="prompt_assembler",
        order=80,
        stage=StageId.PROMPT_ASSEMBLER,
        when=None,  # 出口：无条件（见模块 docstring）
        inputs={
            "response_format": StateBinding(
                "context_policy.response_format", required=False
            ),
            "model_metadata": StateBinding(
                "context_policy.model_metadata", required=False
            ),
        },
        description="组装最终 ModelRequest（system + 历史 + 必须在的消息 + 工具）。",
    )

    async def process(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
    ) -> ContextBuildContext:

        messages = context.assembled_messages()

        if not messages:

            raise ContextBuildError(
                "组装之后的上下文是空的（既没有 system 片段，也没有历史"
                "消息）。\n"
                "这不是「可以继续」的状态：模型会收到一个空请求。"
                "常见原因：history 处理器没有选出任何消息 —— 它只从图 "
                "state 的 messages 里选（不再有别的来源，见 "
                "processors/history.py），所以 messages 为空就是空的。"
                "调用方要么把消息放进 state，要么把 include_history "
                "打开让当前用户消息作为「必须在的消息」补进来。"
            )

        metadata: dict[str, Any] = {
            "context_builder": context.builder_type,
            "context_phase": context.phase.value,
            "agent_type": context.agent_type,
            "model_role": context.model_role,
        }

        extra = inputs.get("model_metadata")

        if isinstance(extra, dict):
            # 调用方要带的自定义元信息（追踪标签之类）。放在这里而不是
            # 让处理器去读写图 state：它本来就属于"模型调用元信息"。
            metadata.update(extra)

        context.draft = ModelRequest(
            messages=tuple(messages),
            tools=tuple(context.tools),
            response_format=inputs.get("response_format"),
            metadata=metadata,
        )

        return context
