"""
摘要器：把一批消息压成一份**事实性**摘要（**唯一**会调模型的端口）。

谁调它
------
`processors/summary.py`（order 30，SUMMARY_LOADER 阶段）—— 当
`turns.plan_window(...)` 判出"未覆盖的轮数超过了窗口上限"时，它把
"上一版摘要 + 中间的差分消息"交给这里的实现，产出一份新的完整摘要。

⚠️ 本模块不叫 compaction，尽管内容原来住在那里
-----------------------------------------
上下文构建里现在有两件**独立**的事：

    summary 阶段      生成摘要（本模块，调模型）
    compaction 阶段   在 message 列表上按轮截取（纯切片，不调模型）

它们原来挤在同一个处理器里（"达到 token 阈值就压缩"），而那让两个互不
相干的时钟互相不知道对方：压缩拿**全量对话**的 token 做判据，历史选择器
按**轮次窗口**做裁剪，中间那一段谁都不管。拆开之后，"谁生成摘要"与
"谁裁剪窗口"各有唯一一处，见 `turns.py` 顶部那条不变量。

`previous` 的语义是**合并**，不是追加
--------------------------------
提示词里明写"把新历史合并进上一版，而不是分成两段"（见
app/agent/prompt/prompts/summarization.py），因此调用方只要把
`previous` 原样传进来即可：产出的 `text` 是**完整的新摘要**，直接覆盖
`summary.text`，不是增量。`revision` 与"覆盖到哪条消息"由调用方记账
（见 `SummaryDraft` 的说明）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.model import (
    ModelRegistry,
    ModelRequest,
    ModelRole,
)

from .messages import message_text
from .values import SummarySnapshot

__all__ = [
    "ModelRegistrySummaryCompactor",
    "SummaryCompactor",
    "SummaryDraft",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SummaryDraft:
    """
    摘要模型的一次产出。

    只有文本与用量 —— 版本号与"覆盖到哪"由调用方记账
    （见 processors/summary.py）。
    """

    text: str

    provider: str | None = None
    model: str | None = None

    usage: Any = None

    #: 这次折进去的消息条数（观测用）。
    folded_messages: int = 0

    def as_metadata(self) -> dict[str, Any]:

        return {
            "provider": self.provider,
            "model": self.model,
            "chars": len(self.text),
            "folded_messages": self.folded_messages,
        }


@runtime_checkable
class SummaryCompactor(Protocol):
    """
    "把一批消息压成摘要"。

    端口化是为了两件事：
        测试可以直接注入一个假实现（不联网）
        将来换成"分层摘要 / 结构化摘要"时只换实现
    """

    async def compact(
            self,
            *,
            previous: SummarySnapshot | None,
            messages: Sequence[Any],
            max_output_tokens: int,
    ) -> SummaryDraft:
        ...


class ModelRegistrySummaryCompactor:
    """
    走 ModelRegistry 的默认实现。

    Parameters
    ----------
    model_registry
        模型注册表。**唯一**的模型入口。
    agent_type
        Agent 类型（注册表的解析键之一，由组合根/图提供）。
    model_role
        模型侧角色，默认 `ModelRole.SUMMARIZATION`。
    system_prompt
        摘要提示词。**必填** —— 这里有默认值的时候，它就是这个仓库里的
        第二份提示词正文（第一份在 app/agent/prompt/prompts/
        summarization.py），而"哪一份生效"取决于谁构造了这个摘要器。
        现在它只有一个来源：调用方从 PromptProvider 取（见
        ContextDependencies.resolved_compactor）。

        ⚠️ 为什么不在这里自己调 `get_prompt_provider()`：本模块是**类**，
        而类去摸进程级单例会让"这次用的是什么"在测试与观测里看不出来
        （Processor 那边有一条同样的纪律，见
        context/processor/dependencies.py 的模块 docstring）。
    transcript_char_limit
        单条消息进 prompt 前的截断长度。摘要模型不需要读完整的工具
        输出（那些往往几千行），截断能显著降低压缩成本；而被截掉的部分
        本来也不该进摘要（要求 3：丢弃无关细节）。
    """

    def __init__(
            self,
            *,
            model_registry: ModelRegistry,
            agent_type: str,
            system_prompt: str,
            model_role: str = ModelRole.SUMMARIZATION,
            transcript_char_limit: int = 2000,
    ) -> None:

        if model_registry is None:
            raise ValueError("ModelRegistrySummaryCompactor 需要一个 ModelRegistry。")

        self._registry = model_registry
        self._agent_type = agent_type
        self._model_role = model_role
        self._system_prompt = system_prompt
        self._limit = transcript_char_limit

    # --------------------------------------------------------

    async def compact(
            self,
            *,
            previous: SummarySnapshot | None,
            messages: Sequence[Any],
            max_output_tokens: int,
    ) -> SummaryDraft:

        request = self.build_request(
            previous=previous,
            messages=messages,
            max_output_tokens=max_output_tokens,
        )

        runtime = self._registry.get_runtime(
            agent_type=self._agent_type,
            model_role=self._model_role,
        )

        response = await runtime.ainvoke(request)

        return SummaryDraft(
            text=message_text(response.message),
            provider=response.provider,
            model=response.model,
            usage=response.usage,
            folded_messages=len(messages),
        )

    # --------------------------------------------------------

    def build_request(
            self,
            *,
            previous: SummarySnapshot | None,
            messages: Sequence[Any],
            max_output_tokens: int,
    ) -> ModelRequest:
        """
        组装交给摘要模型的请求。

        单独成方法是为了**可测**：断言"喂进去的 prompt 长什么样"不必
        真的调一次模型（而 prompt 是压缩质量的唯一变量，值得盯住）。
        """

        blocks: list[str] = []

        if previous is not None and not previous.empty:

            blocks.append(
                "【已有摘要（revision {revision}）】\n{text}".format(
                    revision=previous.revision,
                    text=previous.text.strip(),
                )
            )

        lines: list[str] = []

        for index, message in enumerate(messages):

            text = message_text(message).strip()

            if len(text) > self._limit:
                text = text[: self._limit] + "…（已截断）"

            lines.append(f"[{index}] {type(message).__name__}: {text}")

        if lines:
            blocks.append("【新增历史】\n" + "\n".join(lines))

        return ModelRequest(
            messages=(
                SystemMessage(content=self._system_prompt),
                HumanMessage(content="\n\n".join(blocks)),
            ),
            metadata={
                "purpose": "context_summarization",
                "agent_type": self._agent_type,
                "model_role": self._model_role,
                "max_output_tokens": max_output_tokens,
            },
        )

    def describe(self) -> dict[str, Any]:

        return {
            "compactor": type(self).__name__,
            "agent_type": self._agent_type,
            "model_role": self._model_role,
            "transcript_char_limit": self._limit,
        }
