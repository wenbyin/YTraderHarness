"""
`model_role=summarization` 的提示词：上下文压缩。

文本从摘要器搬过来，一字未改（它现在住在
app/agent/context/processor/summarization.py；那个模块原来叫
compaction.py，职责拆分时跟着改名）。
那份提示词原来叫 `DEFAULT_SUMMARY_PROMPT`，是压缩器的**构造参数默认值** ——
于是"用哪份提示词"这件事可以在调用方不声不响地默认掉。搬到这里之后，
压缩器不再自带文本（那个默认值已删），文本只有一个来源，就是这里。

谁用它
------
`ContextDependencies.resolved_compactor()` 建压缩器时取它（同步取 —— 这份
文本是常量，没有任何 I/O，见 app/agent/prompt/contracts.py 关于同步/异步的
说明）。

为什么写得这么**保守**
--------------------
摘要要保留"后面还答得上来"所需要的事实（目标、约束、已经做过的决定、
未决问题），而不是写一篇漂亮的概述。它会把结果拼进下一轮的 system prompt，
因此绝对不能编造 —— 提示里明写"不要推测"。
"""

from __future__ import annotations

from app.agent.model import ModelRole

from ..contracts import PromptDefinition

__all__ = ["PROMPT", "TEXT"]


#: 提示词正文。没有占位符 —— 摘要的"可变部分"（已有摘要 + 新增历史）是
#: 请求里的 **HumanMessage**，由 CompactProcessor 自己拼（见
#: ModelRegistrySummaryCompactor.build_request），不是系统提示的一部分。
TEXT = """\
你是对话上下文压缩器。把给定的对话历史压缩成一份**事实性**摘要。

要求：
1. 只写历史里明确出现过的事实，不要推测、不要补充常识。
2. 保留：用户的目标与约束、已确认的决定、未解决的问题、已经调用过的
   工具及其关键结论。
3. 丢弃：寒暄、重复内容、与后续回答无关的细节。
4. 用第三人称陈述，不要写成对话。
5. 如果已有上一版摘要，把新历史**合并**进去，而不是分成两段。
"""


def _render() -> str:
    """无参数 —— 这份提示词不随运行时数据变化。"""

    return TEXT


PROMPT = PromptDefinition(
    key=ModelRole.SUMMARIZATION,
    description="上下文压缩器的系统提示：只写事实、合并上一版摘要、不推测。",
    render=_render,
)
