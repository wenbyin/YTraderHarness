"""
流水线里流转的**小值对象**。

单独成模块的理由很实际：`context.py`（工作集）需要它们，
`summarization.py`（摘要器，原来的 compaction.py）也需要它们，而它又被
`dependencies.py` import —— 如果这些小对象留在 `context.py` 里，
就会形成

    context.py -> dependencies.py -> summarization.py -> context.py

的环。把它们拆出来，"谁需要谁"就是单向的：

    values.py           纯数据，零依赖
    context.py          工作集（用它 + dependencies）
    summarization.py    摘要器（只用它，不碰工作集）
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ContextSection",
    "SummarySnapshot",
]


@dataclass(frozen=True, slots=True)
class ContextSection:
    """
    一段注入进 system 的上下文（画像 / 记忆 / 检索 / 摘要）。

    为什么要分节而不是拼成一个大字符串：产物要能回答"这一轮到底注入
    了什么"（规格 §十九 / §二十七）。分节之后 `source_ids` 就是那段
    内容的出处，快照里直接列出来。
    """

    name: str

    #: 写进 system prompt 的分节标题。
    label: str

    text: str

    #: 这段内容是否为**不可信引用数据**（检索 / 记忆 / 摘要 / 画像）。
    #:
    #: 为 True 时 `system_message()` 会用 `<ref:...>` 包起来并附上
    #: 引用数据规则（见 reference.py），而不是渲染成 `## 标题` 分节。
    #: 快照仍然拿到原文与 source_ids —— 包装只发生在产物渲染这一层。
    untrusted: bool = False

    #: 这段内容的来源标识（记忆 id / 文档 id / 摘要 revision）。
    source_ids: tuple[str, ...] = ()

    #: 估算 token（由插入方用 TokenCounter 算好）。
    token_estimate: int = 0

    def as_metadata(self) -> dict[str, Any]:

        return {
            "name": self.name,
            "label": self.label,
            "untrusted": self.untrusted,
            "chars": len(self.text),
            "token_estimate": self.token_estimate,
            "source_ids": list(self.source_ids),
        }


@dataclass(frozen=True, slots=True)
class SummarySnapshot:
    """
    对话摘要 + 它的版本与"覆盖到哪里"（规格 §十六）。

        summary                      摘要正文
        revision                     版本号（每压缩一次 +1）
        summarized_until_message_id  这段摘要已经覆盖到最后哪条消息

    三者缺一不可：没有 `revision` 就分不清"这是新摘要还是旧的"，
    没有 `summarized_until_message_id` 就会把已被摘要的消息再塞一遍
    （重复注入 = 白花 token，而且模型会看到两遍同一件事）。
    """

    text: str

    revision: int = 0

    summarized_until_message_id: str | None = None

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    def as_state(self) -> dict[str, Any]:
        """展开成 State 结构（供 `summary.*` 路径绑定）。"""

        return {
            "text": self.text,
            "revision": self.revision,
            "summarized_until_message_id": self.summarized_until_message_id,
        }

    def as_metadata(self) -> dict[str, Any]:

        return {
            "revision": self.revision,
            "chars": len(self.text),
            "summarized_until_message_id": self.summarized_until_message_id,
        }

    @classmethod
    def from_state(
            cls,
            raw: Any,
    ) -> "SummarySnapshot | None":
        """
        从 State 里读一份摘要。

        认两种形态：

            {"text": "...", "revision": 3, ...}      标准形态
            "一段摘要文本"                            简写（revision=0）

        读不到就返回 None —— **没有摘要**是完全正常的状态，
        不是错误（因此这条绑定是 optional 的）。
        """

        if raw is None:
            return None

        if isinstance(raw, str):
            return cls(text=raw) if raw.strip() else None

        if isinstance(raw, Mapping):

            text = raw.get("text") or raw.get("summary") or ""

            if not str(text).strip():
                return None

            revision = raw.get("revision")

            return cls(
                text=str(text),
                revision=revision if isinstance(revision, int) else 0,
                summarized_until_message_id=_optional_str(
                    raw.get("summarized_until_message_id")
                ),
            )

        return None


def _optional_str(value: Any) -> str | None:

    if value is None:
        return None

    text = str(value)

    return text or None
