"""
阶段 6：KNOWLEDGE_RETRIEVER —— 注入检索到的资料（规格 §二十九 第 8 条）。

⚠️ Context Builder 不做检索
------------------------
检索属于 Document 模块（parser / chunker / embedder / index / retrieve
那一整套）。本处理器只负责"把检索结果放进上下文"，因此：

    有 KnowledgeSource -> 问它要（真正的 RAG）
    没有               -> 用 state 里绑定的 `retrieved_context.items`

⚠️ `optional=True`
----------------
规格 §二十八 把"某个历史检索失败"列为 Optional：检索服务不可用不该
让整轮对话失败 —— 模型没有资料也能答，只是答得差一点。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.common.context import StageId

from ..base import ContextProcessor
from ..context import ContextBuildContext
from ..dependencies import dependencies_of
from ..messages import message_text
from ..spec import ContextProcessorSpec, StateBinding

__all__ = ["RetrievalProcessor"]

LABEL = "检索到的资料"


class RetrievalProcessor(ContextProcessor):
    """把检索结果注入成一段 system 片段。"""

    spec = ContextProcessorSpec(
        name="retrieval",
        order=60,
        stage=StageId.KNOWLEDGE_RETRIEVER,
        when="context_policy.include_retrieved_context",
        inputs={
            "query": StateBinding("current_user_message", required=False),
            "conversation_id": StateBinding("conversation_id", required=False),
            "documents": StateBinding("retrieved_context.items", required=False),
        },
        optional=True,
        description="注入知识库检索结果（检索本身属于 Document 模块）。",
    )

    async def process(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
    ) -> ContextBuildContext:

        deps = dependencies_of(context)

        policy = context.policy

        limit = policy.max_retrieved_items if policy else 5

        items = await self._load(deps, inputs, limit)

        if not items:
            return context

        text, source_ids = _render(items)

        if not text:
            return context

        counter = deps.resolved_token_counter()

        context.add_section(
            name="retrieval",
            label=LABEL,
            text=text,
            source_ids=source_ids,
            token_estimate=counter.count_text(text),
        )

        return context

    # --------------------------------------------------------

    async def _load(
            self,
            deps: Any,
            inputs: Mapping[str, Any],
            limit: int,
    ) -> list[Mapping[str, Any]]:

        if deps.knowledge_source is not None:

            query = inputs.get("query")

            text = message_text(query) if query is not None else ""

            if not text.strip():
                return []

            found = await deps.knowledge_source.retrieve(
                query=text,
                conversation_id=inputs.get("conversation_id"),
                limit=limit,
            )

            return [
                item for item in found if isinstance(item, Mapping)
            ][:limit]

        raw = inputs.get("documents")

        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return []

        return [
            item for item in raw if isinstance(item, Mapping)
        ][:limit]


# ============================================================
# internals
# ============================================================


def _render(
        items: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[str, ...]]:
    """
    把资料渲染成分节文本。

    认的字段：
        text / content / snippet    正文
        title / source / file_name  小标题
        id / document_id / chunk_id 来源标识

    每条前面带序号，便于模型引用（"根据 [2]"）。
    """

    blocks: list[str] = []
    ids: list[str] = []

    for index, item in enumerate(items, start=1):

        text = (
            item.get("text")
            or item.get("content")
            or item.get("snippet")
            or item.get("page_content")
        )

        if not isinstance(text, str) or not text.strip():
            continue

        title = item.get("title") or item.get("source") or item.get("file_name")

        identifier = (
            item.get("id")
            or item.get("document_id")
            or item.get("chunk_id")
        )

        ids.append(
            str(identifier) if identifier is not None else f"context#{index}"
        )

        header = f"[{index}]" + (f" {title}" if title else "")

        blocks.append(f"{header}\n{text.strip()}")

    return "\n\n".join(blocks), tuple(ids)
