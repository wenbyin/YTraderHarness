"""
按 chunk id 取单条（`GET /document-chunks/{document_chunk_id}`）。

三条边界都是"错了也不会报错、只会看起来像没数据"的那类，所以各占一条：

    默认给全文          截断了就变成"回查也没用"，而端点存在的意义正是回查
    preview_chars       只截 content，**不动 char_count**
    取不到 -> NotFound   空壳会让"id 不存在"与"正文为空"看起来一样

映射只有一份也验一下：单条与列表两条路都走 `to_chunk_summary`，两边
各写一份必然漂移（比如一边记得改 char_count、另一边忘了）。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.common.exceptions import NotFoundError
from app.services.document.application.chunk_mapping import to_chunk_summary
from app.services.document.application.document_chunk_service_impl import (
    DocumentChunkServiceImpl,
)
from app.services.document.domain.entities.document_chunk import DocumentChunk

FULL_TEXT = "深圳华强（000062.SZ）· 风险、法律与上市状态（99 条）"


def _chunk(content: str = FULL_TEXT) -> DocumentChunk:
    """一块最小可用的 chunk。字段按实体必填项给齐。"""

    return DocumentChunk(
        document_chunk_id=uuid4(),
        document_version_id=uuid4(),
        chunker_key="TEXT_TXT_RECURSIVE_CHUNKER",
        chunker_config={"chunk_size": 480},
        chunk_index=0,
        start_index=0,
        end_index=len(content),
        content=content,
        content_hash="deadbeef",
        token_count=111,
        # 刻意**不等于** len(content)：真实数据里 char_count 由 chunker 算，
        # 与 content 完全可能差几个字符（归一化时机不同）。用例拿它当哨兵
        # —— 一旦谁把 char_count 改成截断后的长度，这里立刻红。
        char_count=len(content) + 7,
        locator={"page": 1},
        metadata={"source": "eastmoney"},
        created_at=datetime(2026, 10, 16, tzinfo=UTC),
    )


class _FakeChunkRepository:
    """只实现 get()。其余方法不提供 —— 这个服务只该用到一个。"""

    def __init__(self, chunk: DocumentChunk | None) -> None:
        self._chunk = chunk
        self.calls: list = []

    async def get(self, document_chunk_id):
        self.calls.append(document_chunk_id)
        if self._chunk is None:
            return None
        if self._chunk.document_chunk_id != document_chunk_id:
            return None
        return self._chunk


def _service(chunk: DocumentChunk | None) -> DocumentChunkServiceImpl:
    return DocumentChunkServiceImpl(
        chunk_repository=_FakeChunkRepository(chunk),
    )


# ============================================================
# 取单条
# ============================================================


def test_default_returns_full_content() -> None:
    """不传 preview_chars 就是**全文** —— 默认截断会让端点失去意义。"""

    chunk = _chunk()

    summary = asyncio.run(
        _service(chunk).get_chunk(chunk.document_chunk_id)
    )

    assert summary.content == FULL_TEXT
    assert summary.content_truncated is False


def test_preview_chars_truncates_content_but_not_char_count() -> None:
    """
    截的是 content，**不是 char_count**。

    char_count 是截断前的真实长度：前端要靠它说「共 N 字符」并判断
    还有多少没看到。一旦跟着截断，界面上就会写成「共 20 字符」——
    用户以为这块就这么点，而它其实有 480。
    """

    chunk = _chunk()

    summary = asyncio.run(
        _service(chunk).get_chunk(
            chunk.document_chunk_id,
            preview_chars=10,
        )
    )

    assert summary.content == FULL_TEXT[:10]
    assert summary.content_truncated is True
    assert summary.char_count == len(FULL_TEXT) + 7


def test_preview_chars_longer_than_content_is_not_marked_truncated() -> None:
    """N 比正文还长时**不算截断** —— 标了会让前端给出没有意义的"查看全文"。"""

    chunk = _chunk()

    summary = asyncio.run(
        _service(chunk).get_chunk(
            chunk.document_chunk_id,
            preview_chars=10_000,
        )
    )

    assert summary.content == FULL_TEXT
    assert summary.content_truncated is False


def test_missing_chunk_raises_not_found() -> None:
    """取不到就是 404，**不返回空壳**。"""

    with pytest.raises(NotFoundError):
        asyncio.run(_service(None).get_chunk(uuid4()))


def test_other_ids_are_not_required() -> None:
    """
    仓储只按 document_chunk_id 查一次 —— 这个端点不校验版本、不解析
    retrieval。评测侧根本没有那两个 id（`EvalRunCase` 里没有）。
    """

    chunk = _chunk()
    repository = _FakeChunkRepository(chunk)

    service = DocumentChunkServiceImpl(chunk_repository=repository)

    asyncio.run(service.get_chunk(chunk.document_chunk_id))

    assert repository.calls == [chunk.document_chunk_id]


# ============================================================
# 映射只有一份
# ============================================================


def test_list_and_single_share_one_mapping() -> None:
    """
    列表（VersionChunkService）与单条（DocumentChunkService）必须
    给出**同一个**摘要。

    两边各留一份映射时，最典型的漂移就是"一边改了 preview_chars 的
    语义、另一边没改" —— 于是同一个 chunk 在两个页面上字符数不一样。
    """

    from app.services.document.application.version_chunk_service_impl import (
        VersionChunkServiceImpl,
    )

    chunk = _chunk()

    assert (
        VersionChunkServiceImpl._to_summary(chunk, 10)
        == to_chunk_summary(chunk, 10)
    )
