from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.services.document.domain.schemas import (
    FusionRequest,
    IndexSearchResult,
    RerankRequest,
)

# 注意：ChunkStrategy 已退役。
# 分块能力并入 DocumentProcessor 体系（role=CHUNK），
# 契约在 chunker/base.py 的 DocumentChunker，实现按 role 放
# chunker/ 包下，注册进 PipelineRegistry.processors。
# 保留一个并列的 ChunkStrategy 会让"分块实现在哪"有两个答案。


class FusionStrategy(ABC):

    @abstractmethod
    async def fuse(
            self,
            request: FusionRequest,
    ) -> list[IndexSearchResult]:
        raise NotImplementedError


class RerankStrategy(ABC):

    @abstractmethod
    async def rerank(
            self,
            request: RerankRequest,
    ) -> list[IndexSearchResult]:
        raise NotImplementedError
