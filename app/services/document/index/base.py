from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from app.services.document.domain.schemas import (
    IndexUpsertRequest,
    IndexDeleteRequest,
    IndexSearchRequest,
    IndexSearchResponse,
)
from app.services.document.domain.schemas.enum import StorageBackend


class IndexBackend(ABC):
    """
    文档检索 Index 的统一抽象。

    实现：
        ElasticsearchIndexBackend
        MilvusIndexBackend
        FaissIndexBackend
        QdrantIndexBackend
        ...

    注意：
        IndexBackend 只负责：
            - 写入 Index
            - 删除 Index
            - 从 Index 召回

        不负责：
            - Chunk
            - Fusion
            - Rerank
            - DocumentType 路由

    注册方式与其它能力不同
    ----------------------
    Parser / Chunker / Embedder 那些按**名字**注册（字符串键，来自配置），
    而 IndexBackend 按 **StorageBackend 枚举**注册 —— 一个后端类型只有
    一个实现，配置里存的是枚举值而不是名字。因此它不在
    CAPABILITY_PACKAGES 那套里，另有一套发现逻辑（见 registry.py）。

    客户端由外部注入
    ----------------
    后端是有状态的：它持有一个 HTTP/gRPC 客户端。客户端不在这里构造，
    而是由组合根（main.py）从 **datasource_manager** 取现成的给进来 ——
    那些客户端的生命周期、连接池、认证都由 infra 层管，
    在业务层再建一套会多出一份连接池与一份凭据来源。
    """

    #: 本实现服务哪个存储后端。注册表按这个值索引它。
    #: 具体实现**必须**覆盖；留 None 的不会被注册。
    backend_type: ClassVar[StorageBackend | None] = None

    @abstractmethod
    async def upsert(
            self,
            request: IndexUpsertRequest,
    ) -> None:
        """
        将一个 DocumentVersion 的 Canonical Chunks
        投影写入当前 Index。
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(
            self,
            request: IndexDeleteRequest,
    ) -> None:
        """
        删除指定 DocumentVersion 在当前 Index 中的内容。
        """
        raise NotImplementedError

    @abstractmethod
    async def search(
            self,
            request: IndexSearchRequest,
    ) -> IndexSearchResponse:
        """
        从当前 Index 进行一次独立召回。
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """
        检查当前 Index 后端是否可用。
        """
        raise NotImplementedError
