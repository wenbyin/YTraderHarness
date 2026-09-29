"""
Document runtime contract.

Responsibilities:
    - Document 的创建 / 查询 / 更新 / 逻辑删除
    - DocumentVersion 的查询

Non-responsibilities:
    - Document 定义管理（见 DocumentDefinitionService）
    - 实际的文件解析 / 切分 / 建索引
    - 处理进度（后续由 progress 接口提供）

架构：

    管理端 / 业务调用方
         |
         v
    DocumentRuntimeService
         |
         +--> DocumentRepository
         +--> DocumentVersionRepository
         +--> DocumentTypeDefinitionRepository
         +--> DocumentProcessorDefinitionRepository
         +--> FileRepository（跨模块，只读文件元信息）

状态说明：
    对外只暴露 version_status（pending/active/superseded/failed/deleted）。
    processing_status 暂不对外，后续通过 progress 接口提供。
"""

from __future__ import annotations

from abc import abstractmethod

from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import (
    DocumentCreateRequest,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUpdateRequest,
    DocumentVersionDetailResponse,
    DocumentVersionListResponse,
    ReprocessDocumentVersionRequest,
    ReprocessDocumentVersionResponse,
)
from ..domain.schemas.enum import VersionStatus


class DocumentRuntimeService(BaseService):
    """
    Document 运行时应用服务契约。

    重要规则：
        1. tenant_id 未提供时从 DocumentTypeDefinition 继承。
        2. 一个真实文件只有一个扩展名，因此一个版本只选一条
           DocumentProcessorDefinition。
        3. 处理流水线未接入，创建后版本可直接生效（active）。
        4. 删除是逻辑删除。
    """

    @abstractmethod
    async def create_document(
            self,
            request: DocumentCreateRequest,
    ) -> DocumentResponse:
        """
        创建 Document 及其首个 DocumentVersion。

        Semantics:
            - file 必须已通过 File Service 上传完成。
            - document_key 在 (tenant, type_definition) 下唯一。
            - 按 file 扩展名匹配 DocumentProcessorDefinition：
                  analysis_mode=config 且无匹配 -> 拒绝
                  无匹配                        -> processor_definition_id 为空，
                                                  走系统默认 Reader
            - 当次使用的 pipeline 配置快照写入 version.meta_data。
            - RAG 执行未接入，版本会停留在未处理状态。

        Raises:
            NotFoundError / ConflictError
        """
        raise NotImplementedError

    @abstractmethod
    async def get_document(
            self,
            document_id: UUID,
    ) -> DocumentDetailResponse:
        """详情：本体 + 当前版本 + 全部版本。"""
        raise NotImplementedError

    @abstractmethod
    async def list_documents(
            self,
            *,
            tenant_id: UUID | None = None,
            document_type_definition_id: UUID | None = None,
            status=None,
            keyword: str | None = None,
            page: int = 1,
            page_size: int = 20,
    ) -> DocumentListResponse:
        raise NotImplementedError

    @abstractmethod
    async def update_document(
            self,
            document_id: UUID,
            request: DocumentUpdateRequest,
    ) -> DocumentResponse:
        """
        更新 Document。

        三种操作互不冲突时可组合：
            title / metadata          -> 改 Document 本身
            file_id                   -> 追加新版本
            current_document_version_id -> 切换当前版本

        Raises:
            NotFoundError / ConflictError
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_document(
            self,
            document_id: UUID,
    ) -> None:
        """逻辑删除。不物理删除版本与 Chunk。"""
        raise NotImplementedError

    @abstractmethod
    async def list_versions(
            self,
            document_id: UUID,
            *,
            status: VersionStatus | None = None,
            page: int = 1,
            page_size: int = 20,
    ) -> DocumentVersionListResponse:
        raise NotImplementedError

    @abstractmethod
    async def get_version(
            self,
            document_id: UUID,
            document_version_id: UUID,
    ) -> DocumentVersionDetailResponse:
        """
        Raises:
            NotFoundError:
                版本不存在，或不属于该 Document。
        """
        raise NotImplementedError

    @abstractmethod
    async def reprocess_version(
            self,
            document_version_id: UUID,
            request: ReprocessDocumentVersionRequest,
    ) -> ReprocessDocumentVersionResponse:
        """
        重新处理某个版本。

        只**登记**一次调度，不在这里跑流水线 ——
        实际触发由组合根在事务提交后进行（见 dependencies.py）。

        语义：
            document_processor_definition_ids 为空
                -> 重跑全部适用的 retrieval
            非空
                -> 只重跑这些 processor 映射到的 retrieval
            force
                -> 绕过"已完成即跳过"的短路

        返回的 job_id 就是本次运行的 pipeline_run_id。

        Raises:
            NotFoundError: 版本不存在。
            ConflictError: 版本已逻辑删除。
        """
        raise NotImplementedError
