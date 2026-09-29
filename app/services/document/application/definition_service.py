"""
Document definition management contract.

Responsibilities:
    - DocumentTypeDefinition 的创建 / 查询 / 更新 / 逻辑删除
    - 级联保存 processors / retrievals / bindings
    - 扁平读取 DocumentProcessorDefinition / DocumentRetrievalDefinition

Non-responsibilities:
    - Document 运行时（创建 / 版本 / Chunk）
    - 实际的文件解析 / 切分 / 建索引

架构：

    管理端页面
         |
         v
    DocumentDefinitionService          <- 写只走父级
         |
         +--> DocumentTypeDefinitionRepository
         +--> DocumentProcessorDefinitionRepository
         +--> DocumentRetrievalDefinitionRepository
         +--> DocumentProcessorRetrievalBindingRepository

    扁平读：
    DocumentProcessorDefinitionRepository
    DocumentRetrievalDefinitionRepository

写入语义见 domain/schemas/definition.py 的模块文档字符串。
"""

from __future__ import annotations

from abc import abstractmethod

from uuid import UUID

from app.common.exceptions import (
    ConflictError,
    NotFoundError,
)

from app.common.service.base import BaseService

from ..domain.schemas import (
    DocumentProcessorDefinitionListResponse,
    DocumentProcessorDefinitionResponse,
    DocumentRetrievalDefinitionListResponse,
    DocumentRetrievalDefinitionResponse,
    DocumentTypeDefinitionCreateRequest,
    DocumentTypeDefinitionDetailResponse,
    DocumentTypeDefinitionListResponse,
    DocumentTypeDefinitionUpdateRequest,
)

# 兼容既有命名。异常本体在 app.common.exceptions。
DefinitionNotFoundError = NotFoundError
DefinitionConflictError = ConflictError


class DocumentDefinitionService(BaseService):
    """
    定义管理应用服务契约。

    重要规则：
        1. 写入只通过父级（DocumentTypeDefinition）。
        2. 子项按自然键识别，全量替换，未出现即软删。
        3. 更新必须携带 revision。
        4. 查询是扁平的，不需要父 id。
    """

    # ========================================================
    # DocumentTypeDefinition（唯一写入入口）
    # ========================================================

    @abstractmethod
    async def create_type_definition(
            self,
            request: DocumentTypeDefinitionCreateRequest,
    ) -> DocumentTypeDefinitionDetailResponse:
        """
        创建类型定义并级联创建子项。

        Raises:
            ConflictError:
                type_key 重复，
                或子项自然键重复，
                或 bindings 引用了不存在的子项。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_type_definition(
            self,
            document_type_definition_id: UUID,
    ) -> DocumentTypeDefinitionDetailResponse:
        """
        获取类型定义详情（含 processors / retrievals / bindings）。

        Raises:
            NotFoundError
        """
        raise NotImplementedError

    @abstractmethod
    async def update_type_definition(
            self,
            document_type_definition_id: UUID,
            request: DocumentTypeDefinitionUpdateRequest,
    ) -> DocumentTypeDefinitionDetailResponse:
        """
        更新类型定义并级联全量替换子项。

        Semantics:
            - 父级顶层字段部分更新。
            - processors / retrievals / bindings 为 None 时不触碰；
              为数组时全量替换，未出现的既有子项软删除。
            - 子项被软删时，其名下的 binding 自动级联删除。
            - 成功更新后 revision +1。

        Raises:
            NotFoundError:
                DocumentTypeDefinition 不存在。
            ConflictError:
                revision 不匹配，资源已逻辑删除，
                或 bindings 校验失败。
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_type_definition(
            self,
            document_type_definition_id: UUID,
    ) -> None:
        """
        逻辑删除类型定义及其全部子项。

        Raises:
            NotFoundError / ConflictError
        """
        raise NotImplementedError

    @abstractmethod
    async def list_type_definitions(
            self,
            *,
            tenant_id: UUID | None = None,
            modality_type=None,
            keyword: str | None = None,
            status=None,
            include_deleted: bool = False,
            page: int = 1,
            page_size: int = 20,
    ) -> DocumentTypeDefinitionListResponse:
        raise NotImplementedError

    # ========================================================
    # 扁平读
    # ========================================================

    @abstractmethod
    async def get_processor(
            self,
            processor_definition_id: UUID,
    ) -> DocumentProcessorDefinitionResponse:
        """
        扁平读取单个文件前处理定义。

        Raises:
            NotFoundError
        """
        raise NotImplementedError

    @abstractmethod
    async def list_processors(
            self,
            *,
            document_type_definition_id: UUID | None = None,
            file_extension: str | None = None,
            status=None,
            include_deleted: bool = False,
            page: int = 1,
            page_size: int = 20,
    ) -> DocumentProcessorDefinitionListResponse:
        raise NotImplementedError

    @abstractmethod
    async def get_retrieval(
            self,
            retrieval_definition_id: UUID,
    ) -> DocumentRetrievalDefinitionResponse:
        """
        扁平读取单个检索定义。

        Raises:
            NotFoundError
        """
        raise NotImplementedError

    @abstractmethod
    async def list_retrievals(
            self,
            *,
            document_type_definition_id: UUID | None = None,
            retrieval_type=None,
            status=None,
            include_deleted: bool = False,
            page: int = 1,
            page_size: int = 20,
    ) -> DocumentRetrievalDefinitionListResponse:
        raise NotImplementedError
