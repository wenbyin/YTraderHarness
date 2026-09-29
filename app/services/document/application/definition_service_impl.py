from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.common.exceptions import (
    ConflictError,
    NotFoundError,
)

from app.common.service.base import ServiceContext

from ..domain.entities import (
    DocumentProcessorDefinition,
    DocumentRetrievalDefinition,
    DocumentTypeDefinition,
)
from ..domain.schemas import (
    DocumentProcessorDefinitionListResponse,
    DocumentProcessorDefinitionRequest,
    DocumentProcessorDefinitionResponse,
    DocumentRetrievalDefinitionListResponse,
    DocumentRetrievalDefinitionRequest,
    DocumentRetrievalDefinitionResponse,
    DocumentTypeDefinitionCreateRequest,
    DocumentTypeDefinitionDetailResponse,
    DocumentTypeDefinitionListResponse,
    DocumentTypeDefinitionResponse,
    DocumentTypeDefinitionUpdateRequest,
    ProcessorRetrievalBindingRequest,
    ProcessorRetrievalBindingResponse,
)
from ..domain.schemas.enum import (
    DefinitionStatus,
    RetrievalType,
    StorageBackend,
)
from ..repository import (
    DocumentProcessorDefinitionRepository,
    DocumentProcessorRetrievalBindingRepository,
    DocumentRetrievalDefinitionRepository,
    DocumentTypeDefinitionRepository,
)
from .definition_service import DocumentDefinitionService


# ============================================================
# retrieval_type -> 默认 storage_backend
# ============================================================
#
# 调用方不传 storage_backend 时按此映射补齐。

_DEFAULT_STORAGE: dict[RetrievalType, StorageBackend] = {
    RetrievalType.DOCUMENT: StorageBackend.POSTGRESQL,
    RetrievalType.KEYWORD: StorageBackend.ELASTICSEARCH,
    RetrievalType.SPARSE_VECTOR: StorageBackend.ELASTICSEARCH,
    RetrievalType.DENSE_VECTOR: StorageBackend.MILVUS,
    RetrievalType.GRAPH: StorageBackend.NEO4J,
}


class DocumentDefinitionServiceImpl(DocumentDefinitionService):
    """
    DocumentDefinitionService 默认实现。

    事务边界由调用方（dependencies）持有的 session 决定，
    本服务内部不 commit / rollback。
    """

    def __init__(
            self,
            type_definition_repository: (
                DocumentTypeDefinitionRepository
            ),
            processor_definition_repository: (
                DocumentProcessorDefinitionRepository
            ),
            retrieval_definition_repository: (
                DocumentRetrievalDefinitionRepository
            ),
            binding_repository: (
                DocumentProcessorRetrievalBindingRepository
            ),
            *,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._type_repository = type_definition_repository
        self._processor_repository = (
            processor_definition_repository
        )
        self._retrieval_repository = (
            retrieval_definition_repository
        )
        self._binding_repository = binding_repository

    # ============================================================
    # Create
    # ============================================================

    async def create_type_definition(
            self,
            request: DocumentTypeDefinitionCreateRequest,
    ) -> DocumentTypeDefinitionDetailResponse:

        existing = await self._type_repository.get_by_type_key(
            request.type_key
        )
        if existing is not None:
            raise ConflictError(
                f"type_key already exists: {request.type_key}"
            )

        now = datetime.now(timezone.utc)

        type_definition = DocumentTypeDefinition(
            document_type_definition_id=uuid4(),
            tenant_id=request.tenant_id,
            type_key=request.type_key,
            name=request.name,
            description=request.description,
            modality_type=request.modality_type,
            analysis_mode=request.analysis_mode,
            revision=1,
            allow_update=request.allow_update,
            version_mode=request.version_mode,
            fusion_strategy=request.fusion_strategy,
            fusion_config=dict(request.fusion_config),
            rerank_strategy=request.rerank_strategy,
            rerank_config=dict(request.rerank_config),
            metadata=dict(request.metadata),
            status=DefinitionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )

        created = await self._type_repository.add(type_definition)

        # 创建时三个数组都有默认值（空数组），直接全量替换。
        processors = await self._save_processors(
            created.document_type_definition_id,
            request.processors,
            now,
        )
        retrievals = await self._save_retrievals(
            created.document_type_definition_id,
            request.retrievals,
            now,
        )
        await self._replace_bindings(
            processors,
            retrievals,
            request.bindings,
        )

        return await self._to_detail(created)

    # ============================================================
    # Read
    # ============================================================

    async def get_type_definition(
            self,
            document_type_definition_id: UUID,
    ) -> DocumentTypeDefinitionDetailResponse:

        type_definition = await self._require_type_definition(
            document_type_definition_id
        )

        return await self._to_detail(type_definition)

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

        effective = self._resolve_status(status, include_deleted)

        definitions, total = await self._type_repository.list(
            tenant_id=tenant_id,
            status=effective,
            keyword=keyword,
            offset=(page - 1) * page_size,
            limit=page_size,
        )

        items = [
            self._to_type_response(d)
            for d in definitions
        ]

        # modality_type 是内存过滤：仓储层未按该列索引，
        # 且类型定义数量有限。
        if modality_type is not None:
            items = [
                i for i in items
                if i.modality_type == modality_type
            ]

        return DocumentTypeDefinitionListResponse(
            items=items,
            page=page,
            page_size=page_size,
            total=total,
            total_pages=self._total_pages(total, page_size),
        )

    # ============================================================
    # Update
    # ============================================================

    async def update_type_definition(
            self,
            document_type_definition_id: UUID,
            request: DocumentTypeDefinitionUpdateRequest,
    ) -> DocumentTypeDefinitionDetailResponse:

        type_definition = await self._require_type_definition(
            document_type_definition_id
        )

        if type_definition.status == DefinitionStatus.DELETED:
            raise ConflictError(
                "DocumentTypeDefinition has been deleted: "
                f"{document_type_definition_id}"
            )

        # --------------------------------------------------------
        # 乐观锁
        # --------------------------------------------------------

        if request.revision != type_definition.revision:
            raise ConflictError(
                "revision mismatch: expected "
                f"{type_definition.revision}, got "
                f"{request.revision}. "
                "该记录已被其他人修改，请重新加载。"
            )

        now = datetime.now(timezone.utc)

        # --------------------------------------------------------
        # 父级顶层字段：部分更新
        # --------------------------------------------------------

        provided = request.model_dump(
            exclude_unset=True,
            exclude={"processors", "retrievals", "bindings",
                     "revision"},
        )

        for field_name, value in provided.items():
            setattr(type_definition, field_name, value)

        type_definition.revision = type_definition.revision + 1
        type_definition.updated_at = now

        await self._type_repository.update(type_definition)

        # --------------------------------------------------------
        # 子项：全量替换
        # --------------------------------------------------------

        processors = None
        retrievals = None

        if request.processors is not None:
            processors = await self._save_processors(
                document_type_definition_id,
                request.processors,
                now,
            )

        if request.retrievals is not None:
            retrievals = await self._save_retrievals(
                document_type_definition_id,
                request.retrievals,
                now,
            )

        # bindings 用下标引用 processors / retrievals，
        # 下标只在同一次 payload 内才有意义，
        # 因此必须与这两个数组一起提交。
        if request.bindings is not None:

            if processors is None or retrievals is None:
                raise ConflictError(
                    "bindings 使用下标引用 processors / retrievals，"
                    "必须与这两个数组在同一次请求中一起提交。"
                )

            await self._replace_bindings(
                processors,
                retrievals,
                request.bindings,
            )

        return await self._to_detail(type_definition)

    # ============================================================
    # Delete
    # ============================================================

    async def delete_type_definition(
            self,
            document_type_definition_id: UUID,
    ) -> None:

        type_definition = await self._require_type_definition(
            document_type_definition_id
        )

        if type_definition.status == DefinitionStatus.DELETED:
            raise ConflictError(
                "DocumentTypeDefinition already deleted: "
                f"{document_type_definition_id}"
            )

        now = datetime.now(timezone.utc)

        processors = (
            await self._processor_repository
            .list_all_by_type_definition(
                document_type_definition_id,
                include_deleted=True,
            )
        )
        retrievals = (
            await self._retrieval_repository
            .list_all_by_type_definition(
                document_type_definition_id,
                include_deleted=True,
            )
        )

        for processor in processors:
            if processor.status == DefinitionStatus.ACTIVE:
                processor.status = DefinitionStatus.DELETED
                processor.deleted_at = now
                processor.updated_at = now
                await self._processor_repository.update(processor)

        for retrieval in retrievals:
            if retrieval.status == DefinitionStatus.ACTIVE:
                retrieval.status = DefinitionStatus.DELETED
                retrieval.deleted_at = now
                retrieval.updated_at = now
                await self._retrieval_repository.update(retrieval)

        # 绑定是纯配置关系，直接物理删除。
        await self._binding_repository.delete_by_processors(
            [p.document_processor_definition_id for p in processors]
        )

        type_definition.status = DefinitionStatus.DELETED
        type_definition.updated_at = now
        type_definition.revision = type_definition.revision + 1
        await self._type_repository.update(type_definition)

    # ============================================================
    # 扁平读
    # ============================================================

    async def get_processor(
            self,
            processor_definition_id: UUID,
    ) -> DocumentProcessorDefinitionResponse:

        processor = await self._processor_repository.get(
            processor_definition_id
        )
        if processor is None:
            raise NotFoundError(
                "DocumentProcessorDefinition not found: "
                f"{processor_definition_id}"
            )

        bindings = (
            await self._binding_repository.list_by_processor(
                processor_definition_id
            )
        )
        retrieval_by_id = {
            r.retrieval_definition_id: r
            for r in await self._retrieval_repository.list_by_ids(
                [b.document_retrieval_definition_id
                 for b in bindings]
            )
        }

        return self._to_processor_response(
            processor,
            retrieval_by_id=retrieval_by_id,
            bindings=bindings,
        )

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

        processors, total = await self._processor_repository.list(
            document_type_definition_id=document_type_definition_id,
            file_extension=file_extension,
            status=self._resolve_status(status, include_deleted),
            offset=(page - 1) * page_size,
            limit=page_size,
        )

        # 一次性取回这些 processor 的绑定，避免 N+1。
        bindings = (
            await self._binding_repository.list_by_processors(
                [p.document_processor_definition_id for p in processors]
            )
        )
        retrieval_by_id = {
            r.retrieval_definition_id: r
            for r in await self._retrieval_repository.list_by_ids(
                [b.document_retrieval_definition_id
                 for b in bindings]
            )
        }

        by_processor: dict[UUID, list] = {}
        for binding in bindings:
            by_processor.setdefault(
                binding.document_processor_definition_id, []
            ).append(binding)

        return DocumentProcessorDefinitionListResponse(
            items=[
                self._to_processor_response(
                    p,
                    retrieval_by_id=retrieval_by_id,
                    bindings=by_processor.get(
                        p.document_processor_definition_id, []
                    ),
                )
                for p in processors
            ],
            page=page,
            page_size=page_size,
            total=total,
            total_pages=self._total_pages(total, page_size),
        )

    async def get_retrieval(
            self,
            retrieval_definition_id: UUID,
    ) -> DocumentRetrievalDefinitionResponse:

        retrieval = await self._retrieval_repository.get(
            retrieval_definition_id
        )
        if retrieval is None:
            raise NotFoundError(
                "DocumentRetrievalDefinition not found: "
                f"{retrieval_definition_id}"
            )

        return await self._to_retrieval_response(retrieval)

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

        retrievals, total = await self._retrieval_repository.list(
            document_type_definition_id=document_type_definition_id,
            retrieval_type=retrieval_type,
            status=self._resolve_status(status, include_deleted),
            offset=(page - 1) * page_size,
            limit=page_size,
        )

        return DocumentRetrievalDefinitionListResponse(
            items=[
                await self._to_retrieval_response(r)
                for r in retrievals
            ],
            page=page,
            page_size=page_size,
            total=total,
            total_pages=self._total_pages(total, page_size),
        )

    # ============================================================
    # 级联保存
    # ============================================================

    async def _save_processors(
            self,
            document_type_definition_id: UUID,
            items: list[DocumentProcessorDefinitionRequest],
            now: datetime,
    ) -> list[DocumentProcessorDefinition]:
        """
        全量替换 processors。

        自然键 file_extension：
            命中（含已软删）-> 复活并覆盖字段
            未命中          -> 新建
        未出现在 items 中的既有 processor -> 软删，
        并级联删除其绑定。
        """

        # 自然键重复，直接拒绝
        extensions = [i.file_extension for i in items]
        if len(extensions) != len(set(extensions)):
            raise ConflictError(
                "duplicate file_extension in processors: "
                f"{sorted({e for e in extensions if extensions.count(e) > 1})}"
            )

        existing = (
            await self._processor_repository
            .list_all_by_type_definition(
                document_type_definition_id,
                include_deleted=True,
            )
        )
        by_extension = {p.file_extension: p for p in existing}

        kept_ids: list[UUID] = []

        for item in items:

            current = by_extension.get(item.file_extension)

            if current is None:
                created = await self._processor_repository.add(
                    DocumentProcessorDefinition(
                        document_processor_definition_id=uuid4(),
                        document_type_definition_id=(
                            document_type_definition_id
                        ),
                        file_extension=item.file_extension,
                        mime_types=list(item.mime_types),
                        loader_name=item.loader_name,
                        loader_config=dict(item.loader_config),
                        processor_name=item.processor_name,
                        processor_config=dict(
                            item.processor_config
                        ),
                        status=DefinitionStatus.ACTIVE,
                        created_by=None,
                        created_at=now,
                        updated_at=now,
                        deleted_at=None,
                    )
                )
                kept_ids.append(created.document_processor_definition_id)
                by_extension[item.file_extension] = created
                continue

            current.mime_types = list(item.mime_types)
            current.loader_name = item.loader_name
            current.loader_config = dict(item.loader_config)
            current.processor_name = item.processor_name
            current.processor_config = dict(item.processor_config)
            current.status = DefinitionStatus.ACTIVE
            current.deleted_at = None
            current.updated_at = now

            await self._processor_repository.update(current)
            kept_ids.append(current.document_processor_definition_id)

        removed = [
            p for p in existing
            if p.document_processor_definition_id not in set(kept_ids)
            and p.status == DefinitionStatus.ACTIVE
        ]

        for processor in removed:
            processor.status = DefinitionStatus.DELETED
            processor.deleted_at = now
            processor.updated_at = now
            await self._processor_repository.update(processor)

        # 软删的子项，其绑定立即级联删除，避免悬空引用。
        await self._binding_repository.delete_by_processors(
            [p.document_processor_definition_id for p in removed]
        )

        # 必须按 payload 顺序返回：
        # bindings 用下标引用 processors，顺序变了下标就错了。
        # 不能用 list_all_by_type_definition（它按 file_extension 排序）。
        return [
            by_extension[item.file_extension]
            for item in items
        ]

    async def _save_retrievals(
            self,
            document_type_definition_id: UUID,
            items: list[DocumentRetrievalDefinitionRequest],
            now: datetime,
    ) -> list[DocumentRetrievalDefinition]:
        """全量替换 retrievals，语义同 _save_processors。"""

        types = [i.retrieval_type for i in items]
        if len(types) != len(set(types)):
            raise ConflictError(
                "duplicate retrieval_type in retrievals: "
                f"{sorted({t for t in types if types.count(t) > 1})}"
            )

        existing = (
            await self._retrieval_repository
            .list_all_by_type_definition(
                document_type_definition_id,
                include_deleted=True,
            )
        )
        by_type = {r.retrieval_type: r for r in existing}

        kept_ids: list[UUID] = []

        for item in items:

            backend = item.storage_backend or _DEFAULT_STORAGE.get(
                item.retrieval_type,
                StorageBackend.POSTGRESQL,
            )

            current = by_type.get(item.retrieval_type)

            if current is None:
                created = await self._retrieval_repository.add(
                    DocumentRetrievalDefinition(
                        retrieval_definition_id=uuid4(),
                        document_type_definition_id=(
                            document_type_definition_id
                        ),
                        retrieval_type=item.retrieval_type,
                        chunker_name=item.chunker_name,
                        chunker_config=dict(item.chunker_config),
                        retriever_name=item.retriever_name,
                        retriever_config=dict(
                            item.retriever_config
                        ),
                        storage_backend=backend,
                        storage_config=dict(item.storage_config),
                        status=DefinitionStatus.ACTIVE,
                        created_by=None,
                        created_at=now,
                        updated_at=now,
                        deleted_at=None,
                    )
                )
                kept_ids.append(created.retrieval_definition_id)
                by_type[item.retrieval_type] = created
                continue

            current.chunker_name = item.chunker_name
            current.chunker_config = dict(item.chunker_config)
            current.retriever_name = item.retriever_name
            current.retriever_config = dict(item.retriever_config)
            current.storage_backend = backend
            current.storage_config = dict(item.storage_config)
            current.status = DefinitionStatus.ACTIVE
            current.deleted_at = None
            current.updated_at = now

            await self._retrieval_repository.update(current)
            kept_ids.append(current.retrieval_definition_id)

        removed = [
            r for r in existing
            if r.retrieval_definition_id not in set(kept_ids)
            and r.status == DefinitionStatus.ACTIVE
        ]

        for retrieval in removed:
            retrieval.status = DefinitionStatus.DELETED
            retrieval.deleted_at = now
            retrieval.updated_at = now
            await self._retrieval_repository.update(retrieval)

        await self._binding_repository.delete_by_retrievals(
            [r.retrieval_definition_id for r in removed]
        )

        # 同 _save_processors：按 payload 顺序返回，供下标引用。
        return [
            by_type[item.retrieval_type]
            for item in items
        ]

    async def _replace_bindings(
            self,
            processors: list[DocumentProcessorDefinition],
            retrievals: list[DocumentRetrievalDefinition],
            bindings: list[ProcessorRetrievalBindingRequest],
    ) -> None:
        """
        全量替换绑定关系。

        bindings 用下标引用同一次 payload 中的
        processors / retrievals，因此：

            - 传入的两个列表必须保持 payload 顺序
              （_save_processors / _save_retrievals 保证）
            - bindings 必须与这两个数组一起提交，
              不允许单独提交（下标脱离上下文没有意义）

        校验：
            - 两个下标都必须在数组范围内
            - (processor_index, retrieval_index) 不可重复

        实现上按 processor 分组调用 replace_for_processor，
        没有绑定的 processor 传空列表即解绑全部。
        """

        grouped: dict[UUID, list[UUID]] = {
            p.document_processor_definition_id: []
            for p in processors
        }

        seen: set[tuple[int, int]] = set()
        details: dict[str, str] = {}

        for position, binding in enumerate(bindings):

            if not 0 <= binding.processor_index < len(processors):
                details[
                    f"bindings[{position}].processor_index"
                ] = (
                    f"{binding.processor_index} 越界，"
                    f"processors 长度为 {len(processors)}"
                )
                continue

            if not 0 <= binding.retrieval_index < len(retrievals):
                details[
                    f"bindings[{position}].retrieval_index"
                ] = (
                    f"{binding.retrieval_index} 越界，"
                    f"retrievals 长度为 {len(retrievals)}"
                )
                continue

            pair = (
                binding.processor_index,
                binding.retrieval_index,
            )

            if pair in seen:
                details[f"bindings[{position}]"] = (
                    f"(processor_index={pair[0]}, "
                    f"retrieval_index={pair[1]}) 重复"
                )
                continue

            seen.add(pair)

            processor = processors[binding.processor_index]
            retrieval = retrievals[binding.retrieval_index]

            grouped[
                processor.document_processor_definition_id
            ].append(retrieval.retrieval_definition_id)

        if details:
            raise ConflictError(
                "binding validation failed: " + str(details)
            )

        for processor_definition_id, retrieval_ids in grouped.items():
            await self._binding_repository.replace_for_processor(
                processor_definition_id,
                retrieval_ids,
            )

    # ============================================================
    # Guards / Helpers
    # ============================================================

    async def _require_type_definition(
            self,
            document_type_definition_id: UUID,
    ) -> DocumentTypeDefinition:

        definition = await self._type_repository.get(
            document_type_definition_id
        )
        if definition is None:
            raise NotFoundError(
                "DocumentTypeDefinition not found: "
                f"{document_type_definition_id}"
            )
        return definition

    @staticmethod
    def _resolve_status(status, include_deleted: bool):
        """status 显式指定优先；否则按 include_deleted 决定。"""

        if status is not None:
            return status
        if include_deleted:
            return None
        return DefinitionStatus.ACTIVE

    @staticmethod
    def _total_pages(total: int, page_size: int) -> int:
        return (total + page_size - 1) // page_size

    async def _to_detail(
            self,
            type_definition: DocumentTypeDefinition,
    ) -> DocumentTypeDefinitionDetailResponse:

        processors = (
            await self._processor_repository
            .list_all_by_type_definition(
                type_definition.document_type_definition_id
            )
        )
        retrievals = (
            await self._retrieval_repository
            .list_all_by_type_definition(
                type_definition.document_type_definition_id
            )
        )
        bindings = (
            await self._binding_repository.list_by_type_definition(
                type_definition.document_type_definition_id
            )
        )

        retrieval_by_id = {
            r.retrieval_definition_id: r for r in retrievals
        }

        by_processor: dict[UUID, list] = {}
        for binding in bindings:
            by_processor.setdefault(
                binding.document_processor_definition_id, []
            ).append(binding)

        binding_items = [
            ProcessorRetrievalBindingResponse(
                file_extension=processor.file_extension,
                retrieval_type=retrieval_by_id[
                    binding.document_retrieval_definition_id
                ].retrieval_type,
            )
            for processor in processors
            for binding in by_processor.get(
                processor.document_processor_definition_id, []
            )
            if binding.document_retrieval_definition_id
            in retrieval_by_id
        ]

        return DocumentTypeDefinitionDetailResponse(
            **self._to_type_response(type_definition).model_dump(),
            processors=[
                self._to_processor_response(
                    p,
                    retrieval_by_id=retrieval_by_id,
                    bindings=by_processor.get(
                        p.document_processor_definition_id, []
                    ),
                )
                for p in processors
            ],
            retrievals=[
                self._to_retrieval_response_sync(
                    r,
                    file_extensions=[
                        p.file_extension
                        for p in processors
                        if any(
                            b.document_retrieval_definition_id
                            == r.retrieval_definition_id
                            for b in by_processor.get(
                                p.document_processor_definition_id, []
                            )
                        )
                    ],
                )
                for r in retrievals
            ],
            bindings=binding_items,
        )

    # ============================================================
    # Entity -> DTO
    # ============================================================

    @staticmethod
    def _to_type_response(
            definition: DocumentTypeDefinition,
    ) -> DocumentTypeDefinitionResponse:

        return DocumentTypeDefinitionResponse(
            document_type_definition_id=(
                definition.document_type_definition_id
            ),
            revision=definition.revision,
            tenant_id=definition.tenant_id,
            type_key=definition.type_key,
            name=definition.name,
            description=definition.description,
            modality_type=definition.modality_type,
            analysis_mode=definition.analysis_mode,
            allow_update=definition.allow_update,
            version_mode=definition.version_mode,
            fusion_strategy=definition.fusion_strategy,
            fusion_config=dict(definition.fusion_config),
            rerank_strategy=definition.rerank_strategy,
            rerank_config=dict(definition.rerank_config),
            metadata=dict(definition.metadata),
            status=definition.status,
            created_at=definition.created_at,
            updated_at=definition.updated_at,
        )

    @staticmethod
    def _to_processor_response(
            processor: DocumentProcessorDefinition,
            *,
            retrieval_by_id: dict,
            bindings: list,
    ) -> DocumentProcessorDefinitionResponse:

        return DocumentProcessorDefinitionResponse(
            processor_definition_id=(
                processor.document_processor_definition_id
            ),
            document_type_definition_id=(
                processor.document_type_definition_id
            ),
            file_extension=processor.file_extension,
            mime_types=list(processor.mime_types),
            loader_name=processor.loader_name,
            loader_config=dict(processor.loader_config),
            processor_name=processor.processor_name,
            processor_config=dict(processor.processor_config),
            retrieval_types=[
                retrieval_by_id[
                    b.document_retrieval_definition_id
                ].retrieval_type
                for b in bindings
                if b.document_retrieval_definition_id
                in retrieval_by_id
            ],
            status=processor.status,
            created_at=processor.created_at,
            updated_at=processor.updated_at,
        )

    @staticmethod
    def _to_retrieval_response_sync(
            retrieval: DocumentRetrievalDefinition,
            *,
            file_extensions: list[str],
    ) -> DocumentRetrievalDefinitionResponse:

        return DocumentRetrievalDefinitionResponse(
            retrieval_definition_id=(
                retrieval.retrieval_definition_id
            ),
            document_type_definition_id=(
                retrieval.document_type_definition_id
            ),
            retrieval_type=retrieval.retrieval_type,
            chunker_name=retrieval.chunker_name,
            chunker_config=dict(retrieval.chunker_config),
            retriever_name=retrieval.retriever_name,
            retriever_config=dict(retrieval.retriever_config),
            storage_backend=retrieval.storage_backend,
            storage_config=dict(retrieval.storage_config),
            file_extensions=file_extensions,
            status=retrieval.status,
            created_at=retrieval.created_at,
            updated_at=retrieval.updated_at,
        )

    async def _to_retrieval_response(
            self,
            retrieval: DocumentRetrievalDefinition,
    ) -> DocumentRetrievalDefinitionResponse:
        """扁平读取时反查绑定它的文件后缀名。"""

        processors = (
            await self._processor_repository
            .list_all_by_type_definition(
                retrieval.document_type_definition_id,
                include_deleted=True,
            )
        )
        bindings = (
            await self._binding_repository.list_by_processors(
                [p.document_processor_definition_id for p in processors]
            )
        )

        extension_by_processor = {
            p.document_processor_definition_id: p.file_extension
            for p in processors
        }

        return self._to_retrieval_response_sync(
            retrieval,
            file_extensions=sorted({
                extension_by_processor[
                    b.document_processor_definition_id
                ]
                for b in bindings
                if b.document_retrieval_definition_id
                == retrieval.retrieval_definition_id
                and b.document_processor_definition_id
                in extension_by_processor
            }),
        )
