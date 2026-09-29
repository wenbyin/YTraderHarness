from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from app.common.exceptions import (
    ConflictError,
    NotFoundError,
)
from app.infra.graph import DispatchEnvelope, new_thread_id
from app.infra.graph.dispatcher import GraphDispatcher
from app.services.file.repository import FileRepository

from app.common.service.base import ServiceContext

from ..pipeline.catalog import GRAPH_NAME as DOCUMENT_GRAPH_NAME

from ..domain.entities import Document, DocumentVersion
from ..domain.schemas import (
    DocumentCreateRequest,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUpdateRequest,
    DocumentVersionDetailResponse,
    DocumentVersionListResponse,
    DocumentVersionSummary,
    ReprocessDocumentVersionRequest,
    ReprocessDocumentVersionResponse,
)
from ..domain.schemas.enum import (
    AnalysisMode,
    DefinitionStatus,
    DocumentStatus,
    ProcessingStatus,
    VersionStatus,
)
from ..repository import (
    DocumentProcessorDefinitionRepository,
    PipelineRunRepository,
    DocumentRepository,
    DocumentTypeDefinitionRepository,
    DocumentVersionRepository,
)
from .document_service import DocumentRuntimeService


logger = logging.getLogger(__name__)

# meta_data 中存放 pipeline 快照的固定键名。
# 单独抽出来是为了让读写两侧不会拼错。
PIPELINE_CONFIG_KEY = "pipeline_config"

#: 组合根在事务提交后要执行的钩子。
#: 服务侧只往列表里追加，**不执行** —— 执行时机由事务边界决定。
PostCommitHook = Callable[[], Awaitable[None]]


class DocumentRuntimeServiceImpl(DocumentRuntimeService):
    """
    DocumentRuntimeService 默认实现。

    事务边界由调用方（dependencies）持有的 session 决定，
    本服务内部不 commit / rollback。
    """

    def __init__(
            self,
            document_repository: DocumentRepository,
            version_repository: DocumentVersionRepository,
            type_definition_repository: (
                DocumentTypeDefinitionRepository
            ),
            processor_definition_repository: (
                DocumentProcessorDefinitionRepository
            ),
            file_repository: FileRepository,
            processing_dispatcher: GraphDispatcher | None = None,
            post_commit: list[PostCommitHook] | None = None,
            pipeline_run_repository: (
                PipelineRunRepository | None
            ) = None,
            *,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._document_repository = document_repository
        self._version_repository = version_repository
        self._type_repository = type_definition_repository
        self._processor_repository = (
            processor_definition_repository
        )
        self._file_repository = file_repository

        # 流水线绝不能在本服务所在的事务里跑：
        #   1. 后台任务在提交前读到的是未提交的版本，必然查不到
        #   2. 一次运行可能几分钟，压在请求上会超时
        #
        # 因此这里只**登记**一次调度，由组合根在事务提交后触发。
        # 详见 pipeline/dispatcher.py 与 dependencies.py。
        self._dispatcher = processing_dispatcher

        # 没显式给就接到上下文那条上：经 ServiceFactory.scope() 构造时，
        # ctx.post_commit 正是 scope() 在事务提交后要遍历的那个列表。
        # 两边共用同一个 list 对象，因此这里 append 的钩子必然会被执行，
        # 且必然在提交**之后**。
        self._post_commit = (
            post_commit
            if post_commit is not None
            else self._ctx.post_commit
        )

        # 只用于 reprocess 时把新运行挂到上一次运行下面，
        # 串成 reprocess 链（parent_pipeline_run_id）。
        self._pipeline_runs = pipeline_run_repository

    # ============================================================
    # Create
    # ============================================================

    async def create_document(
            self,
            request: DocumentCreateRequest,
    ) -> DocumentResponse:

        # --------------------------------------------------------
        # 1. 校验 DocumentTypeDefinition
        # --------------------------------------------------------

        type_definition = await self._type_repository.get(
            request.document_type_definition_id
        )

        if type_definition is None:
            raise NotFoundError(
                "DocumentTypeDefinition not found: "
                f"{request.document_type_definition_id}"
            )

        if type_definition.status == DefinitionStatus.DELETED:
            raise ConflictError(
                "DocumentTypeDefinition has been deleted: "
                f"{request.document_type_definition_id}"
            )

        # --------------------------------------------------------
        # 2. 校验 File
        # --------------------------------------------------------

        file_entity = await self._file_repository.get(
            request.file_id
        )

        if file_entity is None:
            raise NotFoundError(
                f"File not found: {request.file_id}"
            )

        # --------------------------------------------------------
        # 3. 租户：请求优先，否则从类型定义继承
        # --------------------------------------------------------

        tenant_id = (
            request.tenant_id
            if request.tenant_id is not None
            else type_definition.tenant_id
        )

        # --------------------------------------------------------
        # 4. document_key 唯一性
        # --------------------------------------------------------

        if request.document_key:

            existing = await self._document_repository.get_by_key(
                tenant_id,
                request.document_type_definition_id,
                request.document_key,
            )

            if existing is not None:
                raise ConflictError(
                    "document_key already exists: "
                    f"{request.document_key}"
                )

        # --------------------------------------------------------
        # 5. 按扩展名匹配前处理定义
        # --------------------------------------------------------

        extension = Path(file_entity.file_name).suffix.lower()

        processor = await self._match_processor(
            request.document_type_definition_id,
            extension,
            analysis_mode=type_definition.analysis_mode,
        )

        # --------------------------------------------------------
        # 6. 创建 Document
        # --------------------------------------------------------

        now = datetime.now(timezone.utc)

        document = Document(
            document_id=uuid4(),
            document_type_definition_id=(
                request.document_type_definition_id
            ),
            tenant_id=tenant_id,
            document_key=request.document_key,
            title=request.title,
            current_document_version_id=None,
            status=DocumentStatus.ACTIVE,
            metadata=dict(request.metadata),
            created_by=None,
            created_at=now,
            updated_at=now,
            deleted_at=None,
        )

        created_document = await self._document_repository.add(
            document
        )

        # --------------------------------------------------------
        # 7. 创建首个版本
        # --------------------------------------------------------

        version = self._build_version(
            document_id=created_document.document_id,
            file_id=request.file_id,
            version_number=1,
            processor=processor,
            type_definition=type_definition,
            extension=extension,
            now=now,
        )

        created_version = await self._version_repository.add(
            version
        )

        # --------------------------------------------------------
        # 8. 指向当前版本
        #   移动至pipeline结束后指向该版本
        # --------------------------------------------------------

        # document.current_document_version_id = (
        #     created_version.document_version_id
        # )
        # document.updated_at = now
        #
        # created_document = await self._document_repository.update(
        #     document
        # )

        # --------------------------------------------------------
        # 9. 登记处理流水线（提交后触发）
        # --------------------------------------------------------

        self._enqueue_pipeline(
            document_id=created_document.document_id,
            document_version_id=(
                created_version.document_version_id
            ),
            trigger_source="create_document",
        )

        return self._to_response(created_document)

    # ============================================================
    # Read
    # ============================================================

    async def get_document(
            self,
            document_id: UUID,
    ) -> DocumentDetailResponse:

        document = await self._require_document(document_id)

        versions = (
            await self._version_repository.list_all_by_document(
                document_id
            )
        )

        current_id = document.current_document_version_id

        summaries = [
            self._to_version_summary(version, current_id)
            for version in versions
        ]

        current = next(
            (s for s in summaries
             if s.document_version_id == current_id),
            None,
        )

        return DocumentDetailResponse(
            **self._to_response(document).model_dump(),
            current_document_version=current,
            versions=summaries,
        )

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

        documents, total = await self._document_repository.list(
            tenant_id=tenant_id,
            document_type_definition_id=(
                document_type_definition_id
            ),
            status=status,
            keyword=keyword,
            offset=(page - 1) * page_size,
            limit=page_size,
        )

        return DocumentListResponse(
            items=[self._to_response(d) for d in documents],
            page=page,
            page_size=page_size,
            total=total,
        )

    async def list_versions(
            self,
            document_id: UUID,
            *,
            status: VersionStatus | None = None,
            page: int = 1,
            page_size: int = 20,
    ) -> DocumentVersionListResponse:

        document = await self._require_document(document_id)

        versions, total = (
            await self._version_repository.list_by_document(
                document_id,
                status=status,
                offset=(page - 1) * page_size,
                limit=page_size,
            )
        )

        current_id = document.current_document_version_id

        return DocumentVersionListResponse(
            items=[
                self._to_version_summary(v, current_id)
                for v in versions
            ],
            page=page,
            page_size=page_size,
            total=total,
        )

    async def get_version(
            self,
            document_id: UUID,
            document_version_id: UUID,
    ) -> DocumentVersionDetailResponse:

        document = await self._require_document(document_id)

        version = await self._version_repository.get(
            document_version_id
        )

        if version is None:
            raise NotFoundError(
                "DocumentVersion not found: "
                f"{document_version_id}"
            )

        if version.document_id != document_id:
            raise NotFoundError(
                f"DocumentVersion {document_version_id} "
                f"does not belong to Document {document_id}"
            )

        summary = self._to_version_summary(
            version,
            document.current_document_version_id,
        )

        return DocumentVersionDetailResponse(
            **summary.model_dump(),
            pipeline_config=dict(
                version.meta_data.get(PIPELINE_CONFIG_KEY) or {}
            ),
            updated_at=version.updated_at,
        )

    # ============================================================
    # Update
    # ============================================================

    async def update_document(
            self,
            document_id: UUID,
            request: DocumentUpdateRequest,
    ) -> DocumentResponse:

        document = await self._require_document(document_id)

        if document.status == DocumentStatus.DELETED:
            raise ConflictError(
                f"Document has been deleted: {document_id}"
            )

        now = datetime.now(timezone.utc)
        type_definition = await self._type_repository.get(
            document.document_type_definition_id
        )

        # 与 create_document 对齐的 None 检查。
        # 缺了它，下面 type_definition.allow_update 会 AttributeError →
        # 500，而正确的语义是 404（类型定义行不在了）。
        # 可达性低：type_definition_impl.get() 不过滤逻辑删除，只有
        # FK 悬空（定义行被物理删除）才会返回 None —— 属于防御性缺口。
        if type_definition is None:
            raise NotFoundError(
                "DocumentTypeDefinition not found: "
                f"{document.document_type_definition_id}"
            )

        # 只有真正追加了新版本才需要跑流水线；
        # 单独改标题 / 切当前版本不需要。下面据此判断。
        created_version: DocumentVersion | None = None

        # --------------------------------------------------------
        # 1. 追加新版本
        # --------------------------------------------------------

        if request.file_id is not None:

            file_entity = await self._file_repository.get(
                request.file_id
            )
            if file_entity is None:
                raise NotFoundError(
                    f"File not found: {request.file_id}"
                )

            if not type_definition.allow_update:
                raise ConflictError(
                    "DocumentTypeDefinition does not allow "
                    f"new versions: {document_id}"
                )

            extension = Path(
                file_entity.file_name
            ).suffix.lower()

            processor = await self._match_processor(
                document.document_type_definition_id,
                extension,
                analysis_mode=type_definition.analysis_mode,
            )

            next_number = (
                await self._version_repository.max_version_number(
                    document_id
                )
            ) + 1

            version = self._build_version(
                document_id=document_id,
                file_id=request.file_id,
                version_number=next_number,
                processor=processor,
                type_definition=type_definition,
                extension=extension,
                now=now,
            )

            created_version = (
                await self._version_repository.add(version)
            )

            # document.current_document_version_id = (
            #     created_version.document_version_id
            # )

        # --------------------------------------------------------
        # 2. 切换当前版本
        # --------------------------------------------------------

        elif request.current_document_version_id is not None:

            target = await self._version_repository.get(
                request.current_document_version_id
            )

            if target is None or target.document_id != document_id:
                raise NotFoundError(
                    "DocumentVersion not found in this Document: "
                    f"{request.current_document_version_id}"
                )

            document.current_document_version_id = (
                target.document_version_id
            )

            # 指向哪个版本，哪个版本就得是 active —— 与 finalize 跑完时
            # 做的是同一件事，走同一个方法，避免两处规则漂移。
            #
            # 差异只有一处：finalize 那边要"全部执行成功"才激活，这里是
            # 用户**显式**切换，所以不加 processing_status 的前置条件 ——
            # 他就是要这一版。切到一个 failed 的版本是允许的，
            # 此时 version_status=active 与 processing_status=failed 并存，
            # 前端两个字段都要看。
            await self._version_repository.activate(
                document_id=document_id,
                version_id=target.document_version_id,
                now=now,
            )

        # --------------------------------------------------------
        # 3. Document 自身字段
        # --------------------------------------------------------

        if request.title is not None:
            document.title = request.title

        if request.metadata is not None:
            document.metadata = dict(request.metadata)

        document.updated_at = now

        updated = await self._document_repository.update(document)

        # 追加了新版本才需要跑流水线；只改标题/切版本不用。
        if created_version is not None:
            self._enqueue_pipeline(
                document_id=document_id,
                document_version_id=(
                    created_version.document_version_id
                ),
                trigger_source="update_document",
            )

        # TODO:
        #   旧版本应在新版本处理成功后置为 superseded。
        #   现在不能直接置 —— 新版本可能处理失败，
        #   那时旧版本还应保持可用。这一步属于 finalize 的职责。

        return self._to_response(updated)

    # ============================================================
    # Reprocess
    # ============================================================

    async def reprocess_version(
            self,
            document_version_id: UUID,
            request: ReprocessDocumentVersionRequest,
    ) -> ReprocessDocumentVersionResponse:

        version = await self._version_repository.get(
            document_version_id
        )

        if version is None:
            raise NotFoundError(
                "DocumentVersion not found: "
                f"{document_version_id}"
            )

        if version.version_status == VersionStatus.DELETED:
            raise ConflictError(
                "DocumentVersion has been deleted: "
                f"{document_version_id}"
            )

        document = await self._require_document(
            version.document_id
        )

        # 找到该版本的上一次运行，把新运行挂到它下面，
        # 串成 reprocess 链。前端因此能展示"这一版被重跑过几次"。
        previous_run_id: UUID | None = None

        if self._pipeline_runs is not None:
            previous, _total = await self._pipeline_runs.list_runs(
                document_version_id=document_version_id,
                limit=1,
            )
            if previous:
                previous_run_id = previous[0].pipeline_run_id

        # force 的语义是"绕过已完成即跳过的短路"，由流水线侧在
        # 选取阶段处理，这里只负责把它传下去。
        pipeline_run_id = self._enqueue_pipeline(
            document_id=document.document_id,
            document_version_id=document_version_id,
            trigger_source="reprocess",
            document_processor_definition_ids=(
                list(
                    request.document_processor_definition_ids
                )
            ),
            force=request.force,
            parent_pipeline_run_id=previous_run_id,
            checkpoint_level=request.checkpoint_level,
        )

        return ReprocessDocumentVersionResponse(
            document_version_id=document_version_id,
            status=version.processing_status,
            job_id=(
                None
                if pipeline_run_id is None
                else str(pipeline_run_id)
            ),
            message=(
                "Reprocess accepted."
                if pipeline_run_id is not None
                else "Pipeline dispatcher is not wired; "
                     "nothing was scheduled."
            ),
        )

    # ============================================================
    # Delete
    # ============================================================

    async def delete_document(
            self,
            document_id: UUID,
    ) -> None:

        document = await self._require_document(document_id)

        if document.status == DocumentStatus.DELETED:
            raise ConflictError(
                f"Document already deleted: {document_id}"
            )

        # TODO:
        #   逻辑删除后应通知 Cleanup Job 清理对象存储与各 Index。
        await self._document_repository.soft_delete(document_id)

    # ============================================================
    # Helpers
    # ============================================================

    def _enqueue_pipeline(
            self,
            *,
            document_id: UUID,
            document_version_id: UUID,
            trigger_source: str,
            document_processor_definition_ids: list[
                UUID
            ] | None = None,
            force: bool = False,
            parent_pipeline_run_id: UUID | None = None,
            checkpoint_level: str | None = None,
    ) -> UUID | None:
        """
        登记一次流水线运行，返回 pipeline_run_id。

        ⚠️ 这里**不执行**任何东西，只把闭包塞进 post_commit。

        为什么不能直接 await dispatcher.dispatch(...)：
            本方法跑在请求级事务里（dependencies.py 的
            session.begin()）。此刻 DocumentVersion 还没提交，
            后台任务立刻去读会查不到，直接失败。

        post_commit 里的钩子在 `async with session.begin()` 退出
        **之后**执行（见 dependencies.py）。若本次请求抛异常回滚，
        生成器会在 yield 处收到异常，那段代码被完全跳过 ——
        因此不会出现"事务回滚了但任务已经投出去"。

        接真队列时，把这里换成"在本事务内写 outbox 行，由 worker
        轮询"即可 —— 那是唯一能跨网络跳保持原子的模式。
        端口与调用方都不用动。
        """

        if self._dispatcher is None:
            # 未接线（例如单元测试直接构造本服务）。
            return None

        pipeline_run_id = uuid4()
        thread_id = new_thread_id(
            "docver", document_version_id, pipeline_run_id
        )

        envelope = DispatchEnvelope(
            graph_name=DOCUMENT_GRAPH_NAME,
            pipeline_run_id=pipeline_run_id,
            thread_id=thread_id,
            payload={
                "document_id": str(document_id),
                "document_version_id": str(document_version_id),
                "trigger_source": trigger_source,
                "document_processor_definition_ids": [
                    str(x)
                    for x in (
                        document_processor_definition_ids or []
                    )
                ],
                "force": force,
                "parent_pipeline_run_id": (
                    None
                    if parent_pipeline_run_id is None
                    else str(parent_pipeline_run_id)
                ),
                "checkpoint_level": checkpoint_level,
            },
        )

        dispatcher = self._dispatcher

        async def _hook() -> None:
            handle = await dispatcher.dispatch(envelope)
            if not handle.accepted:
                # 受理失败已经由 dispatcher 明确返回，不抛异常 ——
                # 此时业务事务已经提交了，抛出去只会变成 500，
                # 而调用方需要的是"文档建好了但没跑起来"这个事实。
                # 记日志 + 可观测接口里能看到没有 run 行。
                logger.warning(
                    "Pipeline dispatch rejected for version %s: %s",
                    document_version_id,
                    handle.message,
                )

        self._post_commit.append(_hook)

        return pipeline_run_id

    async def _require_document(
            self,
            document_id: UUID,
    ) -> Document:

        document = await self._document_repository.get(
            document_id
        )
        if document is None:
            raise NotFoundError(
                f"Document not found: {document_id}"
            )
        return document

    async def _match_processor(
            self,
            document_type_definition_id: UUID,
            extension: str,
            *,
            analysis_mode: AnalysisMode,
    ):
        """
        按扩展名匹配前处理定义。

        一个真实文件只有一个扩展名，因此最多命中一条。

        analysis_mode=config 时必须有匹配，否则拒绝创建；
        其他模式下允许为空，表示走系统默认 Reader。
        """

        processors = (
            await self._processor_repository
            .list_all_by_type_definition(
                document_type_definition_id
            )
        )

        matched = [
            p for p in processors
            if p.file_extension.lower() == extension
        ]

        if matched:
            return matched[0]

        if analysis_mode == AnalysisMode.CONFIG:
            raise ConflictError(
                f"analysis_mode=config 但缺少扩展名 "
                f"'{extension}' 的前处理定义，"
                "请先在类型定义下配置。"
            )

        return None

    @staticmethod
    def _build_version(
            *,
            document_id: UUID,
            file_id: UUID,
            version_number: int,
            processor,
            type_definition,
            extension: str,
            now: datetime,
    ) -> DocumentVersion:
        """
        构造新版本，并把当次使用的 pipeline 配置写入 meta_data。

        快照内容对应 document.yaml 的 PipelineConfigSnapshot：
            modality_type / loader / processor / extension

        注意：
            这里只记录**配置**，不含处理结果。
            处理流水线接入后，绑定到的 RetrievalDefinition
            与 Index 实例也应补进这个快照。
        """

        pipeline_config = {
            "modality_type": type_definition.modality_type.value,
            "analysis_mode": type_definition.analysis_mode.value,
            "file_extension": extension,
            "loader": (
                {
                    "name": processor.loader_name,
                    "config": dict(processor.loader_config),
                }
                if processor is not None
                else None
            ),
            "processor": (
                {
                    "name": processor.processor_name,
                    "config": dict(processor.processor_config),
                }
                if processor is not None
                else None
            ),
        }

        return DocumentVersion(
            document_version_id=uuid4(),
            document_id=document_id,
            file_id=file_id,
            version_number=version_number,
            version_status=VersionStatus.ACTIVE,
            document_processor_definition_id=(
                processor.document_processor_definition_id
                if processor is not None
                else None
            ),
            processing_status=ProcessingStatus.PENDING,
            meta_data={PIPELINE_CONFIG_KEY: pipeline_config},
            created_by=None,
            created_at=now,
            updated_at=now,
            completed_at=None,
        )

    # ============================================================
    # Entity -> DTO
    # ============================================================

    @staticmethod
    def _to_response(document: Document) -> DocumentResponse:

        return DocumentResponse(
            document_id=document.document_id,
            document_type_definition_id=(
                document.document_type_definition_id
            ),
            tenant_id=document.tenant_id,
            document_key=document.document_key,
            title=document.title,
            current_document_version_id=(
                document.current_document_version_id
            ),
            status=document.status,
            metadata=dict(document.metadata),
            created_by=document.created_by,
            created_at=document.created_at,
            updated_at=document.updated_at,
            deleted_at=document.deleted_at,
        )

    @staticmethod
    def _to_version_summary(
            version: DocumentVersion,
            current_document_version_id: UUID | None,
    ) -> DocumentVersionSummary:

        return DocumentVersionSummary(
            document_version_id=version.document_version_id,
            document_id=version.document_id,
            version_number=version.version_number,
            file_id=version.file_id,
            document_processor_definition_id=(
                version.document_processor_definition_id
            ),
            version_status=version.version_status,
            is_current=(
                current_document_version_id
                == version.document_version_id
            ),
            created_at=version.created_at,
        )
