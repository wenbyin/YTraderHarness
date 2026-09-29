"""
Document 模块的依赖装配。

各服务的请求依赖**不在这里**
--------------------------
它们已经收敛成一个泛型入口，见 app/services/dependencies.py：

    from app.services.dependencies import Service

    async def handler(
            service: DocumentRuntimeService = Service[
                DocumentRuntimeService
            ],
    ):
        ...

原来这里为每个服务写一个几乎一样的 async 生成器
（document_definition_service / document_runtime_service /
document_pipeline_query_service / document_retrieve_service /
version_chunk_service / qa_dataset_service /
processor_capability_service），加一个服务就要再抄一遍。
现在只剩下面这一个真正需要自定义的东西。

事务与提交时机
--------------
开 session、开事务、提交后才跑 post-commit 钩子 —— 这套顺序由
ServiceFactory.scope() 统一持有，不再在每个依赖函数里重写一遍。
完整的理由（为什么流水线不能压在请求事务里、为什么钩子必须在
提交之后）见 common/service/factory.py 的 scope()。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import UUID

from app.common.repository.factory import RepositoryFactory

from .repository.document import DocumentRepository
from .repository.document_version import DocumentVersionRepository


def make_pipeline_state_loader(
        session_factory,
) -> Callable[[UUID], Awaitable[dict | None]]:
    """
    产出一个"按版本号构造初始 state"的加载器。

    用闭包注入 session_factory，而不是去 `app.state` 里摸 ——
    这样本函数不依赖应用已启动，测试里也好替换。

    这个加载器由 dispatcher 的 handler 调用，跑在后台任务里，
    不持有任何请求事务，因此自己开一段短会话：仓储工厂处在
    "session_factory 模式"，用 scope() 派生一个绑定新 session 的
    工厂（begin=False —— 这是纯读路径，不需要显式事务）。

    初始 state 里的 version_number / pipeline_config /
    analysis_mode / document_type_definition_id 只在库里，
    infra 侧不认识它们 —— 所以必须由业务侧提供。
    """

    repositories = RepositoryFactory(
        session_factory=session_factory,
    )

    async def load(
            document_version_id: UUID,
    ) -> dict | None:

        async with repositories.scope(begin=False) as repos:
            version = await repos.create(
                DocumentVersionRepository
            ).get(document_version_id)

            if version is None:
                return None

            document = await repos.create(
                DocumentRepository
            ).get(version.document_id)

            if document is None:
                return None

            pipeline_config = dict(
                version.meta_data.get("pipeline_config", {})
            )

            return {
                "document_id": str(document.document_id),
                "document_version_id": str(
                    version.document_version_id
                ),
                "document_type_definition_id": str(
                    document.document_type_definition_id
                ),
                "tenant_id": (
                    None
                    if document.tenant_id is None
                    else str(document.tenant_id)
                ),
                "file_id": str(version.file_id),
                "version_number": version.version_number,
                "pipeline_config": pipeline_config,
                "analysis_mode": pipeline_config.get(
                    "analysis_mode", "default"
                ),
                # 阶段 1-6 的产物引用与扇出计划，全部由节点自己填。
                "source_ref": None,
                "detected_format": None,
                "parse_ref": None,
                "normalize_ref": None,
                "clean_ref": None,
                "extract_ref": None,
                "branches": [],
                "chunk_refs": {},
                "enrich_refs": {},
                "vector_refs": {},
                "retrieval_results": [],
                "stage_errors": [],
                "status": "pending",
            }

    return load
