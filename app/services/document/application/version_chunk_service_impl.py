"""
处理结果页查询的实现。

数据来源
--------
三个仓储，都是只读：

    DocumentVersionRepository              校验版本存在
    DocumentVersionRetrievalBindingRepository  该版本参与过哪些 retrieval
    DocumentRetrievalDefinitionRepository  那些 retrieval 的展示信息
    DocumentChunkRepository                某一路的 chunk

一次请求里最多四跳查询：版本 -> 绑定 -> 定义（一次批量）-> chunk。
retrieval 的数量级是个位数，所以没有做更激进的合并。
"""

from __future__ import annotations

import json
import math
from typing import Any
from uuid import UUID

from app.common.exceptions import NotFoundError

from app.common.service.base import ServiceContext

from ..chunker.identity import (
    chunker_identity,
    identity_digest,
)
from ..domain.entities import DocumentChunk
from ..domain.schemas.enum import EmbeddingOutput
from ..domain.schemas.version_chunks import (
    ChunkProjection,
    ChunkProjectionEntry,
    ChunkProjectionListResponse,
    ChunkerBinding,
    DocumentChunkListResponse,
    DocumentChunkSummary,
    DocumentVersionRetrievalListResponse,
    DocumentVersionRetrievalSummary,
    SparseSummary,
    VectorSummary,
)
from ..pipeline.observability.artifacts import ArtifactStore
from ..pipeline.registry import PipelineRegistry
from ..repository.document_chunk import DocumentChunkRepository
from ..repository.document_version import (
    DocumentVersionRepository,
)
from ..repository.document_version_retrieval_binding import (
    DocumentVersionRetrievalBindingRepository,
)
from ..repository.pipeline_run import PipelineRunRepository
from ..repository.retrieval_definition import (
    DocumentRetrievalDefinitionRepository,
)
from .version_chunk_service import VersionChunkService

__all__ = ["VersionChunkServiceImpl"]


#: 摘要里回显多少个分量 / 词项。
_SUMMARY_HEAD = 8

#: 产物里代表"向量"的键 -> 形态。graph 也在这里，
#: 但它的产物形态是局部图，摘要另行处理（暂不展开）。
_VECTOR_KEYS: dict[str, EmbeddingOutput] = {
    "vector": EmbeddingOutput.DENSE,
    "sparse": EmbeddingOutput.SPARSE,
}

#: 投影里属于"向量"的键，组装 fields 时要剔掉。
_VECTOR_FIELD_KEYS = frozenset({"vector", "sparse", "graph", "kind"})


class VersionChunkServiceImpl(VersionChunkService):

    def __init__(
            self,
            *,
            version_repository: DocumentVersionRepository,
            binding_repository: (
                DocumentVersionRetrievalBindingRepository
            ),
            retrieval_definition_repository: (
                DocumentRetrievalDefinitionRepository
            ),
            chunk_repository: DocumentChunkRepository,
            run_repository: PipelineRunRepository,
            artifact_store: ArtifactStore,
            pipeline_registry: PipelineRegistry,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._version_repository = version_repository
        self._binding_repository = binding_repository
        self._retrieval_repository = retrieval_definition_repository
        self._chunk_repository = chunk_repository
        self._run_repository = run_repository
        self._artifact_store = artifact_store
        # 用来把 retrieval 定义解析成 chunker 身份 —— chunk 不记
        # retrieval，"哪几路用它"得由身份推。
        self._pipeline_registry = pipeline_registry

    # ========================================================
    # tab 列表
    # ========================================================

    async def list_retrievals(
            self,
            document_version_id: UUID,
    ) -> DocumentVersionRetrievalListResponse:

        await self._require_version(document_version_id)

        bindings = await self._binding_repository.list_by_version(
            document_version_id
        )

        if not bindings:
            # 版本存在但一路都没跑过（还在排队、或者没有匹配的 retrieval）。
            # 返回空列表而不是 404 —— "这个版本没有可展示的 tab" 是合法状态，
            # 前端应当显示空态而不是错误页。
            return DocumentVersionRetrievalListResponse(
                items=[], total=0,
            )

        # 展示信息（retrieval_type / storage_backend）在定义上，
        # 绑定行只存 id。一次批量取回来，避免 N+1。
        definitions = {
            definition.retrieval_definition_id: definition
            for definition in (
                await self._retrieval_repository.list_by_ids(
                    [
                        b.document_retrieval_definition_id
                        for b in bindings
                    ]
                )
            )
        }

        items: list[DocumentVersionRetrievalSummary] = []

        for binding in bindings:

            definition = definitions.get(
                binding.document_retrieval_definition_id
            )

            # 定义被物理删掉时会取不到。跳过是错的（前端会以为这一路
            # 不存在），但也不能瞎编一个 retrieval_type —— 记为 unknown
            # 又会让前端去做类型分支。
            #
            # 选择：跳过并在 chunk_count 上保持沉默做不到，所以这里
            # 直接跳过。正常流程下定义不会物理删除（逻辑删除仍能取到），
            # 所以这条分支只在数据被手工清理后出现。
            if definition is None:
                continue

            # chunk 数按 (版本, 该路的 chunker 身份) 查。
            #
            # ⚠️ 共用同一个 chunker 的两路会报出**相同**的 chunk_count ——
            # 它们本来就用同一份 chunk，这不是重复计数，是事实。
            identity = chunker_identity(
                self._pipeline_registry,
                definition.chunker_name,
                definition.chunker_config,
            )

            _, chunk_count = await self._chunk_repository.list(
                document_version_id=document_version_id,
                chunker_key=identity.chunker_key,
                chunker_config=identity.config,
                limit=1,
            )

            items.append(
                DocumentVersionRetrievalSummary(
                    document_version_retrieval_binding_id=(
                        binding.document_version_retrieval_binding_id
                    ),
                    document_retrieval_definition_id=(
                        binding.document_retrieval_definition_id
                    ),
                    retrieval_type=definition.retrieval_type,
                    storage_backend=definition.storage_backend,
                    # 运行时冻结值目前没有记录 —— 见 ChunkerBinding 的说明。
                    # 这里显式给 None，而不是回退去读 definition.chunker_name：
                    # 那会把"今天的配置"冒充成"当时用的配置"。
                    chunker=None,
                    chunk_count=chunk_count,
                    processing_status=binding.processing_status,
                    error_message=binding.error_message,
                )
            )

        return DocumentVersionRetrievalListResponse(
            items=items,
            total=len(items),
        )

    # ========================================================
    # 某个 tab 的内容
    # ========================================================

    async def list_chunks(
            self,
            document_version_id: UUID,
            document_retrieval_definition_id: UUID,
            *,
            preview_chars: int | None = None,
            page: int = 1,
            page_size: int = 20,
    ) -> DocumentChunkListResponse:

        await self._require_version(document_version_id)

        # 该 retrieval 必须**参与过这个版本**，否则 404。
        #
        # 不放过这一条检查的话，一个存在于类型定义、但这个版本没跑的
        # retrieval 会返回空列表 —— 前端分不清"没跑"和"跑了但没切出东西"。
        binding = (
            await self._binding_repository
            .get_by_version_and_retrieval(
                document_version_id,
                document_retrieval_definition_id,
            )
        )

        if binding is None:
            raise NotFoundError(
                "This retrieval did not participate in the document "
                f"version: version={document_version_id} "
                f"retrieval={document_retrieval_definition_id}"
            )

        # 这一路用哪份 chunk 由它的 chunker 身份决定，不再由 retrieval
        # 决定 —— 见 DocumentChunk 的说明。
        definition = await self._retrieval_repository.get(
            document_retrieval_definition_id
        )

        if definition is None:
            raise NotFoundError(
                "DocumentRetrievalDefinition not found: "
                f"{document_retrieval_definition_id}"
            )

        identity = chunker_identity(
            self._pipeline_registry,
            definition.chunker_name,
            definition.chunker_config,
        )

        chunks, total = await self._chunk_repository.list(
            document_version_id=document_version_id,
            chunker_key=identity.chunker_key,
            chunker_config=identity.config,
            offset=(page - 1) * page_size,
            limit=page_size,
        )

        return DocumentChunkListResponse(
            items=[
                self._to_summary(chunk, preview_chars)
                for chunk in chunks
            ],
            page=page,
            page_size=page_size,
            total=total,
        )

    # ========================================================
    # 投影
    # ========================================================

    async def get_projections(
            self,
            document_chunk_ids: list[UUID],
            *,
            document_retrieval_definition_id: UUID | None = None,
            include_vector: bool = False,
    ) -> ChunkProjectionListResponse:
        """
        一条 chunk 会有**多路**投影 —— chunk 是版本级的事实，不归任何
        一路 retrieval（见 DocumentChunk 的说明），所以"它的投影"本来
        就是一组而不是一个。

        外层仍是每条 chunk 一项、与请求一一对应且同序；里面多了一个
        projections 列表。
        """

        # 先定位每条 chunk —— 版本与 chunker 身份都在它身上。
        located: dict[UUID, DocumentChunk | None] = {}

        for chunk_id in document_chunk_ids:
            located[chunk_id] = await self._chunk_repository.get(
                chunk_id
            )

        # 版本 -> 用到该 chunk 的 retrieval 定义（由 chunker 身份推导）
        users = await self._retrieval_users(
            located, document_retrieval_definition_id
        )

        # 每个 (版本, retrieval) 的产物只读一次，建 chunk_id -> item 索引。
        projections: dict[tuple[UUID, UUID], dict[UUID, dict]] = {}
        reasons: dict[tuple[UUID, UUID], str | None] = {}
        metas: dict[tuple[UUID, UUID], dict[str, Any]] = {}

        for version_id, definitions in users.items():

            for definition in definitions:

                key = (
                    version_id,
                    definition.retrieval_definition_id,
                )

                if key in projections:
                    continue

                meta, loaded, reason = (
                    await self._load_projection_items(*key)
                )

                metas[key] = meta
                reasons[key] = reason
                projections[key] = {
                    item_id: item
                    for item in loaded
                    if (
                        item_id := _as_uuid(
                            item.get("document_chunk_id")
                        )
                    ) is not None
                }

        items: list[ChunkProjection] = []

        for chunk_id in document_chunk_ids:

            chunk = located.get(chunk_id)

            if chunk is None:
                items.append(_missing(
                    chunk_id, "找不到这条 chunk（document_chunk 里没有）。"
                ))
                continue

            entries: list[ChunkProjectionEntry] = []

            for definition in users.get(
                    chunk.document_version_id, []
            ):

                key = (
                    chunk.document_version_id,
                    definition.retrieval_definition_id,
                )

                item = projections.get(key, {}).get(chunk_id)

                if item is None:
                    entries.append(_entry_missing(
                        definition,
                        await self._why_no_projection(
                            definition, reasons.get(key)
                        ),
                    ))
                    continue

                entries.append(
                    _to_entry(
                        definition,
                        item,
                        metas.get(key) or {},
                        include_vector,
                    )
                )

            items.append(
                ChunkProjection(
                    document_chunk_id=chunk_id,
                    document_version_id=(
                        chunk.document_version_id
                    ),
                    chunker_key=chunk.chunker_key,
                    projections=entries,
                )
            )

        return ChunkProjectionListResponse(
            items=items,
            total=len(items),
            include_vector=include_vector,
        )

    async def _retrieval_users(
            self,
            located: dict[UUID, DocumentChunk | None],
            only_retrieval_id: UUID | None,
    ) -> dict[UUID, list]:
        """
        版本 -> 用到该版本 chunk 的 retrieval 定义。

        推导依据：检索定义里的 (chunker_name, chunker_config) 归一化后
        与 chunk 行上的 (chunker_key, chunker_config) 相等。

        为什么不查绑定表：chunk 与 retrieval 的关系**不落库**，存一份
        必然与 chunk 行漂移，而且"谁为准"没有答案。见 DocumentChunk
        的说明。

        只对请求里真的出现了 chunk 的版本算 —— 其余版本不必白跑。
        """

        version_ids = {
            chunk.document_version_id
            for chunk in located.values()
            if chunk is not None
        }

        result: dict[UUID, list] = {}

        for version_id in version_ids:

            # 该版本下请求实际涉及的那几份切分产出（可能不止一份）
            wanted = {
                identity_digest(
                    chunk.chunker_key, chunk.chunker_config
                )
                for chunk in located.values()
                if chunk is not None
                and chunk.document_version_id == version_id
            }

            bindings = (
                await self._binding_repository.list_by_version(
                    version_id
                )
            )

            if not bindings:
                result[version_id] = []
                continue

            definitions = {
                definition.retrieval_definition_id: definition
                for definition in (
                    await self._retrieval_repository.list_by_ids(
                        [
                            b.document_retrieval_definition_id
                            for b in bindings
                        ]
                    )
                )
            }

            users = []

            for binding in bindings:

                definition = definitions.get(
                    binding.document_retrieval_definition_id
                )

                if definition is None:
                    continue

                if (
                    only_retrieval_id is not None
                    and definition.retrieval_definition_id
                    != only_retrieval_id
                ):
                    continue

                identity = chunker_identity(
                    self._pipeline_registry,
                    definition.chunker_name,
                    definition.chunker_config,
                )

                if identity.digest in wanted:
                    users.append(definition)

            result[version_id] = users

        return result

    async def _load_projection_items(
            self,
            document_version_id: UUID,
            document_retrieval_definition_id: UUID,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
        """
        Returns:
            (产物外层信息, 条目列表, 取不到时的原因)。

        ⚠️ kind / embedder / dimensions 在产物的**外层**，不在每条 item 里：

            {"kind": "dense", "embedder": "...", "dimensions": 512,
             "items": [{**chunk 字段, "vector": [...]}, ...]}

        所以它们必须连同 items 一起返回 —— 从 item 上读永远读不到。
        条目为空时列表为空、原因非空，好让调用方给出**具体**的提示。
        """
        """
        取出某一路的投影条目。

        链路：版本的最近一次运行 -> 该运行里 embedding 节点在本分支上的
        节点行 -> 它的 output_summary.ref.file_id -> 读产物。

        为什么绕这么多：投影**不落库**（indexing 节点现算现发），
        唯一的痕迹就是 embedding 节点写出的那份产物，而产物与
        (版本, retrieval) 的关联只存在于节点行的 output_summary 里。
        """

        runs, _ = await self._run_repository.list_runs(
            document_version_id=document_version_id,
            limit=1,
        )

        if not runs:
            return {}, [], (
                "该版本还没有任何流水线运行记录 —— 它可能还在排队。"
            )

        nodes = await self._run_repository.list_all_node_runs(
            runs[0].pipeline_run_id
        )

        file_id: str | None = None
        saw_branch = False

        for node in nodes:

            if node.node_id != "embedding":
                continue

            if (
                node.document_retrieval_definition_id
                != document_retrieval_definition_id
            ):
                continue

            saw_branch = True

            # file_id 可能在两处：
            #   vector_ref.file_id   阶段 9 用的形状
            #   ref.file_id          阶段 8 等节点用的形状
            # 两种都认，免得只因为换了个键名就读不到产物。
            summary = node.output_summary or {}
            vector_ref = summary.get("vector_ref") or {}
            ref = summary.get("ref") or {}

            file_id = vector_ref.get("file_id") or ref.get("file_id")

            if file_id:
                break

        if not saw_branch:
            return {}, [], (
                "这次运行的节点行里没有本分支的 embedding 记录 —— "
                "可能该分支在上游就失败了。"
            )

        if not file_id:
            # 历史数据会走到这里：早期版本的 embedding 节点只把 checksum
            # 写进了 output_summary，没有 file_id，凭它找不回产物。
            # 重跑一次（reprocess）就会带上 —— 见 embedding.py 的说明。
            return {}, [], (
                "该次运行的 embedding 节点行没有记 file_id（早期版本的"
                "遗漏），凭 checksum 找不回产物。reprocess 一次即可。"
            )

        try:
            raw = await self._artifact_store.get_text(UUID(file_id))
        except (ValueError, TypeError):
            return {}, [], f"产物引用不是合法的 file_id: {file_id!r}"

        if not raw:
            return {}, [], (
                f"产物读不到内容（file_id={file_id}）—— "
                "可能已被清理。"
            )

        payload = json.loads(raw)

        # 向量产物是 {"kind":..., "items":[...]}；不是这个形状说明
        # 那一路没有投影（embedding 透传时产物是上游的 chunk 数组）。
        if not isinstance(payload, dict):
            return {}, [], (
                "该分支的上游产物不是向量产物 —— "
                "这一路在 embedding 阶段是透传的，没有单独的投影。"
            )

        # 外层三项是产物级的元信息，供每条 item 复用（见上面的说明）。
        meta = {
            "kind": payload.get("kind"),
            "embedder": payload.get("embedder"),
            "dimensions": payload.get("dimensions"),
        }

        return meta, list(payload.get("items") or []), None

    async def _why_no_projection(
            self,
            definition,
            fallback: str | None,
    ) -> str:
        """
        取不到投影时，尽量给出**具体**的原因，而不是一句"没有了"。

        优先级：这一路**本来就不需要**向量化 > 产物层给出的具体原因 >
        泛泛的兜底。

        第一条要排在最前 —— 对 keyword / document 那两路，"取不到"
        是正常的（它们在阶段 9 是透传的），而不是某种故障。

        入参从 chunk 改成了 retrieval 定义：chunk 上已经没有 retrieval
        了，"这一路需不需要向量"只有定义知道。
        """

        needs = definition.retrieval_type.value in (
            "dense_vector", "sparse_vector", "graph",
        )

        if not needs:
            return (
                f"retrieval_type="
                f"{definition.retrieval_type.value!r} 不需要向量化，"
                "embedding 阶段是透传的，因此没有单独的投影产物 —— "
                "这一路写给索引的就是 chunk 本身。"
            )

        if fallback:
            return fallback

        # 产物读到了，但里面没有这条 chunk —— 说明它是在那次运行之后
        # 才产生的（例如 reprocess 换了 chunker），或者 id 对不上。
        return (
            "该次的投影产物里没有这条 chunk —— "
            "它可能是在那次运行之后才产生的（例如换过 chunker 后 reprocess）。"
        )

    # ========================================================
    # helpers
    # ========================================================

    async def _require_version(
            self,
            document_version_id: UUID,
    ) -> None:
        if await self._version_repository.get(document_version_id) is None:
            raise NotFoundError(
                f"DocumentVersion not found: {document_version_id}"
            )

    @staticmethod
    def _to_summary(
            chunk: DocumentChunk,
            preview_chars: int | None,
    ) -> DocumentChunkSummary:
        """
        实体 -> 对外契约。

        preview_chars 只截 content，**不动 char_count** ——
        后者是截断前的真实长度，前端要靠它判断"这条到底有多大"。
        如果把 char_count 也改成截断后的长度，前端就看不出还有多少没展示。
        """

        content = chunk.content
        truncated = False

        if preview_chars is not None and len(content) > preview_chars:
            content = content[:preview_chars]
            truncated = True

        return DocumentChunkSummary(
            document_chunk_id=chunk.document_chunk_id,
            chunk_index=chunk.chunk_index,
            content=content,
            content_truncated=truncated,
            char_count=chunk.char_count,
            token_count=chunk.token_count,
            start_index=chunk.start_index,
            end_index=chunk.end_index,
            content_hash=chunk.content_hash,
            locator=chunk.locator,
            metadata=dict(chunk.metadata or {}),
            created_at=chunk.created_at,
        )


# ============================================================
# 投影的组装（模块级 —— 不依赖实例状态）
# ============================================================


def _as_uuid(value: Any) -> UUID | None:
    """产物里的 id 是字符串；转不动就返回 None，不抛。"""

    if value is None:
        return None

    try:
        return UUID(str(value))
    except (ValueError, TypeError):
        return None


def _missing(
        chunk_id: UUID,
        reason: str,
) -> ChunkProjection:
    """chunk 本身就取不到时的占位 —— 没有 chunk 就没有版本与投影。"""

    return ChunkProjection(
        document_chunk_id=chunk_id,
        unavailable_reason=reason,
    )


def _entry_missing(
        definition,
        reason: str,
) -> ChunkProjectionEntry:
    """某一路 retrieval 取不到投影时的占位 —— 其余路照常返回。"""

    return ChunkProjectionEntry(
        document_retrieval_definition_id=(
            definition.retrieval_definition_id
        ),
        retrieval_type=definition.retrieval_type,
        unavailable_reason=reason,
    )


def _to_entry(
        definition,
        item: dict[str, Any],
        meta: dict[str, Any],
        include_vector: bool,
) -> ChunkProjectionEntry:
    """
    产物里的一条 item -> **某一路 retrieval 的**投影。

    item 的形状是 {**chunk 字段, vector|sparse|graph}（见 embedding.py）：
    非向量的那些键就是"当时发给索引的字段"，在这里归到 fields。
    """

    found = None
    for key, kind in _VECTOR_KEYS.items():
        if key in item:
            found = (key, kind)
            break

    if found is None:
        # 产物里有这条 chunk，但没有可识别的向量键 —— graph 目前
        # 走不到这里（它的抽取还没实现），其余情况说明产物形状变了。
        return ChunkProjectionEntry(
            document_retrieval_definition_id=(
                definition.retrieval_definition_id
            ),
            retrieval_type=definition.retrieval_type,
            fields=_fields_of(item),
            unavailable_reason=(
                "产物里有这条 chunk，但没有可识别的向量字段。"
            ),
        )

    key, kind = found

    entry = ChunkProjectionEntry(
        document_retrieval_definition_id=(
            definition.retrieval_definition_id
        ),
        retrieval_type=definition.retrieval_type,
        kind=kind,
        # ⚠️ 从**产物外层**读，不是从 item：这两项是产物级的元信息。
        embedder=meta.get("embedder"),
        dimensions=meta.get("dimensions"),
        fields=_fields_of(item),
    )

    if kind is EmbeddingOutput.DENSE:
        vector = list(item.get(key) or [])
        entry.vector_summary = _vector_summary(vector)
        if include_vector:
            entry.vector = vector

    elif kind is EmbeddingOutput.SPARSE:
        # 产物的稀疏向量键是字符串（JSON 限制），转回 int。
        sparse = {
            int(dim): float(weight)
            for dim, weight in (item.get(key) or {}).items()
        }
        entry.sparse_summary = _sparse_summary(sparse)
        if include_vector:
            entry.sparse = sparse

    return entry


def _fields_of(item: dict[str, Any]) -> dict[str, Any]:
    """投影里非向量的那些字段 —— 就是当时发给索引的标量/正文/元数据。"""

    return {
        key: value
        for key, value in item.items()
        if key not in _VECTOR_FIELD_KEYS
    }


def _vector_summary(vector: list[float]) -> VectorSummary:

    nonzero = sum(1 for value in vector if value != 0.0)

    return VectorSummary(
        dimensions=len(vector),
        nonzero_count=nonzero,
        l2_norm=math.sqrt(sum(value * value for value in vector)),
        head=[float(value) for value in vector[:_SUMMARY_HEAD]],
    )


def _sparse_summary(sparse: dict[int, float]) -> SparseSummary:

    top = sorted(
        sparse.items(),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )[:_SUMMARY_HEAD]

    return SparseSummary(
        nonzero_count=len(sparse),
        top=[
            {"dim": dim, "weight": weight}
            for dim, weight in top
        ],
    )
