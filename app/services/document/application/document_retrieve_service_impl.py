"""
DocumentRetrieveService 的默认实现。

事务边界由调用方（dependencies）持有的 session 决定，
本服务内部不 commit / rollback。

一轮检索的完整顺序
------------------
    1. 校验 DTD（全部存在且 active，否则整体 404）
    2. 取每个 DTD 的 active RetrievalDefinition
    3. 逐路**内联召回**（IndexBackend.search，不经 Recaller 组件）
    4. 按路归一化分数（见 retrieve/base.py 的不变量）
    5. 回表补齐正文与归属（PostgreSQL 是它们的唯一事实来源）
    6. 装配 RetrieveProcessor 并跑 融合 -> 过滤 -> 精排 -> 裁剪
    7. 回显实际生效的 RetrieveProcessor

为什么召回内联而不是做成 Recaller 组件
--------------------------------------
规格明确 Recaller 是内部细节、不通过公开 API 暴露；调用方也不该
指定"用哪个召回器"。做成可配置组件只会多一层没有调用方的抽象。
候选集的口径由 DTD 的 RetrievalDefinition 决定，这就是它的配置面。

为什么先回表再跑管线
--------------------
过滤要看归属（FilterConfig.document_ids / document_version_ids），
精排要看正文 —— 两者都只有 PostgreSQL 有（索引召回只回
chunk_id / score / retrieval_source / metadata，见 index/es.py）。
因此回表必须排在这两步之前，否则过滤会永远匹配不上、精排拿到空正文。
代价是回表量按融合前的候选数算，由各路的 top_k 决定。
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID, uuid4

from app.common.exceptions import NotFoundError

from app.common.service.base import ServiceContext
from app.services.tool.registry import tool

from ..domain.entities import (
    DocumentRetrievalDefinition,
    DocumentTypeDefinition,
)
from ..domain.schemas import (
    DegradedRoute,
    DocumentRetrieveRequest,
    DocumentRetrieveResponse,
    DocumentRetrieveResult,
    IndexSearchRequest,
    RetrievalRecallConfig,
)
from ..domain.schemas.enum import (
    DefinitionStatus,
    EmbeddingOutput,
    RetrievalType,
    RetrieveProcessorMode,
    RetrieveScope,
)
from ..pipeline.containers import external_container
from ..pipeline.registry import CapabilityNotFound, PipelineRegistry
from ..retrieve import RetrieveCandidate, RetrieveProcessor, normalize_scores
from ..repository import (
    DocumentChunkRepository,
    DocumentRepository,
    DocumentRetrievalDefinitionRepository,
    DocumentTypeDefinitionRepository,
    DocumentVersionRepository,
)
from .document_retrieve_service import DocumentRetrieveService

__all__ = ["DocumentRetrieveServiceImpl"]

logger = logging.getLogger(__name__)


def _degraded(
        retrieval: DocumentRetrievalDefinition,
        reason: str,
) -> DegradedRoute:
    """把"这一路没参与"记成响应里的一条，而不是只留在日志里。"""

    return DegradedRoute(
        document_retrieval_definition_id=(
            retrieval.retrieval_definition_id
        ),
        retrieval_type=retrieval.retrieval_type,
        reason=reason,
    )

#: 单路召回的默认条数。
#:
#: ⚠️ 这个值决定的是**候选池**大小，不是最终返回条数（那是 cutter.top_k）。
#: 两者之间必须有足够的缓冲：池子太窄时，语义高度相关但排序靠后的 chunk
#: **根本进不来**，后面的融合 / 精排 / 裁剪再准也救不回来。
#:
#: 实测依据：一份 559 条事件的档案被切成 259 个 chunk（同一个版本），
#: 而目标 chunk 在关键词那一路排第 61 —— 原来的 50 把它直接截断了，
#: 表现是"响应里怎么也找不到这条，但它确实在索引里"。
#:
#: retriever_config 里若给了合法的 recall_top_k 就用它 —— 但注意该
#: 字段当前被写入期复用去装 embedder 配置（见 embedding.py 的说明），
#: 因此不优先采信，只在类型与取值都合理时采用。
_DEFAULT_RECALL_TOP_K = 200

#: 单路召回条数上限，与 IndexSearchRequest.top_k 的 le=500 对齐。
_MAX_RECALL_TOP_K = 500

#: retrieval_type -> 召回它需要哪种形态的**查询**向量。
#:
#: 与写入期的 pipeline/nodes/embedding.py::_REQUIRED_OUTPUT 是同一个
#: 对应关系 —— 那边决定"写入时要产出什么向量"，这边决定"召回时要拿
#: 什么向量去查"。两边必须一致：不一致时维度可能照样对得上、检索
#: 照样返回结果，只是排序没有意义。
#:
#: 两种形态落到请求里**不同的字段**：
#:
#:     DENSE   -> query_vector   （list[float]）
#:     SPARSE  -> sparse_query   （dict[int, float]）
#:
#: keyword / document / graph 不需要查询表示，不在表里。
_QUERY_VECTOR_KIND: dict[str, EmbeddingOutput] = {
    "dense_vector": EmbeddingOutput.DENSE,
    "sparse_vector": EmbeddingOutput.SPARSE,
}


class DocumentRetrieveServiceImpl(DocumentRetrieveService):

    def __init__(
            self,
            *,
            type_definition_repository: (
                DocumentTypeDefinitionRepository
            ),
            retrieval_definition_repository: (
                DocumentRetrievalDefinitionRepository
            ),
            chunk_repository: DocumentChunkRepository,
            version_repository: DocumentVersionRepository,
            document_repository: DocumentRepository,
            pipeline_registry: PipelineRegistry,
            ctx: ServiceContext | None = None,
    ) -> None:

        super().__init__(ctx)

        self._types = type_definition_repository
        self._retrievals = retrieval_definition_repository
        self._chunks = chunk_repository
        self._versions = version_repository
        self._documents = document_repository
        self._pipeline_registry = pipeline_registry

    # ========================================================
    # 入口
    # ========================================================
    @tool
    async def retrieve_documents(
            self,
            request: DocumentRetrieveRequest,
    ) -> DocumentRetrieveResponse:
        """
        执行一轮检索。

        Args:
            request:
                目标 DTD、查询文本，以及可选的 RetrieveProcessor 覆盖。

        Returns:
            DocumentRetrieveResponse:
                本轮结果 + 实际生效的 RetrieveProcessor 回显。
                没有任何召回命中时 results 为空列表，**不是**错误。

        Raises:
            NotFoundError:
                任一 document_type_definition_id 不存在或已删除。
                整体失败而不是跳过它 —— 静默少查一个类型会让调用方
                把"这个类型没数据"和"这个类型没查"混为一谈。
            ValueError:
                DTD 上的 fusion / rerank 配置非法（例如 top_k 越界）。
        """

        definitions = await self._load_definitions(
            request.document_type_definition_ids
        )

        query = request.input_text

        # 「只检索当前生效版本」的版本集合。一次算好，所有 DTD、所有路
        # 共用 —— 它只取决于请求里的 DTD，与具体哪一路无关。
        #
        # 为 None 表示不过滤（only_active=False）。
        active_version_ids = await self._active_versions(
            request, definitions
        )

        # 集合为空 = 这些 DTD 下一个生效版本都没有，那什么也召不回来。
        # 短路放在 _recall_one 里（每条路各自直接返回空），而不是在这里
        # 提前拼一个空响应 —— 那会把响应组装逻辑复制成两份。
        #
        # 也不能把空集合传下去：它表达不出"什么都不要"，索引后端要么
        # 语法非法，要么静默退化成"不过滤"，那就会返回不该返回的结果。

        # 没能参与本轮的路。多路召回里一路的后端故障不该让整个查询失败，
        # 但也**不能静默**跳过 —— 调用方必须能分清"没召回到"和"后端挂了"。
        degraded: list[DegradedRoute] = []

        # 定向召回：只保留调用方点名的那几路。None = 不筛。
        #
        # 交集是**自然**得到的：下面只遍历该 DTD 实际配了的 retrieval
        # 定义，所以"点名了一个文档里没有的类型"不会凭空召出东西 ——
        # 它只是筛不掉任何一条。不需要额外求交集的代码。
        wanted_types = (
            set(request.retrieval_types)
            if request.retrieval_types
            else None
        )

        # {document_type_definition_id: [每路一个列表]}
        per_type_sets: dict[UUID, list[list[RetrieveCandidate]]] = {}

        for definition in definitions:

            per_type_sets[
                definition.document_type_definition_id
            ] = await self._recall(
                definition,
                query,
                active_version_ids,
                degraded,
                wanted_types,
                request.retrieval_config,
            )

        await self._hydrate(
            [
                result_set
                for sets in per_type_sets.values()
                for result_set in sets
            ]
        )

        override = request.retrieve_processor

        if len(definitions) == 1:

            definition = definitions[0]

            processor = RetrieveProcessor.for_document_type(
                definition, override=override
            )

            candidates = await processor.run(
                query=query,
                result_sets=per_type_sets[
                    definition.document_type_definition_id
                ],
            )

            mode = RetrieveProcessorMode.DTD

        else:

            # 每个 DTD 先各跑自己的 RetrieveProcessor，各产出一路
            combined: list[list[RetrieveCandidate]] = []

            for definition in definitions:

                per_type_processor = (
                    RetrieveProcessor.for_document_type(
                        definition, override=override
                    )
                )

                combined.append(
                    await per_type_processor.run(
                        query=query,
                        result_sets=per_type_sets[
                            definition
                            .document_type_definition_id
                        ],
                    )
                )

            # 再过公共链：把"每个 DTD 一路"当成多路输入，
            # 由 COMMON_RETRIEVE_PROCESSOR_CHAIN 做跨类型融合。
            processor = RetrieveProcessor.common_chain(
                override=override
            )

            candidates = await processor.run(
                query=query,
                result_sets=combined,
            )

            mode = RetrieveProcessorMode.COMMON_CHAIN

        return DocumentRetrieveResponse(
            retrieve_id=uuid4(),
            retrieve_scope=(
                RetrieveScope.SINGLE
                if mode is RetrieveProcessorMode.DTD
                else RetrieveScope.MULTI
            ),
            retrieve_processor_mode=mode,
            document_type_definition_ids=list(
                request.document_type_definition_ids
            ),
            results=self._build_results(candidates),
            degraded_routes=degraded,
            retrieve_processor=processor.effective(),
        )

    # ========================================================
    # 校验
    # ========================================================

    async def _load_definitions(
            self,
            document_type_definition_ids: list[UUID],
    ) -> list[DocumentTypeDefinition]:
        """
        一次取齐并校验。

        任一不存在或已删除即整体失败 —— 静默跳过会让调用方把
        "这个类型没数据"和"这个类型根本没查"混为一谈。
        """

        found = await self._types.get_many(
            document_type_definition_ids
        )

        by_id = {
            definition.document_type_definition_id: definition
            for definition in found
        }

        definitions: list[DocumentTypeDefinition] = []

        for definition_id in document_type_definition_ids:

            definition = by_id.get(definition_id)

            if definition is None:
                raise NotFoundError(
                    "DocumentTypeDefinition not found: "
                    f"{definition_id}"
                )

            if definition.status is not DefinitionStatus.ACTIVE:
                raise NotFoundError(
                    "DocumentTypeDefinition is not active: "
                    f"{definition_id}"
                )

            definitions.append(definition)

        return definitions

    # ========================================================
    # 召回
    # ========================================================

    async def _active_versions(
            self,
            request: DocumentRetrieveRequest,
            definitions: list[DocumentTypeDefinition],
    ) -> list[UUID] | None:
        """
        算出"只检索当前生效版本"要用的版本集合。

        返回 None 表示**不过滤**（调用方明确要了历史版本）。
        返回空列表表示**这些 DTD 下没有生效版本** —— 调用方据此
        直接短路，不要把它当成"不过滤"传给索引后端。
        """

        if not request.only_active:
            return None

        return (
            await self._versions.list_active_ids_by_type_definitions(
                [
                    definition.document_type_definition_id
                    for definition in definitions
                ]
            )
        )

    async def _recall(
            self,
            definition: DocumentTypeDefinition,
            query: str,
            active_version_ids: list[UUID] | None,
            degraded: list[DegradedRoute],
            wanted_types: set[RetrievalType] | None = None,
            recall_config: dict[
                RetrievalType,
                RetrievalRecallConfig,
            ] | None = None,
    ) -> list[list[RetrieveCandidate]]:
        """
        对一个 DTD 逐路召回，返回"每路一个列表"。

        任何一路拿不到结果都**只影响那一路**：
        后端没启用、向量召回尚未实现、索引不存在 —— 都是
        "这一路这次没有贡献"，不是整个查询失败。这与写入期的
        立场一致（ES 挂了但 Milvus 成了，文档在那部分索引里
        是真实可检索的）。
        """

        retrieval_definitions = (
            await self._retrievals.list_all_by_type_definition(
                definition.document_type_definition_id,
                include_deleted=False,
            )
        )

        result_sets: list[list[RetrieveCandidate]] = []

        for retrieval in retrieval_definitions:

            if retrieval.status is not DefinitionStatus.ACTIVE:
                continue

            # 定向召回：点名的类型之外一律不参与。
            #
            # 注意这里**不报 warning** —— 被筛掉是调用方自己要求的，
            # 不是"本该跑却没跑"（那种才会进 degraded_routes）。
            # 也因此"点名了一个文档里没有的类型"不会留下任何痕迹：
            # 交集就是空的，符合调用方的要求。
            if (
                wanted_types is not None
                and retrieval.retrieval_type not in wanted_types
            ):
                continue

            # 该路的请求级参数。没给这一路就是 None —— 各字段再各自
            # 回落到默认（见 _recall_top_k 等）。
            recalled = await self._recall_one(
                retrieval,
                query,
                active_version_ids,
                degraded,
                (recall_config or {}).get(retrieval.retrieval_type),
            )

            if recalled:
                result_sets.append(recalled)

        return result_sets

    async def _recall_one(
            self,
            retrieval: DocumentRetrievalDefinition,
            query: str,
            active_version_ids: list[UUID] | None,
            degraded: list[DegradedRoute],
            recall_config: RetrievalRecallConfig | None = None,
    ) -> list[RetrieveCandidate]:

        if active_version_ids == []:
            # 这些 DTD 下没有生效版本 —— 什么都召不回来。
            # 不把空集合传给后端：它表达不出"什么都不要"。
            return []

        retrieval_type = retrieval.retrieval_type.value

        container = external_container(retrieval_type)

        if container is None:
            # 当前只有 document（PostgreSQL）落在这里：它的 chunk
            # 就在 document_chunk 表里，没有外部索引可查。走它需要
            # 一条按 keyword 查 chunk 表的召回路径，目前没实现。
            logger.warning(
                "retrieval_type=%s 没有外部索引容器，本轮跳过。",
                retrieval_type,
            )
            return []

        try:
            backend = self._pipeline_registry.index_backend(
                retrieval.storage_backend
            )

        except CapabilityNotFound:
            logger.warning(
                "IndexBackend %s 未注册（数据源未启用或连不上），"
                "本轮跳过 retrieval_type=%s。",
                retrieval.storage_backend,
                retrieval_type,
            )
            return []

        # 查询表示要在把请求交给索引后端**之前**算好 ——
        # 索引后端不持有 embedder（理由见 IndexSearchRequest 的说明）。
        vector_kind = _QUERY_VECTOR_KIND.get(retrieval_type)

        query_vector = None
        sparse_query = None

        if vector_kind is not None:

            embedded = await self._query_vector(
                retrieval, vector_kind, query
            )

            if embedded is None:
                # 需要查询表示却拿不到（该形态没有已注册的 embedder）。
                # _query_vector 已经留痕，这里只是不再往下走 ——
                # 把 None 传给后端只会换来一个看不懂的报错。
                return []

            # 两种形态落到不同字段，由形状决定（见 _QUERY_VECTOR_KIND）。
            if vector_kind is EmbeddingOutput.SPARSE:
                sparse_query = embedded
            else:
                query_vector = embedded

        timeout = (
            recall_config.timeout_ms / 1000
            if recall_config is not None
            and recall_config.timeout_ms is not None
            else None
        )

        try:
            response = await asyncio.wait_for(
                backend.search(
                    IndexSearchRequest(
                        query=query,
                        query_vector=query_vector,
                        sparse_query=sparse_query,
                        # None = 不过滤（only_active=False）；非空集合 =
                        # 只在这些生效版本里召回。空集合到不了这里。
                        document_version_ids=active_version_ids,
                        filters={},
                        top_k=self._recall_top_k(
                            retrieval,
                            recall_config,
                        ),
                        # 原始分下限，由后端下推到索引层。**必须**在
                        # 归一化之前生效 —— 下面 return 前的
                        # normalize_scores 会把头名压成恒等于 1.0，
                        # 到那时再比阈值就永远滤不掉东西。
                        min_score=(
                            recall_config.min_score
                            if recall_config is not None
                            else None
                        ),
                        search_params=(
                            dict(recall_config.search_params)
                            if recall_config is not None
                            else {}
                        ),
                        index_identifier=container,
                    )
                ),
                # None = 不设超时，交给后端自己的（httpx / pymilvus）。
                timeout=timeout,
            )

        except TimeoutError:
            # 单路超时 —— 与后端报错同样处理：只跳过这一路。
            #
            # ⚠️ 必须排在下面那个 except Exception **前面**：
            # TimeoutError 是 OSError 的子类，会被它吞掉，那时 reason
            # 里只剩一句 "TimeoutError"，分不出是超时还是别的 IO 错。
            logger.warning(
                "retrieval_type=%s 的召回超时（%d ms），本轮跳过该路。",
                retrieval_type,
                recall_config.timeout_ms,
            )
            degraded.append(
                _degraded(
                    retrieval,
                    f"召回超时（超过 {recall_config.timeout_ms} ms）",
                )
            )
            return []

        except NotImplementedError as exc:
            # 后端明确表示"这一路我还没实现"。当前 ES 与 Milvus 都
            # 已接通（keyword / sparse 走 ES，dense 走 Milvus），所以
            # 这条分支留给将来新增的后端（如 graph / Neo4j）。
            #
            # 这是**已知缺口**而非本次查询的错误，因此只跳过这一路，
            # 不让整个请求失败。
            logger.warning(
                "retrieval_type=%s 的召回尚未接通，本轮跳过：%s",
                retrieval_type,
                exc,
            )
            degraded.append(_degraded(retrieval, str(exc)))
            return []

        except Exception as exc:
            # 后端调用失败（Milvus 在恢复、ES 超时、连接断了……）。
            #
            # 只跳过**这一路**：其余路召回的内容是真实可检索的，让整个
            # 查询连带失败没有道理 —— 这正是写入期对扇出分支的立场
            # （见 indexing.py：ES 挂了但 Milvus 成了，不该笼统报失败）。
            #
            # 但绝不静默：记 warning，并把这一路放进响应的 degraded_routes，
            # 让调用方能分清"没召回到"和"后端挂了"。
            logger.warning(
                "retrieval_type=%s 的召回失败，本轮跳过该路："
                "%s: %s",
                retrieval_type,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            degraded.append(_degraded(retrieval, str(exc)))
            return []

        candidates = [
            RetrieveCandidate(
                document_chunk_id=result.document_chunk_id,
                score=result.score,
                retrieval_source=result.retrieval_source,
                pipeline_key=retrieval_type,
                # 召回阶段每条候选只来自一路；融合时会被并成多路。
                # 在这里就填上，未走融合的路径（单路 / 融合关闭）才不会
                # 在响应里露出空列表。
                pipeline_keys=[retrieval_type],
                metadata=dict(result.metadata),
                # 索引侧给的标记片段，**按产生它的那一路**归位。
                # 召回阶段每条候选只来自一路，所以这里只有一个键；
                # 融合时才把多条并按路累加（见 merge_highlights）。
                # 空片段不入字典：键在字典里就意味着"这一路标出了东西"。
                highlights=(
                    {retrieval_type: list(result.highlights)}
                    if result.highlights
                    else {}
                ),
            )
            for result in response.results
        ]

        # 按路归一化：不同路的分数尺度不可比（BM25 几十、
        # 余弦 0..1），不归一化的话融合与 score_threshold 都失义。
        return normalize_scores(candidates)

    async def _query_vector(
            self,
            retrieval: DocumentRetrievalDefinition,
            kind: EmbeddingOutput,
            query: str,
    ) -> list[float] | dict[int, float] | None:
        """
        用**写入期同一个 embedder** 把查询文本向量化。

        选 embedder 的方式与写入期完全一致（都是
        registry.embedder_for(kind)，见 embedding.py），这不是巧合：
        查询向量必须落在与文档向量同一个空间里，换一个模型就不会
        报错，只会让召回率悄悄下降。

        配置也刻意与写入期同源 —— 都读 retrieval 定义的那份
        （embedding.py 的注释把这处"将就"写明了：retriever_config
        本该只放召回参数，但两边读同一处至少保证了模型与参数一致）。

        返回形状**随形态而变**，调用方据此决定放进哪个字段：

            DENSE   list[float]
            SPARSE  dict[int, float]（维度索引 -> 权重）

        刻意不在这里统一成 list：稀疏向量展开成定长稠密列表会是几十万
        个浮点（词表规模），内存与序列化都被拖垮，而且 ES 要的本来就是
        稀疏映射。返回 None 表示"该形态没有可用的 embedder"，调用方跳过。
        """

        embedder = self._pipeline_registry.embedder_for(kind)

        if embedder is None:
            logger.warning(
                "retrieval_type=%s 需要 %s 形态的查询表示，但该形态"
                "没有已注册的 embedder，本轮跳过。",
                retrieval.retrieval_type.value,
                kind.value,
            )
            return None

        embedded = await embedder.embed_query(
            query,
            config=dict(retrieval.retriever_config or {}),
        )

        if kind is EmbeddingOutput.SPARSE:
            # 键转回 int（JSON 往返会把它们变成字符串）—— 与
            # CanonicalChunk.sparse_vector 的约定一致。
            return {
                int(dim): float(weight)
                for dim, weight in embedded.items()
            }

        return list(embedded)

    @staticmethod
    def _recall_top_k(
            retrieval: DocumentRetrievalDefinition,
            recall_config: RetrievalRecallConfig | None = None,
    ) -> int:
        """
        单路召回条数。优先级从高到低：

            1. 请求级 retrieval_config[type].top_k   —— 本次调用说了算
            2. DTD 的 retriever_config["recall_top_k"] —— 该类型的默认
            3. _DEFAULT_RECALL_TOP_K（200）

        第 2 级是**历史遗留**：retriever_config 本该只放召回参数，但
        写入期复用了它去装 embedder 配置（见 embedding.py），所以那
        个键不优先采信，只在类型与取值都合理时采用。

        注意 bool 是 int 的子类 —— 不排掉的话
        {"recall_top_k": true} 会被当成 1。

        第 1 级不需要这类防御：顶层 schema 的 ge/le 已经卡过一遍，
        到这里必定是合法整数。仍然取 min 是防止它绕过上限 ——
        上限与 IndexSearchRequest.top_k 的 le=500 是同一个约束，
        只能在索引层之前兜住。
        """

        if recall_config is not None and recall_config.top_k is not None:
            return min(recall_config.top_k, _MAX_RECALL_TOP_K)

        value = (retrieval.retriever_config or {}).get(
            "recall_top_k"
        )

        if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
        ):
            return min(value, _MAX_RECALL_TOP_K)

        return _DEFAULT_RECALL_TOP_K

    # ========================================================
    # 回表
    # ========================================================

    async def _hydrate(
            self,
            result_sets: list[list[RetrieveCandidate]],
    ) -> None:
        """
        把正文与归属回填进候选（原地修改）。

        Chunk 只有 document_version_id，没有 document_id ——
        要拿 document_type_definition_id 必须经过
        version -> document 两跳。

        每跳都是一次 in_ 批量查询，不是逐个 get：候选数由各路
        top_k 决定，逐个查会变成几十次往返。
        """

        candidates = [
            candidate
            for result_set in result_sets
            for candidate in result_set
        ]

        if not candidates:
            return

        chunks = await self._chunks.list_by_ids(
            list(
                {
                    candidate.document_chunk_id
                    for candidate in candidates
                }
            )
        )

        if not chunks:
            return

        by_chunk = {
            chunk.document_chunk_id: chunk for chunk in chunks
        }

        versions = await self._versions.get_many(
            list({chunk.document_version_id for chunk in chunks})
        )

        by_version = {
            version.document_version_id: version
            for version in versions
        }

        documents = await self._documents.get_many(
            list({version.document_id for version in versions})
        )

        by_document = {
            document.document_id: document
            for document in documents
        }

        for candidate in candidates:

            chunk = by_chunk.get(candidate.document_chunk_id)

            if chunk is None:
                # 索引里有、库里没有：多半是版本已被清理而索引
                # 还没删干净。保留候选（分数与 metadata 仍然有效），
                # 由最终组装决定怎么呈现。
                logger.warning(
                    "召回结果在 document_chunk 中不存在：%s",
                    candidate.document_chunk_id,
                )
                continue

            candidate.content = chunk.content

            candidate.document_version_id = (
                chunk.document_version_id
            )

            # 召回侧的 metadata 与库里的合并，库里的是权威；
            # 但索引里可能有库里没有的字段（例如 page_number）。
            candidate.metadata = {
                **candidate.metadata,
                **dict(chunk.metadata),
            }

            version = by_version.get(chunk.document_version_id)

            if version is None:
                continue

            candidate.document_id = version.document_id

            document = by_document.get(version.document_id)

            if document is None:
                continue

            candidate.document_type_definition_id = (
                document.document_type_definition_id
            )

    # ========================================================
    # 组装
    # ========================================================

    def _build_results(
            self,
            candidates: list[RetrieveCandidate],
    ) -> list[DocumentRetrieveResult]:
        """
        组装最终结果。

        回表不完整的候选（正文或归属为空）**丢弃**：规格里
        document_id 等字段是 required，而调用方拿到一条没有正文的
        结果没有意义 —— 它既不能引用也不能送进 LLM。

        rank 按**剔除之后**的顺序重新编号：规格说 rank 从 1 开始，
        留空洞会让调用方以为中间漏了数据。
        """

        results: list[DocumentRetrieveResult] = []

        for candidate in candidates:

            if (
                    candidate.content is None
                    or candidate.document_id is None
                    or candidate.document_version_id is None
                    or candidate.document_type_definition_id is None
            ):

                logger.warning(
                    "候选 %s 回表不完整，已从结果中剔除。",
                    candidate.document_chunk_id,
                )

                continue

            results.append(
                DocumentRetrieveResult(
                    rank=len(results) + 1,
                    score=candidate.score,
                    document_chunk_id=(
                        candidate.document_chunk_id
                    ),
                    document_version_id=(
                        candidate.document_version_id
                    ),
                    document_id=candidate.document_id,
                    document_type_definition_id=(
                        candidate.document_type_definition_id
                    ),
                    # 融合之后 pipeline_key 只是代表值 —— 列全部路。
                    pipeline_keys=candidate.routes(),
                    highlights=dict(candidate.highlights),
                    content=candidate.content,
                    metadata=dict(candidate.metadata),
                )
            )

        return results
