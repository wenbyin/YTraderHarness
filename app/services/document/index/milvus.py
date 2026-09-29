"""
Milvus 索引后端。

服务哪些 retrieval
------------------
    dense_vector    稠密向量检索（FLOAT_VECTOR）

客户端是**同步**的
------------------
pymilvus 的 MilvusClient（3.0.1）只有同步接口，因此每个调用都包在
`asyncio.to_thread` 里 —— 直接调用会阻塞事件循环，而阶段 10 是**扇出**
的，N 个分支共用一个循环，一个分支卡住会把其余分支一起冻住。
（infra/datasource/milvus.py 的 health_check 用的是同一套办法。）

维度是索引的物理事实
--------------------
collection 的 schema 里 dim 一旦定下就不能改。所以：

    - 建 collection 前必须知道 dim，来源是 chunk 携带的 vector 长度
    - 同一个 collection 里**所有**向量的维度必须一致，不一致直接报错，
      而不是让 Milvus 抛一个难懂的 schema 错误

不要拿这个后端服务 sparse_vector
--------------------------------
Milvus 2.4+ 确实支持 SPARSE_FLOAT_VECTOR，但本项目的稀疏向量落在
**ES** 上（见 retrieval_type -> storage_backend 的默认映射）。
真要用 Milvus 存稀疏向量，那是另一个实现、另一个 backend_type，
不要在这里加分支 —— 一个后端同时处理两种 schema 会让建表逻辑
迅速失控。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import UUID

from pymilvus import LexicalHighlighter, MilvusClient

from app.services.document.domain.schemas import (
    CanonicalChunk,
    IndexDeleteRequest,
    IndexSearchRequest,
    IndexSearchResponse,
    IndexSearchResult,
    IndexUpsertRequest,
)
from app.services.document.domain.schemas.enum import StorageBackend

from .base import IndexBackend

__all__ = ["MilvusIndexBackend"]


#: collection 的标量字段。向量字段（vector）在建表时按 dim 动态加。
#:
#: max_length 是 Milvus 的硬要求：VARCHAR 必须给上限。给得宽松一点
#: （content 可能很长），但注意 Milvus 的 VARCHAR 上限就是 65535。
_SCALAR_FIELDS: tuple[tuple[str, str, int | None], ...] = (
    ("document_chunk_id", "VARCHAR", 64),
    ("document_id", "VARCHAR", 64),
    ("document_version_id", "VARCHAR", 64),
    ("content_hash", "VARCHAR", 128),
    ("content", "VARCHAR", 65535),
    ("chunk_index", "INT64", None),
    ("token_count", "INT64", None),
    ("char_count", "INT64", None),
)

#: content 的长度上限（与上面 VARCHAR(65535) 对齐）。
#:
#: 超长会被 Milvus 直接拒绝整批写入。这里先截断并**保留完整内容在
#: char_count 里**，因为 Milvus 侧只用于"召回后回表取原文"的定位，
#: 真正展示的正文从 Postgres 的 document_chunk 取。
_CONTENT_MAX_CHARS = 65535

#: 召回时回带的标量字段。
#:
#: **刻意不含 content**：正文很长（召回 top_k=50 就是几十万字符的
#: 序列化开销），而 PostgreSQL 的 document_chunk 才是正文的事实来源，
#: 查询侧召回后本来就要回表。带回来只会多付一次传输。
#:
#: document_chunk_id 不在列表里 —— 它是主键，Milvus 在 hit 的 id 字段
#: 里已经给了，重复取没有意义。
_OUTPUT_FIELDS: tuple[str, ...] = (
    "document_id",
    "document_chunk_id",
    "document_version_id",
    "chunk_index",
    "content_hash",
    "token_count",
    "char_count",
)


#: 需要开分析器的文本字段。开它们的唯一目的是**词面高亮**
#: （LexicalHighlighter 按词项定位片段），与向量召回无关。
#:
#: ⚠️ 字段参数是**建表时**定死的，Milvus 不支持改字段。已有 collection
#: 要生效只能 drop 重建 —— 重建前先确认没有想留的数据（或准备好重跑）。
_ANALYZED_FIELDS: frozenset[str] = frozenset({"content"})

#: 上面那些字段的分析器参数。
#:
#: enable_analyzer  建立词项索引，高亮才有"词"可定位
#: enable_match     允许按词匹配（highlight_query 用的是 match）
#:
#: 分析器选 chinese：语料是中文，标准分析器会把整句当一个词项，
#: 高亮要么整段标上、要么一个都标不出来。
_ANALYZED_FIELD_PARAMS: dict[str, Any] = {
    "enable_analyzer": True,
    "enable_match": True,
    "analyzer_params": {"type": "chinese"},
}

#: 高亮片段长度与条数（与 ES 侧取同一组值，免得两条路表现不一致）。
_FRAGMENT_SIZE = 150
_MAX_FRAGMENTS = 3

_PRE_TAG = "<em>"
_POST_TAG = "</em>"


def _highlights_of(hit: dict[str, Any]) -> list[str] | None:
    """
    从一条命中里取出高亮片段。

    **实测**的返回形状（Milvus 3.0-beta + pymilvus 3.0.1）：

        hit["highlight"]["<字段名>"]["fragments"] = ["<em>比亚迪</em>2025年…"]

    与 proto 的 common.HighlightResult{field_name, datas:[HighlightData
    {fragments, scores}]} 对得上 —— pymilvus 把它折成了以字段名为键的
    dict，scores 目前恒为空。

    没开 highlighter、或没匹配到可标片段时，键不存在 → 返回 None
    （"不产生高亮"，与"产生了但为空"是两回事，见 IndexSearchResult）。
    取不到 fragments 时退化成 [] 而不是 None：键在就说明高亮是**开了**
    的，只是这次没标出东西。
    """

    highlight = hit.get("highlight")

    if not highlight:
        return None

    field = highlight.get("content")

    if not field:
        return []

    return list(field.get("fragments") or [])


def _literal(value: Any) -> str:
    """
    Python 值 -> Milvus 过滤表达式里的字面量。

    三种类型必须分开拼：

        字符串  加双引号
        数值    裸写 —— 加引号会变成字符串比较，是**静默**的错误
        布尔    小写 true/false（Milvus 不认 Python 的 True/False）

    注意 bool 是 int 的子类，所以必须先判布尔。
    """

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        return str(value)

    return json.dumps(str(value))


class MilvusIndexBackend(IndexBackend):
    """Milvus 实现。持有一个由外部注入的（同步）MilvusClient。"""

    backend_type = StorageBackend.MILVUS

    def __init__(
            self,
            client: MilvusClient,
    ) -> None:
        self._client = client

    # ========================================================
    # 写入
    # ========================================================

    async def upsert(
            self,
            request: IndexUpsertRequest,
    ) -> None:
        """
        把一个 DocumentVersion 的 chunk 批量 upsert 进 collection。

        upsert 而不是 insert：reprocess 时同一批 chunk 会再来一次，
        insert 会因主键重复整批失败。
        """

        if not request.chunks:
            return

        dim = self._resolve_dim(request.chunks)

        await asyncio.to_thread(
            self._ensure_collection,
            request.index_identifier,
            dim,
        )

        rows = [self._to_row(chunk) for chunk in request.chunks]

        await asyncio.to_thread(
            self._client.upsert,
            request.index_identifier,
            rows,
        )

        await self._flush(request.index_identifier)

    async def delete(
            self,
            request: IndexDeleteRequest,
    ) -> None:
        """
        按 document_version_id 删掉本 collection 里的内容。

        ⚠️ Milvus 是**软删除** —— 别用 get_collection_stats 验证
        --------------------------------------------------------
        删除后 query / search **立刻**看不到那些行，但
        `get_collection_stats()["row_count"]` 仍报旧值，要等后台
        compaction 才降。实测：

            删后 query count              -> 0     ← 真实可见行数
            删后 get_collection_stats     -> 2     ← 滞后，不是没删掉

        验证删除要用 query（带上 filter），不要用 stats —— 否则会误判成
        "删除没生效"，然后去改一段本来就正确的代码。
        """

        if not await self._has_collection(request.index_identifier):
            # collection 不存在 = 已经没东西可删。
            # reprocess 第一次跑就会走到这里，不该报错。
            return

        await asyncio.to_thread(
            self._client.delete,
            request.index_identifier,
            filter=(
                "document_version_id == "
                f'"{request.document_version_id}"'
            ),
        )

        await self._flush(request.index_identifier)

    # ========================================================
    # 召回
    # ========================================================

    async def search(
            self,
            request: IndexSearchRequest,
    ) -> IndexSearchResponse:
        """
        稠密向量召回。

        查询向量由**调用方**算好放在 request.query_vector 里 ——
        本后端不持有 embedder，理由见 IndexSearchRequest 的说明。
        没给向量就显式失败：静默返回空结果会让"忘了向量化"和
        "库里确实没有"变成同一个现象。

        metric_type 必须与建表时一致（COSINE），否则 Milvus 会直接
        报错；它由本方法自己压进 search_params，不接受调用方覆盖，
        因为它是**建表时定死的物理事实**，不由调用方逐次决定。

        调用方可以通过 request.search_params 传 HNSW 的 ef / nprobe
        这类**近似搜索精度**参数（见下），但 metric_type 与由
        min_score 推出的 radius 始终以后端为准。
        """

        vector = request.query_vector

        if not vector:
            raise ValueError(
                f"{request.index_identifier}: 向量召回需要 "
                "query_vector，但请求里没有。"
                "调用方必须先用写入期同一个 embedder 把查询文本"
                "向量化再传进来（见 IndexSearchRequest 的说明）。"
            )

        if not await self._has_collection(request.index_identifier):
            # collection 还没建 = 这个后端从没写过数据。
            # 与 delete() 同样的处理：不是错误，就是"没有可召回的内容"。
            return IndexSearchResponse(results=[])

        # ---- search_params 合并 ----------------------------------
        #
        # 顺序是关键：**先铺调用方给的，再压后端定的**。反过来的话
        # 调用方一个 {"metric_type": "L2"} 就能把它顶掉，而它必须与
        # 建表时一致，不一致 Milvus 直接报错；radius 同理，它是从
        # min_score 推出来的，不该被 search_params 里的同名键覆盖。
        #
        # 调用方给 ef / nprobe 这类 HNSW 参数则照单全收 —— 它们只
        # 影响"近似搜索多准"，不改变结果的物理含义。
        search_params: dict[str, Any] = dict(
            request.search_params or {}
        )

        search_params["metric_type"] = "COSINE"

        if request.min_score is not None:
            # ⚠️ 与 L2 的直觉**相反**：COSINE 下越大越相似（见
            # _to_results 的说明），所以 radius 在这里是区间**下界**
            # —— "至少要多相似"。按 L2 的直觉当成上界写反了，
            # 会静默返回完全相反的结果，而且不报错。
            search_params["radius"] = request.min_score

        hits = await asyncio.to_thread(
            self._client.search,
            request.index_identifier,
            data=[list(vector)],
            anns_field="vector",
            limit=request.top_k,
            search_params=search_params,
            filter=self._filter_expression(request),
            output_fields=list(_OUTPUT_FIELDS),
            highlighter=self._highlighter(request.query),
        )

        return IndexSearchResponse(
            results=self._to_results(hits, request)
        )

    async def health_check(self) -> bool:
        try:
            await asyncio.to_thread(self._client.list_collections)
            return True
        except Exception:
            return False

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _resolve_dim(chunks: list[CanonicalChunk]) -> int:
        """
        从这一批 chunk 里定出维度。

        所有向量必须等长 —— 不等长是上游 embedder 出了问题（比如混了
        两个模型），在这里报错比让 Milvus 抛 schema 错误清楚得多。
        """

        dims = {
            len(chunk.vector)
            for chunk in chunks
            if chunk.vector is not None
        }

        if not dims:
            raise ValueError(
                "Milvus 后端收到的一批 chunk 里没有任何稠密向量 —— "
                "dense_vector 那一路的 embedding 没有产出 vector 字段。"
            )

        if len(dims) > 1:
            raise ValueError(
                f"同一批 chunk 出现了多种向量维度 {sorted(dims)} —— "
                "一个 collection 只能有一种维度，请检查 embedder 是否一致。"
            )

        return dims.pop()

    async def _flush(self, name: str) -> None:
        """
        写入与删除之后都必须 flush，否则**下一次查询看不到这次改动**。

        Milvus 的变更先进 WAL，未 flush 的数据既不进查询结果、也不进
        索引。实测（两处都踩过）：

            upsert 返回 upsert_count=2  ->  紧接着 query 返回 0 行
            delete 返回成功             ->  紧接着 query 仍返回 2 行
            flush 之后                  ->  两者才反映真实状态

        这对"先删后写"的 reprocess 尤其危险：删不可见时，旧向量会与
        新向量**同时**留在索引里，检索结果里混进上一版的内容。

        与 ES 那边的 refresh=True 是同一个理由与同一个代价：
        刚跑完的版本必须立刻可检索。量级是"每版本每路一次"，
        这个开销可以接受。
        """

        await asyncio.to_thread(self._client.flush, name)

    async def _has_collection(self, name: str) -> bool:
        return bool(
            await asyncio.to_thread(
                self._client.has_collection, name
            )
        )

    def _ensure_collection(self, name: str, dim: int) -> None:
        """建 collection（幂等）。同步 —— 调用方负责 to_thread。"""

        if self._client.has_collection(name):
            return

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        schema.add_field(
            field_name="document_chunk_id",
            datatype=self._varchar(),
            is_primary=True,
            max_length=64,
        )

        for field_name, kind, max_length in _SCALAR_FIELDS:

            if field_name == "document_chunk_id":
                continue

            # content 要额外开分析器 —— 这是**词面高亮的前提**：
            # LexicalHighlighter 按词项定位片段，字段没开分析器就没有
            # 词项可言，开 match 也没用。
            #
            # ⚠️ 这是建表时的字段参数，建完改不了：已有 collection 必须
            # 重建才生效（见 _ANALYZED_FIELDS 的说明）。
            extra = (
                _ANALYZED_FIELD_PARAMS
                if field_name in _ANALYZED_FIELDS
                else {}
            )

            schema.add_field(
                field_name=field_name,
                datatype=(
                    self._varchar() if kind == "VARCHAR"
                    else self._int64()
                ),
                max_length=max_length,
                **extra,
            )

        schema.add_field(
            field_name="vector",
            datatype=self._float_vector(),
            dim=dim,
        )

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            # 与查询侧一致才有意义。COSINE 要求向量已归一化 ——
            # embedder 默认开了 normalize，这里跟随。
            metric_type="COSINE",
        )

        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )

    @staticmethod
    def _highlighter(query: str):
        """
        Milvus 的**词面**高亮器。

        为什么用 LexicalHighlighter 而不是 SemanticHighlighter
        -----------------------------------------------------
        语义高亮（按语义相似标注片段）需要 Milvus 内部部署一个模型，
        拿 model_deployment_id —— 那是独立的运维动作，且与写入期用的
        embedder 不是同一个，标记出来的片段未必对应"向量为什么命中"。

        词面高亮不需要任何模型，代价是它标的是**查询词的字面出现**，
        对稠密召回而言是"借来"的信号。取舍写在 IndexSearchResult 的
        说明里，调用方应当知道这一点。

        依赖建表时的 enable_analyzer（见 _ANALYZED_FIELDS）——
        collection 没开分析器的话这里传了也不会有高亮。
        """

        return LexicalHighlighter(
            highlight_query=[
                {
                    # ⚠️ 必须是 "TextMatch" 这个**驼峰**字面量，对应
                    # Milvus 的 TEXT_MATCH 过滤条件。服务端会校验它，
                    # 写 "match" 会直接报
                    #   invalid highlight query type: match
                    # 目前只支持这一种类型。
                    "type": "TextMatch",
                    "field": "content",
                    "text": query,
                }
            ],
            pre_tags=[_PRE_TAG],
            post_tags=[_POST_TAG],
            fragment_size=_FRAGMENT_SIZE,
            num_of_fragments=_MAX_FRAGMENTS,
        )

    @staticmethod
    def _to_results(
            hits: Any,
            request: IndexSearchRequest,
    ) -> list[IndexSearchResult]:
        """
        Milvus 的 search 返回值 -> IndexSearchResult。

        MilvusClient.search 返回的是**按查询向量分组**的二维结构
        （`[[hit, ...]]`）—— 本后端每次只传一个查询向量，因此只取
        第 0 组。写成 `hits[0]` 而不是展平，是因为展平会把"将来真的
        传了多个向量"这件事悄悄变成一个错误的扁平结果。
        """

        if not hits:
            return []

        results: list[IndexSearchResult] = []

        for hit in hits[0]:

            entity = hit.get("entity") or {}

            # 主键在 hit["id"] 上；entity 里那份是同一列（已声明为
            # output_field 之外的标量），两边都摸一下更稳。
            raw_id = entity.get("document_chunk_id") or hit.get(
                "id"
            )

            if raw_id is None:
                continue

            results.append(
                IndexSearchResult(
                    document_chunk_id=UUID(str(raw_id)),
                    # Milvus 把它叫 distance，COSINE 下越大越相似，
                    # 语义与其它后端的 score 一致，直接沿用。
                    score=float(hit.get("distance") or 0.0),
                    retrieval_source=(
                        request.index_identifier
                    ),
                    metadata={
                        key: entity[key]
                        for key in _OUTPUT_FIELDS
                        if entity.get(key) is not None
                    },
                    highlights=_highlights_of(hit),
                )
            )

        return results

    @staticmethod
    def _filter_expression(
            request: IndexSearchRequest,
    ) -> str:
        """
        构造 Milvus 的布尔过滤表达式。

        返回空串表示不过滤 —— MilvusClient 把空串当作"没有过滤"，
        传 None 则视版本而定，所以统一收敛成空串。

        值的字面量按类型拼：字符串要引号、数值裸写、布尔用小写 ——
        全都加引号会让数值比较变成字符串比较（静默错误），
        全都不加引号则字符串列直接是语法错误。
        """

        clauses: list[str] = []

        if request.document_version_ids is not None:

            if not request.document_version_ids:
                # 空集合表达不出"恒假"（`in []` 在 Milvus 里不保证合法），
                # 而**静默不过滤**会返回本不该返回的结果。调用方应当在
                # 集合为空时直接短路（服务层就是这么做的），走到这里
                # 说明调用方漏了那一步 —— 显式失败。
                raise ValueError(
                    "document_version_ids 为空集合：无法表达"
                    "'什么都不要'。调用方应在集合为空时直接返回空结果，"
                    "不要把它当成'不过滤'。"
                )

            versions = ", ".join(
                f'"{version_id}"'
                for version_id in request.document_version_ids
            )

            clauses.append(
                f"document_version_id in [{versions}]"
            )

        for key, value in (request.filters or {}).items():
            clauses.append(f"{key} == {_literal(value)}")

        return " and ".join(clauses)

    @staticmethod
    def _to_row(chunk: CanonicalChunk) -> dict[str, Any]:
        """
        CanonicalChunk -> Milvus 行。

        只放 Milvus **需要**的字段。正文写进去是为了召回后能直接看到
        一段内容，但完整正文以 Postgres 的 document_chunk 为准 ——
        所以超长时截断而不报错（见 _CONTENT_MAX_CHARS）。
        """

        return {
            "document_chunk_id": str(chunk.document_chunk_id),
            "document_id": str(chunk.document_id),
            "document_version_id": str(chunk.document_version_id),
            "chunk_index": chunk.chunk_index,
            "content": chunk.content[:_CONTENT_MAX_CHARS],
            "content_hash": chunk.content_hash,
            "token_count": chunk.token_count or 0,
            "char_count": chunk.char_count,
            "vector": list(chunk.vector or []),
        }

    # --------------------------------------------------------
    # Milvus 的 DataType 常量。写成方法而不是模块级常量：
    # import pymilvus.DataType 会把它拉进模块级依赖，
    # 而这些值只在建表时用到。
    # --------------------------------------------------------

    @staticmethod
    def _varchar():
        from pymilvus import DataType
        return DataType.VARCHAR

    @staticmethod
    def _int64():
        from pymilvus import DataType
        return DataType.INT64

    @staticmethod
    def _float_vector():
        from pymilvus import DataType
        return DataType.FLOAT_VECTOR
