"""
Elasticsearch 索引后端。

服务哪些 retrieval
------------------
    keyword         text 字段上的 BM25 全文检索
    sparse_vector   sparse_vector 字段上的学习式稀疏检索

两者共用**同一个 ES 实例**，但落在各自的 index（`index_identifier`）上 ——
稀疏向量与关键词的索引结构、打分方式都不同，混在一个 index 里会让
BM25 的字段统计互相污染。

客户端从哪来
------------
不在这里构造。组合根（main.py）把 datasource_manager 里那个
AsyncElasticsearch 实例注入进来 —— 连接池、认证、超时都由 infra 层管。

⚠️ 那份注释值得一读：ES 客户端**刻意**没用 httpx 后端，因为 httpx 默认
`trust_env=True`，会读 Windows 注册表里的系统代理，把连 localhost:9200
的请求塞进 127.0.0.1:7890，返回 502。见 infra/datasource/elasticsearch.py。

mapping 一律由本后端负责创建
----------------------------
`index_identifier` 指向的 index 可能还不存在。upsert 前先确保它存在，
否则第一次写入会以 404 失败 —— 而这个失败发生在后台流水线里，
排查起来要绕一大圈。
"""

from __future__ import annotations

from typing import Any

from elasticsearch import AsyncElasticsearch

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

__all__ = ["ElasticsearchIndexBackend"]


#: 索引 mapping。
#:
#: 三个刻意的选择：
#:
#: 1. **id 类字段用 keyword 而不是 text** —— 它们是精确匹配与过滤用的，
#:    用 text 会被分词，`document_version_id` 那种 uuid 分词后毫无意义。
#:
#: 2. **locator / metadata 用 enabled: false** —— 它们的内容随 retrieval
#:    与 enricher 配置而变，不需要检索；关掉索引能省下大量 mapping 爆炸
#:    与磁盘（ES 默认会对未知字段做动态映射，键一多就成灾难）。原样存、
#:    原样取回即可。
#:
#: 3. **sparse_vector 字段总是声明** —— 即使本次写的是 keyword。ES 允许
#:    后续给 mapping 加字段，但**不允许改变已有字段的类型**；先声明好，
#:    同一个 index 才能在两种 retrieval 之间复用而不用重建。
#:
#: ⚠️ 一个会让人以为"写入丢了"的 ES 行为
#: --------------------------------------
#: **sparse_vector（以及 dense_vector）类型的字段不进 `_source`。**
#: 实测（原始 API，绕开本项目代码）：
#:
#:     写入  {"content": "x", "sparse_vector": {"13323": 0.5}}
#:     取回  _source = {"content": "x"}        ← 向量"不见了"
#:     检索  term(sparse_vector) -> 命中 1     ← 其实在索引里
#:     取回  search(fields=["sparse_vector"]) -> {"sparse_vector": [{...}]}
#:
#: 也就是说它**被索引了、能检索**，只是不随 _source 返回。排查时别
#: 因为 _source 里没有就以为没写进去。
#: 索引级 settings：分析器定义。
#:
#: 为什么需要一个自定义分析器
#: --------------------------
#: cjk 分析器 = standard 分词 + cjk_bigram，而 **cjk_bigram 只对汉字
#: 做 bigram**，连字符不在它的处理范围里。于是 standard 分词器在 `-`
#: 处断开，日期被打成三段：
#:
#:     查询 "2026-05-22"          -> ['2026', '05', '22']
#:     正文 "#### 2026-04-25 ｜"  -> ['2026', '04', '25', ...]
#:
#: 结果是 match 查询退化成"命中 2026 **或** 05 **或** 22" —— 几乎每篇
#: 都算命中，"2026-05-22"根本定位不到那一天。
#:
#: 对照：不带连字符的数字（`1024127`）本来就是**一个**词项，精确。
#: 所以出问题的只有带连字符的日期这一个格式。
#:
#: 解法：用 pattern_replace **在分词之前**把 `YYYY-MM-DD` 压成
#: `YYYYMMDD`，它就成了一个原子词项。查询侧走同一个分析器，因此
#: "2026-05-22" 与正文里的 "2026-04-25" 各自变成单个词项，能精确对上。
#:
#: 这在 content.dates 子字段上生效，content 保持原样（理由见那里的注释）。
_SETTINGS: dict[str, Any] = {
    "analysis": {
        "char_filter": {
            "date_normalizer": {
                "type": "pattern_replace",
                "pattern": r"(\d{4})-(\d{2})-(\d{2})",
                "replacement": "$1$2$3",
            },
        },
        "analyzer": {
            # 与内置 cjk 逐项对齐（standard + cjk_width + lowercase +
            # cjk_bigram），只多一个 char_filter。刻意不写 "type": "cjk"
            # 继承 —— 自定义分析器不能继承内置的过滤器列表，必须列全。
            "cjk_date": {
                "type": "custom",
                "char_filter": ["date_normalizer"],
                "tokenizer": "standard",
                "filter": ["cjk_width", "lowercase", "cjk_bigram"],
            },
        },
    },
}


_MAPPING: dict[str, Any] = {
    "properties": {
        "document_chunk_id": {"type": "keyword"},
        "document_id": {"type": "keyword"},
        "document_version_id": {"type": "keyword"},
        "chunk_index": {"type": "integer"},
        "content": {
            "type": "text",
            # 中文必须显式指定分析器，**默认的 standard 会按单字切**。
            #
            # 实测（同一段文本）：
            #     standard  东 方 财 富 股 东 户 数      每个汉字一个词项
            #     cjk       东方 方财 财富 富股 股东 东户 户数   bigram
            #
            # 单字切分下"股东户数"退化成 `股`/`东`/`户`/`数` 四个独立词项，
            # 于是**任何含这几个字的 chunk 都算命中** —— 一份财报里几乎每
            # 个 chunk 都含"股""东"，BM25 的排序因此接近随机，高亮也会
            # 把一个词标成一串单字（`<em>股</em><em>东</em><em>户</em><em>数</em>`）。
            #
            # 选 cjk 而不是 ik/smartcn：**它是 ES 内置的**，这两个都要装
            # 插件（当前容器里一个插件都没有，见 `_cat/plugins`）。
            # 将来若要装 IK，把这里换成 `ik_max_word` + `search_analyzer:
            # ik_smart` 即可，其余代码不用动。
            "analyzer": "cjk",
            # term_vector 是 **fvh（Fast Vector Highlighter）的前提** ——
            # 它在索引时额外记录词项的位置与偏移，查询时才能直接定位片段。
            # 不开的话只能用 unified highlighter：功能上也标得出来，但要
            # 在查询时重新分析正文，长文本下慢得多。
            #
            # 代价是索引体积变大（存了 positions + offsets）。
            "term_vector": "with_positions_offsets",
            # 日期精确匹配用的子字段 —— 见下面 _SETTINGS 的 date_normalizer。
            #
            # 为什么是**子字段**而不是直接改 content 的分析器：
            # 把日期压成原子词项之后，索引里就不再存在 `2026` 这个词项，
            # "只输入 2026"会匹配不到任何日期。保留 content 原样，两种
            # 粒度就都有了 —— 大范围靠 content，精确定位靠 content.dates。
            #
            # 另外 content 挂着 fvh 的 term_vector，动它的分析器要重新
            # 验证高亮偏移；子字段独立，风险为零。
            #
            # 不设 term_vector：高亮只从 content 出，这个字段只参与打分。
            "fields": {
                "dates": {
                    "type": "text",
                    "analyzer": "cjk_date",
                },
            },
        },
        "content_hash": {"type": "keyword"},
        "token_count": {"type": "integer"},
        "char_count": {"type": "integer"},
        # 文档级上下文，**可索引**，参与相关性排序。
        #
        # 解决的是这个具体问题：一份按公司维度组织的长文档，中间的纯数据
        # chunk（一长串股东户数）正文里不会重复公司名，于是查询
        # "东方财富 股东户数" 时公司名在 chunk 层面不可见 —— 实测结果里
        # 会混进贵州茅台的 chunk，因为两者正文都只有"股东户数"。
        #
        # 与 content 的关系：
        #     content      原文。展示与高亮的**偏移基准**。
        #     context_text 标题/公司/来源/section 拼成的一小段。
        #
        # 分成两个字段而不是拼进 content：拼进去会改 content_hash，
        # 而 content_hash 与唯一约束一起撑着 chunk 的去重语义 ——
        # 改了它，重新切分时判重失效，而且**不报错**。
        "context_text": {
            "type": "text",
            "analyzer": "cjk",
            # fvh 需要它，同 content。
            "term_vector": "with_positions_offsets",
        },
        # 只回表、不索引。
        #
        # 刻意**不**把它打开来承载上下文：那是个自由 dict，ES 会为每个
        # 新键动态推 mapping，键一多就字段爆炸、类型还可能推错。可索引
        # 的东西要有明确的字段定义（见上面的 context_text）。
        "locator": {"type": "object", "enabled": False},
        "metadata": {"type": "object", "enabled": False},
        "sparse_vector": {"type": "sparse_vector"},
    },
}


#: 高亮片段长度与条数。取值是展示与体积的折中：
#: 片段太短会切断句子、看不出上下文；太多则一条结果的响应体积翻几倍。
_FRAGMENT_SIZE = 150
_MAX_FRAGMENTS = 3

#: 标记用的标签。写死成 <em> 而不是做成配置：调用方拿到的是一段
#: 带标记的文本，标签必须先约定好才解析得了。要换就两边一起换。
_PRE_TAG = "<em>"
_POST_TAG = "</em>"

#: BM25 那一路的最低词项命中比例。
#:
#: 为什么必须给：`match` 默认是 OR，命中**任一**词项即算匹配。中文经
#: cjk 拆成几十个 bigram 之后这个门槛形同虚设 —— 实测一个 12 字查询
#: 在 259 个 chunk 里命中 259 个（100%），BM25 的排序因此没有区分度，
#: 目标 chunk 被埋在第 61 名。
#:
#: 取值是**精度与召回的折中**，而且没有"通用推荐值"可抄 —— 必须实测。
#: 下面是查询「东方财富2026年4月股东人数」在一份 259 chunk 的文档上的
#: 实测结果：
#:
#:     0        命中 259 / 259   目标 chunk 第 61 名   （等于没加）
#:     50%      命中  49 / 259   目标 chunk 第 48 名
#:     70%      命中   0 / 259   ← 整个查询返回空
#:     85%      命中   0 / 259
#:
#: **为什么 70% 会空**：cjk 是 bigram 分词，中文与数字交界处会产生
#: `富2` / `6年` / `年4` 这类**跨字边界**的 bigram，任何一个 chunk 里都
#: 不存在。要求命中七成，等于要求这些不存在的组合也命中 —— 不可能满足。
#:
#: 所以这个值只能取到"能滤掉一部分、但不至于滤空"的区间。50% 在实测里
#: 落在这个区间内。**改它之前先量一遍**，别凭感觉调：取错的表现是
#: "某些查询莫名其妙没结果"，而不会报任何错。
_MIN_SHOULD_MATCH = "50%"


class ElasticsearchIndexBackend(IndexBackend):
    """ES 实现。持有一个由外部注入的 AsyncElasticsearch。"""

    backend_type = StorageBackend.ELASTICSEARCH

    def __init__(
            self,
            client: AsyncElasticsearch,
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
        把一个 DocumentVersion 的 chunk 批量写进 index。

        用 `index` 动作而不是 `create`：_id 就是 document_chunk_id，
        同一个 chunk 重跑（reprocess）时应当覆盖而不是报冲突。
        """

        if not request.chunks:
            return

        await self._ensure_index(request.index_identifier)

        operations: list[dict[str, Any]] = []

        for chunk in request.chunks:
            operations.append({
                "index": {
                    "_index": request.index_identifier,
                    "_id": str(chunk.document_chunk_id),
                }
            })
            operations.append(self._to_document(chunk))

        response = await self._client.bulk(
            operations=operations,
            # 写完即可见。代价是一次 refresh，但让"写完就能召回"成立 ——
            # 否则刚跑完的版本搜不到，前端会以为没成功。
            refresh=True,
        )

        if response.get("errors"):
            raise ValueError(
                "Elasticsearch bulk 部分失败: "
                f"{self._failed_items(response)}"
            )

    async def delete(
            self,
            request: IndexDeleteRequest,
    ) -> None:
        """删掉某个 DocumentVersion 在本 index 里的全部内容。"""

        if not await self._index_exists(request.index_identifier):
            # 索引都不存在，等于已经删干净了。直接返回而不是报错 ——
            # reprocess 时先删再写，第一次跑就会走到这里。
            return

        await self._client.delete_by_query(
            index=request.index_identifier,
            query={
                "term": {
                    "document_version_id": str(
                        request.document_version_id
                    )
                }
            },
            refresh=True,
            # 版本不存在时也算成功（要删的东西本来就没有）
            conflicts="proceed",
        )

    # ========================================================
    # 召回
    # ========================================================

    async def search(
            self,
            request: IndexSearchRequest,
    ) -> IndexSearchResponse:
        """
        召回：**BM25 与稀疏向量两路**，按请求里给了哪个查询表示来选。

        两路的容器不同（document_instance_keyword /
        document_instance_sparse_vector），一次调用只服务一路 —— 调用方
        （DocumentRetrieveService）按该 retrieval 的类型算好对应的查询
        表示再传进来，所以这两个字段在实践中互斥：

            sparse_query 给了 -> 稀疏向量召回
            否则              -> BM25 全文召回（用 query）

        query 两者都会给（它是原文，观测与 metadata 也要用），因此判断
        依据是 **sparse_query 是否存在**，不是 query 是否存在。

        为什么不做成"两个都给了就混合检索"：混合是两个信号的加权，
        而权重该由配置表达（DTD 的 fusion 配置已经能做跨路融合）。
        在这里隐式混合会让"这个 retrieval 到底查了什么"变得说不清。
        """

        if not await self._index_exists(request.index_identifier):
            return IndexSearchResponse(results=[])

        filters = self._filters(request)

        if request.sparse_query:
            must: dict[str, Any] = {
                "sparse_vector": {
                    "field": "sparse_vector",
                    # ⚠️ ES 要求键是**字符串**（JSON 的键只能是字符串）。
                    # 与写入侧 _to_document 是同一处约定，两边必须一致。
                    "query_vector": {
                        str(dim): weight
                        for dim, weight
                        in request.sparse_query.items()
                    },
                }
            }
        else:
            # 两块都搜：content 是原文，context_text 是从文档级继承下来的
            # 上下文（标题/公司/来源/section）。后者让"东方财富"这种
            # **不在正文里出现**的条件也能参与排序。
            #
            # 用 multi_match 而不是把 context 拼进 content：拼接会改
            # content_hash（见 _MAPPING 里的说明），而且高亮的偏移会
            # 对不上存储的正文。
            must = {
                "multi_match": {
                    "query": request.query,
                    # content.dates 是**子字段**，与 content 索引同一段
                    # 正文、只是分析方式不同。加进来是为了让
                    # "2026-05-22" 这种日期能精确命中：content 把它切成
                    # 三个词项（不可用），content.dates 把它压成一个
                    # 原子词项（精确）。
                    #
                    # 两边的分数由 multi_match 取最优，所以日期精确匹配
                    # 不会被 content 那边的模糊命中拉低。
                    "fields": [
                        "content",
                        "content.dates",
                        "context_text",
                    ],
                    # 没有它，`match` 等价于"命中任一词项" —— 中文经 cjk
                    # 拆出几十个 bigram 之后，几乎每一篇都算命中。
                    #
                    # 实测：查询「东方财富2026年4月股东人数」在 259 个
                    # chunk 里命中 259 个 —— 命中率 100%，BM25 的排序
                    # 因此完全没有区分度，排第 61 的那条和排第 1 的
                    # 语义上一样相关。
                    #
                    # 要求命中七成词项，把"沾一点边"的挡在外面。
                    "minimum_should_match": _MIN_SHOULD_MATCH,
                }
            }

        query: dict[str, Any] = {
            "bool": {
                "must": [must],
                "filter": filters,
            }
        }

        # 高亮只对**词项匹配**那一路有意义。
        #
        # sparse_vector 查询走的是向量相似度，ES 的高亮器不参与 ——
        # 它按词项定位片段，而稀疏查询没有"哪个词命中"这个概念
        # （要标得把 token id 反查回词面，需要查询侧持有同一份分词器）。
        # 所以稀疏那一路不带 highlight，highlights 返回 None。
        highlight = (
            None
            if request.sparse_query
            else {
                # fvh：依赖 mapping 里的 term_vector（见 _MAPPING）。
                # 它直接读索引里存好的位置偏移，不必在查询时重分析正文。
                "type": "fvh",
                "fields": {
                    "content": {
                        "fragment_size": _FRAGMENT_SIZE,
                        "number_of_fragments": _MAX_FRAGMENTS,
                    },
                    # 上下文也要高亮 —— 否则"公司名命中了"这件事在结果里
                    # 完全看不出来（正文里没有公司名，标不出来）。
                    "context_text": {
                        "fragment_size": _FRAGMENT_SIZE,
                        "number_of_fragments": _MAX_FRAGMENTS,
                    },
                },
                "pre_tags": [_PRE_TAG],
                "post_tags": [_POST_TAG],
            }
        )

        response = await self._client.search(
            index=request.index_identifier,
            query=query,
            size=request.top_k,
            # min_score 收的是**原始分**（BM25 分 / 稀疏点积），在评分
            # 阶段就丢弃不达标的文档 —— 比"召回后再剔"省下 top_k 名额：
            # ES 是先按 size 截断再返回的，召回后过滤等于让凑数文档
            # 白白占掉本该给相关文档的位置。
            #
            # 不给就是不给：传 None 会被 ES 当成非法值，所以这里必须
            # 条件展开而不是直接写 min_score=None。
            **(
                {"min_score": request.min_score}
                if request.min_score is not None
                else {}
            ),
            **({"highlight": highlight} if highlight else {}),
        )

        results = [
            IndexSearchResult(
                document_chunk_id=hit["_source"]["document_chunk_id"],
                score=float(hit["_score"] or 0.0),
                # 标注来源：多路召回融合时靠它区分结果来自哪，
                # 否则 4 个 index 的结果混在一起无从解释。
                retrieval_source=request.index_identifier,
                metadata=hit["_source"].get("metadata") or {},
                # 没做高亮的那一路这里取不到键 —— 给 None 而不是 []，
                # 两者含义不同（见 IndexSearchResult 的说明）。
                highlights=(
                    self._highlights_of(hit)
                    if highlight
                    else None
                ),
            )
            for hit in response["hits"]["hits"]
        ]

        return IndexSearchResponse(results=results)

    async def health_check(self) -> bool:
        try:
            return bool(await self._client.ping())
        except Exception:
            return False

    # ========================================================
    # internals
    # ========================================================

    async def _index_exists(self, name: str) -> bool:
        return bool(await self._client.indices.exists(index=name))

    async def _ensure_index(self, name: str) -> None:
        """
        索引不存在就按 _MAPPING 建。

        用 exists + create 而不是"create 失败就算了"：后者会把
        "mapping 冲突"这类真问题一起吞掉。
        """

        if await self._index_exists(name):
            return

        await self._client.indices.create(
            index=name,
            settings=_SETTINGS,
            mappings=_MAPPING,
        )

    @staticmethod
    def _to_document(chunk: CanonicalChunk) -> dict[str, Any]:
        """
        CanonicalChunk -> ES 文档。

        sparse_vector 是 {维度索引: 权重}，ES 的 sparse_vector 字段
        接受的正是这种映射 —— 但**键必须是字符串**（JSON 的键只能
        是字符串），所以要转一次。
        """

        document: dict[str, Any] = {
            "document_chunk_id": str(chunk.document_chunk_id),
            "document_id": str(chunk.document_id),
            "document_version_id": str(chunk.document_version_id),
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "content_hash": chunk.content_hash,
            "token_count": chunk.token_count,
            "char_count": chunk.char_count,
            "locator": chunk.locator or {},
            "metadata": chunk.metadata or {},
        }

        if chunk.sparse_vector:
            document["sparse_vector"] = {
                str(dim): weight
                for dim, weight in chunk.sparse_vector.items()
            }

        # 没有上下文就**不写这个键**，而不是写空串。
        #
        # 写空串会让 multi_match 在这个字段上做一次无意义的匹配（空文本
        # 上匹配不到东西，但会参与打分归一），也可能让"字段存在"这种
        # 判断产生歧义。缺键最干净。
        if chunk.context_text:
            document["context_text"] = chunk.context_text

        return document

    @staticmethod
    def _highlights_of(
            hit: dict[str, Any],
    ) -> list[str]:
        """
        把一条命中里**两个字段**的高亮片段合并成一个列表。

        顺序是 content 在前、context_text 在后，不是随便排的：

            content 片段      正文里的一段话，展示时直接可读 ——
                              "命中在哪一段"
            context_text 片段 "东方财富 300059 大事提醒" 这种短上下文 ——
                              "**为什么**这条会被召回"（公司名在正文里
                              根本没有，只有它能标出来）

        前端按顺序渲染就能先给人看命中位置，再看匹配理由。

        调用方已确认本次请求带了 highlight，所以这里不必判空 ——
        返回 [] 表示"有高亮能力但这次没标出东西"（见 IndexSearchResult）。
        """

        highlight = hit.get("highlight") or {}

        return [
            fragment
            for field in ("content", "context_text")
            for fragment in (highlight.get(field) or [])
        ]

    @staticmethod
    def _filters(
            request: IndexSearchRequest,
    ) -> list[dict[str, Any]]:
        """
        把 IndexSearchRequest 的过滤条件翻成 ES 的 filter 子句。

        document_version_ids 单独处理：它是最常用的过滤维度
        （只搜当前生效版本），且必须精确匹配。
        用 terms 而不是 term —— 它是**一个集合**（整个 DTD 下所有
        文档的生效版本），逐个 term 拼会变成一个巨大的 bool。
        """

        filters: list[dict[str, Any]] = []

        if request.document_version_ids is not None:
            filters.append({
                "terms": {
                    "document_version_id": [
                        str(version_id)
                        for version_id
                        in request.document_version_ids
                    ]
                }
            })

        for field, value in (request.filters or {}).items():
            filters.append({"term": {field: value}})

        return filters

    @staticmethod
    def _failed_items(
            response: dict[str, Any],
            limit: int = 3,
    ) -> list[dict[str, Any]]:
        """挑前几条失败原因出来 —— 只报"有错误"没法排查。"""

        failed = [
            item["index"]
            for item in response.get("items", [])
            if item.get("index", {}).get("error")
        ]
        return failed[:limit]
