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

from collections.abc import Sequence
from typing import Any

from elasticsearch import AsyncElasticsearch, BadRequestError

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
#: 索引级 settings。
#:
#: **没有自定义分析器了** —— 这里曾经有一个 `cjk_date`，用 char_filter 把
#: `YYYY-MM-DD` 与中文写法都压成原子词项供 `content.dates` 子字段精确匹配。
#: 它被删掉了，两条理由：
#:
#:   1. `pattern_replace` **做不到条件补零**，所以它只能产出不补零的形式
#:      （`2007-1-12` 与 `2007-11-2` 都变成 `2007112`）—— 两个不同的日子
#:      撞成同一个词项，于是"A 日的题命中 B 日的块"，且不报错。
#:   2. 它想解决的问题（"日期被切成三段，定位不到那一天"）现在由
#:      `content_dates` 字段解决得更彻底：那个字段的值是**应用侧**用
#:      `services/document/dates.py` 算的补零键，精确、无歧义，而且
#:      "查询写中文、正文写短横线"也照样对得上（分析器只能归一它认得的
#:      写法）。
#:
#: content 因此保持内置 `cjk` 分析器（见 _MAPPING 里的说明）。
_SETTINGS: dict[str, Any] = {}

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
        },
        "content_hash": {"type": "keyword"},
        "token_count": {"type": "integer"},
        "char_count": {"type": "integer"},
        # 正文里出现的日期键（**补零**的 `YYYYMMDD`，如 `20070412`），由
        # 应用侧在阶段 10 算好（`services/document/dates.py`）。
        #
        # 用 **keyword** 而不是分析器：值已经是规范化过的字面量，查询侧给的
        # 是同一形态（改写器算的 `date_keys`），`terms` 精确匹配即可 ——
        # 走分析器反而会把它切成 bigram，又回到模糊匹配。
        #
        # ⚠️ 它取代了此前的 `content.dates` 子字段 + `cjk_date` 分析器：
        # 那个 char_filter 做不到条件补零，只能产出不补零的形态，于是
        # `2007-1-12` 与 `2007-11-2` 变成同一个词项（撞键）。这个字段在
        # Python 里算，补零是免费的。
        "content_dates": {"type": "keyword"},
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

#: 短语加权（`boost_phrases`）每一段的 boost。
#:
#: 12 是**量出来的**：同一条 case（金标在 keyword 那一路排 140），把日期
#: 加成 `match_phrase` 短语后名次到 **4**；另外三条 123→11、48→6、18→8。
#:
#: 调它之前先量：加大只会让"含该短语"压过别的信号，而短语本身可能是
#: 改写器认错的（例如把"第 2 项议案"当成判别词）。它只影响**排序**，
#: 不参与筛选（`should` 在有 `must` 时是可选的），所以调错不会让结果
#: 变空 —— 只会让排序偏。
_PHRASE_BOOST = 12


#: `content_dates` 字段那一路的 boost。
#:
#: 比短语那一路（12）更高：字段是**归一化过**的精确匹配（两种写法在
#: 写入期就压成了同一个键），比"在正文里找字符串"更可信 —— 短语匹配会
#: 被"这个日期出现在块里的哪个位置/哪种句式"影响，而字段直接答
#: "这块正文里有没有这一天的记录"。
_DATE_BOOST = 24


def _date_clauses(keys: Sequence[str]) -> list[dict[str, Any]]:
    """
    `date_keys` -> `terms` 子句（打 `content_dates` 字段）。

    用 `should` 而不是 `filter`：见调用处的说明（日期必然出现在金标块里，
    是出题方式决定的 —— 硬筛会把答案漏进检索）。

    ⚠️ 键的形态必须与写入期一致（**补零**的 `YYYYMMDD`，见
    `services/document/dates.py`）。
    对不上的表现是**这条子句永远不加分**，而结果只是"排序没变好"，
    不会报错。
    """

    wanted = [key for key in dict.fromkeys(keys) if key]

    if not wanted:
        return []

    return [
        {"terms": {"content_dates": wanted, "boost": _DATE_BOOST}}
    ]


def _phrase_clauses(phrases: Sequence[str]) -> list[dict[str, Any]]:
    """
    `boost_phrases` -> ES 的 `should` 子句列表。

    只打 `content`（不打 `context_text`）：实测里加分的那些日期出现在
    **正文**（散文与 `#### 2007-04-12 ｜ …` 小标题都在 content 里）；
    `context_text` 是标题+section，里面那个日期是**块所属 section** 的，
    不是这块正文讲的那件事 —— 拿它做短语加权会奖励"块前面的那个日期"，
    而那正是我们判定为错锚的东西。

    空片段与重复片段都过滤掉：空短语在 ES 里是语法错误，重复只是让同一
    条子句加两次分（等价于偷偷调大 boost）。
    """

    seen: list[str] = []

    for phrase in phrases:

        text = (phrase or "").strip()

        if text and text not in seen:
            seen.append(text)

    return [
        {
            "match_phrase": {
                "content": {"query": text, "boost": _PHRASE_BOOST}
            }
        }
        for text in seen
    ]


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
                    # 只有两块：正文与继承下来的上下文。
                    #
                    # 曾经还有 `content.dates`（把日期压成原子词项的子字段）。
                    # 它删了：精确匹配日期现在由 `content_dates` 字段（下面
                    # 的 should）负责，那条**无歧义**（键是应用侧补零算的），
                    # 而子字段的 char_filter 做不到条件补零、会撞键。
                    "fields": [
                        "content",
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

        bool_query: dict[str, Any] = {
            "must": [must],
            "filter": filters,
        }

        # 判别词的两条加权通道（都只加分、不筛选）。
        #
        # 为什么单列 should 而不是拼进 query 文本：拼进去只是多几个词项，
        # 而 `minimum_should_match` 是**按比例**算的 —— 判别词在长问句里
        # 占的份额很小，推不动排序。这两条都不受那条比例约束。
        #
        # ⚠️ 有 `must` 时 `should` 默认是**可选**的（分数加成，不参与
        # 筛选），所以不会把任何候选挡在外面 —— 没命中的只是不加分。
        #
        # ⚠️ 而且这里刻意**不用 `filter`**：那会变成硬筛。日期确实是这批
        # 语料里唯一的判别词，但它"必然出现在金标块里"是**出题方式决定
        # 的**（问题就是从含该日期的块生成的）—— 拿它做硬筛等于把答案
        # 漏进检索，命中率会虚高，而且虚高的部分看不出来。
        should = _phrase_clauses(request.boost_phrases) + _date_clauses(
            request.date_keys
        )

        if should:
            bool_query["should"] = should

        query: dict[str, Any] = {"bool": bool_query}

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

        **不**用"create 失败就算了"：那会把 "mapping 冲突"这类真问题
        一起吞掉。下面只吞**能确认是并发竞态**的那一种失败。

        竞态为什么会发生：容器名是硬编码的全局名字（见 containers.py），
        跨 DTD、跨文档共享，而扇出分支是并发的。两个文档同时开始写入时，
        双方都可能在 exists 返回 False 之后、create 之前走到这里 ——
        后到的那个收到 400 resource_already_exists_exception：

            index [document_instance_keyword/rrBsRUK1QN61B6YR_gsf0g]
            already exists

        这不是异常情况，是**常态**（全量 reprocess 必然触发）。
        """

        if await self._index_exists(name):
            return

        try:
            await self._client.indices.create(
                index=name,
                settings=_SETTINGS,
                mappings=_MAPPING,
            )

        except BadRequestError:
            # 判据不是错误码字符串 —— `ApiError.error` 是 message 的
            # 向后兼容别名（返回的是 reason 文本），按它匹配既脆又跟着
            # ES 版本走。改成看**事实**：再查一次，索引在就说明是并发
            # 建好了；不在就说明 create 真的失败了（mapping 冲突之类），
            # 原样抛出去。
            if not await self._index_exists(name):
                raise

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

        # 没有日期就不写这个键（与 context_text 同一口径）：空数组会占
        # 一份索引空间，而且 `exists` 查询会把它当成"有值"。
        if chunk.dates:
            document["content_dates"] = list(chunk.dates)

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
