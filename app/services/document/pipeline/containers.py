"""
retrieval_type -> 物理容器名的映射。

**写入期与查询期共用这一份。**

原本它只写在 pipeline/nodes/indexing.py 里（写入期用）。查询期
（retrieve/ + DocumentRetrieveService）也必须用**完全相同**的名字去召回 ——
一旦两边各存一份，漂移的表现是"写进去了但搜不到"，而不是报错：
两边名字不一致时索引照样建、照样写，只是永远命中不了。
所以提到这里，谁都不许再抄一遍。

为什么硬编码而不是让用户配
--------------------------
查询期必须**一次查到全部文档**，所以同一个 retrieval_type 的数据必须
落在同一个物理容器里。一旦允许每个类型/每个版本各配一个，查询就要跨
N 个容器做 federation 再融合 —— 复杂度和延迟都上去了，而且用户配错的
表现是"检索搜不到"，不是报错。

为什么按 retrieval_type 隔离，而不是所有类型挤一个容器
------------------------------------------------------
不是怕互相覆盖（chunk 是按 retrieval 各自产一份的，_id 天然不相交），
而是：

  1. 索引设置不同 —— keyword 要中文分词 analyzer，sparse_vector 不需要
     分析（向量由 ES 内部维护倒排）。一个 index 只有一套 settings。
  2. 独立重建 —— 换 embedding 模型重建稀疏索引时，不必碰关键词索引。
  3. 清理与排查 —— 一眼看得出容器里是什么，删某个检索类型不牵连别的。

名字是**契约**：一旦建了索引就不能随便改（改了等于全量重建 + 数据迁移），
所以这里逐条写出来而不是用 f-string 拼 —— 要改的人会看到它是个契约。

跨 DocumentTypeDefinition 共享
------------------------------
名字只由 retrieval_type 决定，所以类型 A 与类型 B 的 dense_vector 都写
document_instance_dense_vector。对 ES 没问题（靠 document_version_id /
document_id 字段过滤隔离业务）。

⚠️ 对 Milvus 要留意：一个 collection 只能有**一种维度**。两个文档类型
若用了不同维度的 embedder，会在这里撞上 —— 这跟命名无关，是"每后端一个
固定容器"自带约束，得靠"一个系统只用一种 dense embedder"来管。
"""

from __future__ import annotations

__all__ = [
    "EXTERNAL_INDEX",
    "PG_RETRIEVAL_TYPES",
    "external_container",
]

#: 需要外部索引后端的 retrieval_type -> 容器名。
#:
#: 不在这里的 retrieval_type（当前是 document）没有外部容器，
#: 见 PG_RETRIEVAL_TYPES。
EXTERNAL_INDEX: dict[str, str] = {
    "keyword": "document_instance_keyword",
    "sparse_vector": "document_instance_sparse_vector",
    "dense_vector": "document_instance_dense_vector",
    # graph 待接入：Neo4j 的 label 同样叫 document_instance_graph，
    # 但图那一路的抽取与入库都还没做（见 embedder/graph.py）。
}

#: 容器**就是** PostgreSQL 里那张表的 retrieval_type。
#:
#: document_chunk 表已经是一张固定表（只有 document 类型用它），
#: 不需要也不应该改叫 document_instance_document —— 它存的是 chunk，
#: 现名更准，改名要动迁移 + 全套仓储/模型/schema。
PG_RETRIEVAL_TYPES: frozenset[str] = frozenset({"document"})


def external_container(
        retrieval_type: str,
) -> str | None:
    """
    取某个 retrieval_type 的外部容器名。

    None 表示它不走外部索引（当前只有 document，见 PG_RETRIEVAL_TYPES）。
    调用方**不要**把 None 当成"用默认容器"—— 那会让写入期与查询期
    对同一份数据给出不同答案。
    """

    return EXTERNAL_INDEX.get(retrieval_type)
