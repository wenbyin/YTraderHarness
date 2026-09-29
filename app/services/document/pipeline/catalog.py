"""
Document 处理流水线的节点目录。

这是**观测契约的来源**：node_id / label / group / sequence 四项
会随每个节点行一起落库并冻结，前端据此渲染运行详情。

三条硬规则
----------
1. **永不重编号已有 sequence。** 它是冻结的历史值，重编号会让
   历史运行的进度条错乱。新节点只往后追加序号。

2. **永不复用退役的 node_id。** 节点下线时把它移进
   RETIRED_NODE_SPECS 并标 deprecated，而不是删掉 ——
   历史运行里的旧 node_id 必须永远能查到解释。

3. **任何拓扑变更必须 bump GRAPH_VERSION。**
   拓扑指纹在 CI 里对比签入的 golden 值，防止误改。
"""

from __future__ import annotations

from app.infra.graph import NodeSpec

__all__ = [
    "EDGES",
    "GRAPH_NAME",
    "GRAPH_VERSION",
    "NODE_SPECS",
    "RETIRED_NODE_SPECS",
    "CONTROL_NODE_IDS",
    "FANOUT_NODE_IDS",
    "LINEAR_NODE_IDS",
    "CALLBACK_NODE_IDS",
]

GRAPH_NAME = "document_ingestion_pipeline"

#: 任何拓扑变更（增删节点、改边、改 sequence）都必须 bump。
GRAPH_VERSION = "1.0.0"


# ============================================================
# 节点
# ============================================================
#
# 阶段 1-6 线性（每版本一次），7-10 按 RetrievalDefinition 扇出，
# 最后 finalize 扇入。与用户给定的阶段表 1:1 对应，
# 外加一个控制节点 finalize_version。

NODE_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        "ingestion", "接入文件", "linear", 0,
        "解析 File 实体，取出 storage_key 与源文件内容。",
    ),
    NodeSpec(
        "format_detection", "格式识别", "linear", 1,
        "按 suffix / MIME 选择 DocumentParser。",
    ),
    NodeSpec(
        "parsing", "解析", "linear", 2,
        "Loader / Reader：文件内容 -> 规范化前的文本。",
    ),
    NodeSpec(
        "normalization", "归一化", "linear", 3,
        "Processor：编码、换行、空白归一。",
    ),
    NodeSpec(
        "cleaning", "清洗", "linear", 4,
        "Processor：去页眉页脚、水印等噪声。",
    ),
    NodeSpec(
        "metadata_extraction", "元数据抽取", "linear", 5,
        "Processor：抽出文档级元数据，并决定扇出哪些 Retrieval。",
    ),
    NodeSpec(
        "chunking", "切分", "fanout", 6,
        "按 RetrievalDefinition 的 chunker 切分。",
    ),
    NodeSpec(
        "chunk_enrichment", "切分增强", "fanout", 7,
        "按 chunker 配置补全 chunk 的 metadata 与定位信息。",
    ),
    NodeSpec(
        "embedding", "向量化", "fanout", 8,
        "按 RetrievalDefinition 的 embedding 配置生成向量。",
    ),
    NodeSpec(
        "indexing", "写入索引", "fanout", 9,
        "写入 IndexBinding 指向的索引实例。",
    ),
    NodeSpec(
        "finalize_version", "汇总落盘", "control", 10,
        "扇入：写 DocumentVersionRetrievalBinding，"
        "聚合出版本级 processing_status。",
    ),
)


#: 已从当前图移除、但历史运行可能仍有其行的节点。
#: 现在为空 —— 一旦有节点下线，移到这里并标 deprecated，
#: 不要直接删。
RETIRED_NODE_SPECS: tuple[NodeSpec, ...] = ()


# ============================================================
# 边
# ============================================================

EDGES: tuple[tuple[str, str], ...] = (
    ("ingestion", "format_detection"),
    ("format_detection", "parsing"),
    ("parsing", "normalization"),
    ("normalization", "cleaning"),
    ("cleaning", "metadata_extraction"),
    # metadata_extraction -> chunking 是条件边（扇出），
    # 这里记录的是它的目标，供拓扑指纹与前端画图使用。
    ("metadata_extraction", "chunking"),
    ("chunking", "chunk_enrichment"),
    ("chunk_enrichment", "embedding"),
    ("embedding", "indexing"),
    # 扇入：所有分支最终汇到 finalize_version。
    ("indexing", "finalize_version"),
)


# ============================================================
# 派生集合
# ============================================================

LINEAR_NODE_IDS = frozenset(
    s.node_id for s in NODE_SPECS if s.group == "linear"
)

FANOUT_NODE_IDS = frozenset(
    s.node_id for s in NODE_SPECS if s.group == "fanout"
)

CONTROL_NODE_IDS = frozenset(
    s.node_id for s in NODE_SPECS if s.group == "control"
)

#: 交给 callback 兜底覆盖的节点。
#:
#: **刻意排除扇出节点**：callback 拿到的 metadata 里，N 个 Send
#: 分支共享同一个 langgraph_node 名字，无法区分是哪个 retrieval。
#: 硬写只会产生一行无归属的"聚合行"，与 node_span 写的分支行并存，
#: 让节点计数重复、前端出现幽灵节点。
#: 扇出分支的完整覆盖由 node_span 逐分支负责。
CALLBACK_NODE_IDS = LINEAR_NODE_IDS | CONTROL_NODE_IDS
