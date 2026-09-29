"""
查询期处理链（RetrieveProcessor）。

对应 app/services/document/document_retrieve.yaml。

    base.py       四阶段契约 + RetrieveCandidate + 分数归一化
    fusion.py     融合实现（rrf / weighted_sum）
    filter.py     过滤实现（分数阈值 / 归属 / metadata）
    rerank.py     精排实现（当前为占位降级实现）
    cut.py        裁剪实现（top_k / 正文长度）
    registry.py   按类型键反射注册
    processor.py  RetrieveProcessor：配置解析 + 一次装配的执行

与写入链（pipeline/）的分工
---------------------------
pipeline/ 负责"文件 -> Chunk -> Index"，是写入期。
本包负责"查询 -> 候选 -> 融合 -> 过滤 -> 精排 -> 裁剪"，是查询期。

两者都通过 PipelineRegistry 拿到 IndexBackend，但**不共用**
ProcessorRole 那套能力目录 —— 规格明确查询期能力不进
/document-processor-capabilities。

一次调用 = 一轮
---------------
本包不包含 Recaller，也不做查询改写、扩召回、重试或 agent loop。
候选集由服务层内联调用 IndexBackend.search 得到后传进来
（规格：Recaller 是内部细节，不通过公开 API 暴露）。
"""

from .base import (
    Cutter,
    Fusioner,
    RetrieveCandidate,
    Reranker,
    ResultFilter,
    normalize_scores,
)
from .processor import RetrieveProcessor
from .registry import (
    RetrieveProcessorRegistry,
    RetrieveStageNotFound,
    get_registry,
)

__all__ = [
    "Cutter",
    "Fusioner",
    "RetrieveCandidate",
    "RetrieveProcessor",
    "RetrieveProcessorRegistry",
    "RetrieveStageNotFound",
    "Reranker",
    "ResultFilter",
    "get_registry",
    "normalize_scores",
]
