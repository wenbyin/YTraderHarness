"""
Chunk 增强能力（阶段 8，role=chunk_enrich）。

    base.py     契约 —— DocumentChunkEnricher
    common.py   通用文本增强 —— TEXT_COMMON_CHUNK_ENRICHMENT

为什么是独立包而不是 policy/
----------------------------
与 cleaner / chunker / embedder 同一个形状：能力实现住在自己的包里，
契约是该包的 base.py。这样 `CAPABILITY_PACKAGES` 一加就能被反射发现。

原契约住在 policy/enricher.py —— 而 policy 既不在 `CAPABILITY_PACKAGES`
里，`ChunkEnricher` 也不是 `DocumentProcessor` 的子类，两条加起来使得
它**从未被注册过**。事件经过见 base.py 的说明。

增强做什么、不做什么
--------------------
做：把文档级元数据（标题、业务标识、来源）与定位信息继承到每条 chunk
    的 `metadata` / `locator` 上。

不做：改 `content`。那会破坏唯一约束与 content_hash 的一致性，而且
    不报错。想改善**向量**召回应当拼给 embedder 的文本，不是这里。
"""

from .base import DocumentChunkEnricher
from .common import TxtCommonChunkEnricher

__all__ = [
    "DocumentChunkEnricher",
    "TxtCommonChunkEnricher",
]
