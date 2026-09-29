"""
稀疏向量实现（词项权重）。

它是什么
--------
把文本编码成 **维度索引 -> 权重** 的稀疏映射：

    "量化交易策略"  ->  {13323: 0.41, 11415: 0.38, 61933: 0.52, ...}

三步，全是确定性的、不需要模型：

    1. 分词    tiktoken 的 o200k_base 编码器（与 chunker 度量长度同一套）
    2. 计权重  词频，可配 binary / tf / sublinear_tf
    3. 定维度  **直接用 token id 当维度索引**

为什么不需要哈希
----------------
常见做法是"特征哈希"（hashing trick），把词映射进一个固定空间。这里
不必 —— tiktoken 给出的 token id 本身就是一个稳定的整数词表索引：

    - 跨进程、跨机器、跨版本一致（BPE 词表是固定的）
    - 没有碰撞（哈希会把两个不同的词映射到同一维度，权重叠加成噪声）
    - 维度就是这个编码器的词表大小，是个真实存在的数

⚠️ 顺带记一笔踩坑点：**绝不要用 Python 内置的 `hash()`** 去做这件事。
它按进程随机加盐（PYTHONHASHSEED），同一个词在两次运行里会落到不同
维度 —— 写入时的向量与查询时的向量对不上，检索静默失效且无法复现。
token id 天然绕开了这个陷阱。

它**不是**什么
--------------
它**不是** SPLADE 那类学习式稀疏模型：

    SPLADE    用预训练模型给词表词打分，含**语义扩展**
              （"汽车" 会点亮 "车辆" 这个词项）—— 需要模型与权重文件
    本实现    只认**字面出现**的词，不做语义扩展

所以它的检索质量在 BM25 那一档（词面匹配），不是语义检索。选它是因为
本机 Ollama 没有可用的稀疏编码器，而 ES 的 sparse_vector 与 Milvus 的
稀疏向量都需要**显式的稀疏表示**。

等接入真正的学习式稀疏模型时，新建一个实现、换掉配置里点的键即可 ——
契约（output_kind=SPARSE、返回 dict[int, float]）不用动。

⚠️ dimensions 是索引的物理事实
------------------------------
它等于 o200k_base 的词表大小，会写进 Milvus 的 collection schema。
它做成 ClassVar 而不是配置项，理由是：任何改动都意味着**已有向量全部
作废**（超出新空间的维度索引无处可放），那不该由一个前端下拉框决定。
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.services.document.domain.schemas.enum import EmbeddingOutput
from app.services.document.tokenization import encoding_for

from .base import DocumentEmbedder

__all__ = [
    "LexicalSparseEmbedder",
    "SparseEmbeddingConfig",
]


#: 采用的编码器。**不暴露成配置** —— 它与 dimensions 绑定，理由见模块
#: docstring 末尾。
_ENCODING_MODEL = "o200k_base"

#: o200k_base 的词表大小（实测 encoding.n_vocab == 200019）。
#:
#: 刻意写成字面量而不是 `encoding_for(...).n_vocab`：后者会在**导入时**
#: 触发 tiktoken 加载 —— 包括首次的联网下载词表，那会让能力注册表
#: 的构造变慢甚至离线失败。embed() 里有一道自检保证这个数字没说谎。
_DIMENSIONS = 200019


class SparseEmbeddingConfig(BaseModel):
    """稀疏向量化的参数结构。"""

    model_config = ConfigDict(title="稀疏向量参数")

    weighting: Literal["binary", "tf", "sublinear_tf"] = Field(
        default="sublinear_tf",
        title="词频权重",
        description=(
            "binary = 出现即 1；tf = 原始词频；"
            "sublinear_tf = 1 + log(tf)（默认）。"
            "长文本里同一个词出现十次，不代表它的重要性是出现一次的十倍，"
            "次线性压缩就是为此。"
        ),
    )

    lowercase: bool = Field(
        default=True,
        title="转小写",
        description=(
            "分词前统一转小写，让 Trading 与 trading 落进同一个词项。"
            "英文场景几乎总是要开；中文不受影响。"
        ),
    )

    normalize: bool = Field(
        default=True,
        title="L2 归一化",
        description=(
            "对最终的稀疏向量做 L2 归一化。开启后长 chunk 不会仅因为"
            "词多就在相似度上占优，索引与查询两侧的算法也统一了。"
        ),
    )


class LexicalSparseEmbedder(DocumentEmbedder):
    """
    词面稀疏编码器。

    ⚠️ 它直接继承 DocumentEmbedder，**没有中间层** —— 与
    embedder/ollama.py 那个四层结构不一样。原因是它没有"族"可归：
    Ollama 那一族共享的是连接方式、批处理、超时，而这些在本实现里
    一个都不存在（纯本地计算、无网络、无批次概念）。

    等将来出现第二个同族实现（例如 SPLADE via ONNX），再抽出共性的
    中间层也不迟 —— 现在凭空造一层，里面会是空的。
    """

    output_kind = EmbeddingOutput.SPARSE

    dimensions: ClassVar[int | None] = _DIMENSIONS

    name = "TEXT_LEXICAL_SPARSE_EMBEDDER"

    config_model = SparseEmbeddingConfig

    # ========================================================
    # 执行
    # ========================================================

    async def embed(
            self,
            texts: list[str],
            *,
            config: dict[str, Any],
    ) -> list[dict[int, float]]:

        if not texts:
            return []

        params = self.config_model.model_validate(config or {})

        encoding = encoding_for(_ENCODING_MODEL)

        # 维度自检：tiktoken 升级换了词表的话，写进索引的维度就与这里
        # 声明的不一致了。宁可在这里炸，也不要往一个 schema 对不上的
        # collection 里写数据。
        if encoding.n_vocab != self.dimensions:
            raise RuntimeError(
                f"{self.get_key()}: 声明维度 {self.dimensions}，但 "
                f"{_ENCODING_MODEL} 的实际词表是 {encoding.n_vocab} —— "
                "索引 schema 会与向量对不上，请同步更新 _DIMENSIONS "
                "并重建索引。"
            )

        # 分词是 CPU 密集的同步计算。放到线程里跑，别占着事件循环 ——
        # 扇出分支是并发的，在这里阻塞会把其它分支一起卡住。
        return await asyncio.to_thread(
            self._encode_all, texts, params, encoding,
        )

    # ========================================================
    # internals
    # ========================================================

    @classmethod
    def _encode_all(
            cls,
            texts: list[str],
            params: SparseEmbeddingConfig,
            encoding,
    ) -> list[dict[int, float]]:
        return [
            cls._encode_one(text, params, encoding)
            for text in texts
        ]

    @staticmethod
    def _encode_one(
            text: str,
            params: SparseEmbeddingConfig,
            encoding,
    ) -> dict[int, float]:
        """
        单条文本 -> {token_id: weight}。

        空文本返回**空字典**而不是全零向量 —— 稀疏表示的"零"就是
        没有键，展开成稠密会把 20 万维的零数组带上路。
        """

        if not text:
            return {}

        if params.lowercase:
            text = text.lower()

        counts = Counter(encoding.encode(text))

        if not counts:
            return {}

        if params.weighting == "binary":
            weights = {token_id: 1.0 for token_id in counts}

        elif params.weighting == "tf":
            weights = {
                token_id: float(count)
                for token_id, count in counts.items()
            }

        else:  # sublinear_tf
            weights = {
                token_id: 1.0 + math.log(count)
                for token_id, count in counts.items()
            }

        if params.normalize:
            norm = math.sqrt(
                sum(weight * weight for weight in weights.values())
            )
            if norm > 0.0:
                weights = {
                    token_id: weight / norm
                    for token_id, weight in weights.items()
                }

        return weights

