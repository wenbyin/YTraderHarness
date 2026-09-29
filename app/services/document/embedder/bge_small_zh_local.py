"""
BGE-small-zh-v1.5 的**本地**向量化实现（不再经 Ollama）。

为什么从 Ollama 换到本地
------------------------
原来走 `langchain_ollama.OllamaEmbeddings`。关键事实：那是**纯 HTTP
客户端**，库里没有任何词表/tokenizer 代码 —— 分词发生在 Ollama 服务端
（llama.cpp）。这带来两个后果：

    1. Python 侧拿不到模型的词表，无法在**模型真实 token 空间**里数长度
    2. chunker 只好用 tiktoken 的 o200k_base 量长度，而那是**另一套词表**

第 2 条的后果见本文件末尾的"512 上限"一节。本地加载之后，词表就在
进程里，两件事一起解决。

换模型 = 换键（已执行）
----------------------
`name` 是**新键** `TEXT_BGE_SMALL_ZH_LOCAL_EMBEDDER`，旧的
`TEXT_BGE_SMALL_ZH_EMBEDDER` 随 Ollama 实现一起删掉了。

这不是洁癖：Ollama 里那份是 `quentinz/bge-small-zh-v1.5`（**GGUF
量化版**），这里是 `BAAI/bge-small-zh-v1.5`（原版）。量化会改变向量
输出，同一个键下混两种向量空间 = Milvus 里新旧 chunk 不可比，而且
**不报错**，只表现为召回率悄悄下降。旧键会冻结进
PipelineConfigSnapshot，历史运行靠它解释"当时用的是哪个模型" ——
沿用旧键等于让历史快照撒谎。

代价是已有向量全部作废，必须重建索引（重跑
`POST /document-versions/{vid}/reprocess`）。

512 上限：模型词表 vs tiktoken
------------------------------
模型 `max_position_embeddings = 512`，即单次输入最多 512 个**它自己
词表的** token。超出的部分被**静默截断**：不报错、向量照常产出、
维度照常正确，只是 chunk 的尾巴从来没有被向量化过，检索时永远命
中不到。

而 chunker 默认 `size = 1200`，量的是 **tiktoken** token。两者不是
一回事 —— 实测中文下 bge 词表要比 tiktoken 多吃 16%~33%：

    中文正文   tiktoken 63  ->  bge 73   (1.16x)
    中文标题   tiktoken 49  ->  bge 65   (1.33x)

所以 1200（tiktoken 口径）在模型看来是 1400~1600 token，**接近上限的
三倍**，约 60% 的内容进不了向量。

修复不在本文件，在 chunker：用 `length_function="model"` 按**本模型
词表**量长度，把 size 控制在 512 以内（见 chunker/txt.py）。

查询侧必须同模型 + 加指令前缀
------------------------------
BGE 系列推荐**查询**侧加指令前缀：

    为这个句子生成表示以用于检索相关文章：

文档侧（写入期）**不加**。所以 `query_instruction` 只声明在类上，
由基类的 `embed_query()` 施加 —— 本文件不做任何前缀处理（见下面
`_build_model` 里为什么把 FlagModel 的 instruction 参数显式关掉）。
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.common.model_factory import get_model_factory

from ..const import BGE_SMALL_ZH_LOCAL_MODEL
from ..domain.schemas.enum import EmbeddingOutput
from ..tokenization import tokenizer_for
from .base import DocumentEmbedder

__all__ = ["LocalBgeSmallZhEmbedder"]

logger = logging.getLogger(__name__)


class LocalBgeSmallZhConfig(BaseModel):
    """本地 bge-small-zh 的参数。"""

    model_config = ConfigDict(title="嵌入参数")

    batch_size: int = Field(
        default=16,
        ge=1,
        le=512,
        title="批大小",
        description=(
            "每次前向送进模型的文本条数。CPU 上前向是主要开销，"
            "批量太小会让每次调用的固定开销被放大 N 倍。"
        ),
    )

    normalize: bool = Field(
        default=True,
        title="L2 归一化",
        description=(
            "对输出向量做 L2 归一化。开启后余弦相似度等价于点积，"
            "索引与查询两侧的算法不必再约定用哪个。"
        ),
    )

    max_length: int = Field(
        default=512,
        ge=1,
        le=512,
        title="单条最大 token 数",
        description=(
            "**模型自己词表**的 token 上限，不是 tiktoken 的。"
            "上界 512 是模型的 max_position_embeddings，改大没有意义"
            "（位置编码就那么多），只会让人误以为不会截断了。"
            "\n\n"
            "超出会被**静默截断**。要避免它，改 chunker 的 "
            "length_function='model' 并按本词表设 size。"
        ),
    )


# ============================================================
# 模型加载
# ============================================================
#
# 加载本身交给 app/common/model_factory.py：那里负责"启动时异步预热、
# 使用时按 key 取、没就绪就阻塞"。
#
# 原来这里是"懒加载 + 模块级字典 + 锁"三件套。它解决了"每个请求加载
# 一次"，但没解决**第一个请求**：那一个请求要独自承担模型加载时间。
# 挪到工厂之后加载与请求解耦，服务起来的瞬间就开始加载。


def register_preload(factory: Any) -> None:
    """把默认向量模型登记到工厂，由启动阶段统一预热。"""

    factory.register(
        BGE_SMALL_ZH_LOCAL_MODEL,
        lambda: _load_bundle(BGE_SMALL_ZH_LOCAL_MODEL),
    )


def _load_bundle(model_name: str):
    """
    加载向量模型**和它自己的词表**。

    两者一起加载是刻意的：chunker 的 length_function="model" 要用词表
    量长度，而它跑在写入管线里、不在请求路径上 —— 等那时才加载会让
    第一次 reprocess 莫名其妙地慢一截，而且是在一个和"模型加载"看起来
    无关的地方慢。

    顺带也把 tokenization.tokenizer_for 的 lru_cache 预热了。
    """

    model = _build_model(model_name)

    tokenizer_for(model_name)

    return model


def _build_model(model_name: str):
    """
    真正构造模型。同步 —— 调用方负责 to_thread。

    走 FlagEmbedding 的 FlagModel：与 retrieve/rerank.py 选 FlagReranker
    是同一个理由 —— 它是 bge 系列原作者实现，比自己拼 transformers 少了
    "池化方式（CLS vs mean）、要不要归一化"这些**容易写错又不会报错**的
    细节。写错的症状是向量照常产出、维度照常正确、召回率悄悄下降。
    """

    from FlagEmbedding import FlagModel

    # query_instruction_for_retrieval 显式关掉：指令前缀由基类的
    # embed_query() 施加（见类上的 query_instruction）。这里再给一份
    # 就会**加两次**，而加两次同样不报错，只是向量被污染。
    #
    # normalize_embeddings 也关掉，改由 _l2_normalize 在 Python 侧做。
    # 两个理由：
    #   1. 它是**构造函数**参数，而模型按名字缓存 —— 构造参数没法随
    #      每次调用的 config 变。放在这里会让 normalize 变成"首次加载
    #      时定死"，config 里那个开关就成了摆设。
    #   2. 512 个 float 的归一化在 Python 里是微不足道的开销。
    #
    # use_fp16=False：CPU 上不支持 fp16，交给库去猜不如显式声明。
    return FlagModel(
        model_name,
        query_instruction_for_retrieval=None,
        use_fp16=False,
        normalize_embeddings=False,
    )


def _l2_normalize(vector: list[float]) -> list[float]:
    """
    L2 归一化。零向量原样返回（归一化会除零）。

    与旧 Ollama 实现里的同名函数语义一致 —— 归一化后余弦相似度退化
    成点积，索引与查询两边的算法才能一致。
    """

    norm = math.sqrt(sum(value * value for value in vector))

    if norm == 0.0:
        return list(vector)

    return [value / norm for value in vector]


class LocalBgeSmallZhEmbedder(DocumentEmbedder):
    """BGE-small-zh-v1.5：中文小模型，512 维，输入上限 512 token。"""

    #: embedder 的身份就是**模型**，不是文件格式。一旦发布**永不复用**，
    #: 它会冻结进 PipelineConfigSnapshot，历史运行靠它解释当时用的是
    #: 哪个模型。换模型 = 换键。
    name = "TEXT_BGE_SMALL_ZH_LOCAL_EMBEDDER"

    #: 本地目录（或 HF 仓库名）。换模型必须同时换键（见上）。
    model_name: ClassVar[str] = BGE_SMALL_ZH_LOCAL_MODEL

    #: Ollama 走常规 embedding 模型，产出定长稠密向量。
    output_kind = EmbeddingOutput.DENSE

    #: 512 维。建 Milvus collection 需要它在调用前就可得，
    #: 因此是类属性字面量而不是运行时探测出来的值。
    dimensions = 512

    #: 度量输入长度时用**本模型自己的词表**（见基类上的说明）。
    #: 这就是本地加载换来的那件事 —— 走 Ollama 时词表在服务端，
    #: Python 侧只能退而用 tiktoken，于是 chunker 按 1200 切、
    #: 模型按 512 截，后 60% 静默丢失。
    tokenizer_id = BGE_SMALL_ZH_LOCAL_MODEL

    #: 查询侧指令前缀。文档侧不加 —— 只加在 embed_query 上，
    #: 写入期走的 embed() 不受影响。
    query_instruction = "为这个句子生成表示以用于检索相关文章："

    config_model = LocalBgeSmallZhConfig

    async def embed(
            self,
            texts: list[str],
            *,
            config: dict[str, Any],
    ) -> list[list[float]]:
        """把一批文本向量化，返回与 texts **等长且等序**的向量列表。"""

        if not texts:
            return []

        params = self._params(config)

        # 走进程级工厂取模型：服务进程里它在启动阶段就预热好了，这里
        # 通常直接命中。loader 只在**没登记过**时兜底 —— tools/ 下的
        # 脚本不走启动流程，仍要能用。
        model = await get_model_factory().get(
            self.model_name,
            loader=lambda: _load_bundle(self.model_name),
        )

        # 前向是同步且 CPU 密集的。放到线程里跑，别占着事件循环 ——
        # 扇出分支是并发的，在这里阻塞会把其它分支一起卡住。
        return await asyncio.to_thread(
            self._encode_all, model, texts, params,
        )

    @staticmethod
    def _encode_all(
            model: Any,
            texts: list[str],
            params: LocalBgeSmallZhConfig,
    ) -> list[list[float]]:
        # ⚠️ 只传 encode() 真正接受的三个参数。
        #
        # FlagModel.encode 的签名是
        #     (sentences, batch_size, max_length, convert_to_numpy, **kwargs)
        # —— 那个 **kwargs 会一路透传到 tokenizer.pad()。多给一个它不
        # 认识的键（比如 normalize_embeddings，它其实是**构造函数**
        # 参数）不会得到"参数被忽略"，而是 tokenizer.pad() 抛
        # TypeError。签名上没有的键一个都别加。
        vectors = model.encode(
            texts,
            batch_size=params.batch_size,
            max_length=params.max_length,
        )

        # FlagModel.encode 返回 numpy 二维数组。转成 list 是必须的：
        # 下游要 JSON 序列化进 Milvus / 做维度校验，numpy 标量在那些
        # 地方会以一个"看起来是 float 但不是 float"的形态出错。
        result = [
            [float(value) for value in vector]
            for vector in vectors
        ]

        if params.normalize:
            result = [_l2_normalize(vector) for vector in result]

        return result

    def _params(
            self,
            config: dict[str, Any],
    ) -> LocalBgeSmallZhConfig:
        """
        配置 -> 参数对象。

        未知键会被 config_model 忽略（pydantic 默认行为）—— 写入期与
        查询期读的是同一份 retrieval 配置，里面还混着召回参数，不能
        因为多了几个键就报错。
        """

        return self.config_model.model_validate(config or {})

