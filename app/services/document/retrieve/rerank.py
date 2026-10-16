"""
精排实现。

现状
----
    FlagEmbeddingReranker   cross-encoder 精排（BAAI/bge-reranker-v2-m3）—— 真实现
    PassthroughReranker     占位，服务 llm / custom

为什么要真正的 cross-encoder
---------------------------
融合（RRF）有个**结构性偏差**：只被一路命中的 chunk，无论在那一路里
排多高，都赢不过任何一条被两路命中的 ——

    两路命中 ≈ 1/(k+65) + 1/(k+70) ≈ 0.030
    单路命中 ≈ 1/(k+65)            ≈ 0.015

实测踩到过：目标 chunk 在关键词那一路排第 5，却进不了最终 top-10。

**用查询/文档各自的向量算余弦解决不了这个问题** —— 那是 bi-encoder，
两边分开编码，本质上是把稠密那一路重算一遍；而目标 chunk 恰恰是稠密
没选中的那个，等于用同一把尺子再量一次。

cross-encoder 把 query 与正文**拼在一起**过一遍模型，是**另一路信号**，
与"被几路命中"无关。

为什么走 FlagEmbedding 而不是 Ollama
-----------------------------------
bge-reranker-v2-m3 是**序列分类**模型，不是生成模型也不是嵌入模型 ——
Ollama 服务不了它（实测 /api/chat 与 /api/embed 都让 llama-server
直接崩，/api/rerank 端点不存在）。所以只能本地加载。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from app.common.model_factory import get_model_factory

from ..const import BGE_RERANKER_MODEL
from ..domain.schemas import RerankerConfig
from ..domain.schemas.enum import RerankerType
from .base import RetrieveCandidate, Reranker, normalize_scores

__all__ = [
    "FlagEmbeddingReranker",
    "PassthroughReranker",
]

logger = logging.getLogger(__name__)


#: 送进精排的候选条数上限。
#:
#: cross-encoder 是 **O(N) 次前向**（每条候选都要和 query 拼一次过模型），
#: 不像向量那样一次编码就能比。融合后的候选池可能有几百条，全打分很慢。
#: 只取前若干条 —— 融合已经给了粗排，精排是在它之上精修，不是重新召回。
_MAX_CANDIDATES = 200

#: 每条候选截断到多少字。
#:
#: 模型输入上限 8192 token，但那是给"整篇文档"用的极端值；精排要的是
#: 相关性判断，前几百字足够看出"这条讲的是不是这件事"。截断同时是
#: 延迟控制 —— 30 条 × 2000 字会明显拖慢。
_MAX_CHARS = 512


class FlagEmbeddingReranker(Reranker):
    """
    用 BAAI/bge-reranker-v2-m3（cross-encoder）做精排。

    模型**由进程级工厂提供**
    ----------------------
    加载 2.3GB 的模型要好几秒、占几百 MB 内存。每个请求加载一次会让
    首次延迟不可接受，也会把内存打爆。

    所以加载与取用都交给 app/common/model_factory.py：服务启动时它就
    在后台预热（见 register_preload），这里只按 key 取。没就绪时工厂
    会阻塞并打日志，而不是在这里再写一套懒加载。

    工厂那个 `loader` 参数只是**兜底**：tools/ 下的脚本不走启动流程，
    没有预热，靠它现场加载。

    同步 API 必须包 to_thread
    ------------------------
    FlagReranker.compute_score 是同步的、CPU 密集的。直接在事件循环里
    调会把整个进程卡住 —— 与 pymilvus 那边同样的理由（见 index/milvus.py）。
    """

    #: 只服务 cross_encoder。llm / custom 仍归占位实现 ——
    #: 一个类型只能有一个实现，两个都声明会互相静默覆盖。
    reranker_types = (RerankerType.CROSS_ENCODER,)

    async def rerank(
            self,
            *,
            query: str,
            candidates: list[RetrieveCandidate],
            top_n: int,
            config: RerankerConfig,
            degraded: dict[str, str] | None = None,
    ) -> list[RetrieveCandidate]:

        if not candidates:
            return []

        # 打分窗口 = 调用方要多少（top_n 的语义就是"送进精排的条数"），
        # 但仍受成本上限约束 —— 传一个大 top_n 不能把开销顶穿。
        limit = min(top_n, _MAX_CANDIDATES)

        selected = candidates[:limit]

        # 超出部分接在后面：它们没被精排过，但**不能丢** ——
        # 精排只该改变顺序，不该变成二次召回。截断在下游（cutter.top_k）。
        rest = candidates[limit:]

        scores = await self._score(query, selected, config)

        if scores is None:
            # 降级：原序返回、**不变长也不变短**。warning 已在 _score 里记过。
            if degraded is not None:
                degraded["reason"] = "本地精排模型不可用（见 warning 日志）"

            return list(candidates)

        # 没被打分的按 0 处理（排在有分的后面），而不是丢掉。
        scored = list(zip(selected, scores))

        scored.sort(key=lambda pair: pair[1], reverse=True)

        ranked = [
            candidate.normalized(score)
            for candidate, score in scored
        ]

        # ⚠️ **不切片。** 这里原本是 `(ranked + rest)[:top_n]` —— 那让精排
        # 变成了**第二次裁剪**：cutter.top_k=30 的一轮里 top_n=10 会把结果
        # 压到 10 条，排在 11~30 的候选再也回不来。而"精排只改变顺序"是
        # 本模块自己写下的规矩（见上面 rest 那段注释），最终条数归
        # `cutter.top_k` 管。
        return normalize_scores(ranked + list(rest))

    async def _score(
            self,
            query: str,
            candidates: list[RetrieveCandidate],
            config: RerankerConfig,
    ) -> list[float] | None:
        """
        给每条候选打分，返回与 candidates **等长且等序**的分数。
        失败返回 None（调用方降级为原序）。
        """

        pairs = [
            [query, (candidate.content or "")[:_MAX_CHARS]]
            for candidate in candidates
        ]

        try:
            model_name = resolve_model_name(config.model)

            # 走进程级工厂取模型：服务进程里它在启动阶段就预热好了，
            # 这里通常直接命中。loader 只在**没登记过**时兜底 ——
            # tools/ 下的脚本不走启动流程，仍要能用。
            model = await get_model_factory().get(
                model_name,
                loader=lambda: _build_model(model_name),
            )

            # 同步且 CPU 密集：必须挪出事件循环，否则整个进程卡住。
            raw = await asyncio.to_thread(
                model.compute_score,
                pairs,
                # 过 sigmoid 落到 [0,1] —— 管线的不变量要求分数已归一化
                # （见 base.py）。不过的话是原始 logit，可能为负、量级
                # 也不固定，下游的 score_threshold 会失义。
                normalize=True,
            )

        except Exception as exc:
            logger.warning(
                "cross-encoder 精排失败，按融合顺序降级返回：%s: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return None

        scores = _as_list(raw, len(candidates))

        if scores is None:
            logger.warning(
                "cross-encoder 返回的分数条数与候选对不上"
                "（期望 %d，拿到 %r），按融合顺序降级返回。",
                len(candidates),
                raw,
            )
            return None

        return scores


def resolve_model_name(configured: str | None) -> str:
    """
    决定加载哪一个模型。

    config.model **只在真能加载时才被采纳**，否则回落到常量：

        G:\\models\\bge-reranker-v2-m3   本地目录    -> 用
        BAAI/bge-reranker-v2-m3         限定仓库名  -> 用
        bge-reranker-v2-m3              裸名字      -> 忽略，回落

    裸名字是个**会联网的陷阱**：huggingface_hub 把它当成"用户或组织
    名"，拼出 huggingface.co/bge-reranker-v2-m3/resolve/main/config.json
    —— 少了 BAAI/ 那一段 —— 于是不走本地，直接发 HEAD 出去。国内必然
    撞 SSL UNEXPECTED_EOF，再按 1+2+4+8+16 秒重试 5 次才抛错。

    实测踩到过：OpenAPI 示例里写的正是裸名，照着抄进请求就中招，
    而常量明明已经指向本地路径了。
    """

    if not configured:
        return BGE_RERANKER_MODEL

    # 本地目录要**先判**：Windows 路径写成正斜杠时也含 "/"，
    # 若先判仓库名会把它误当成线上模型。
    if os.path.isdir(configured):
        return configured

    # 带 / 的是限定仓库名，属于调用方明确要用线上模型，放行。
    if "/" in configured:
        return configured

    logger.warning(
        "精排 model=%r 既不是本地目录，也不是限定仓库名（缺『组织/"
        "模型』那一段），照它加载会去联网解析并卡在 SSL 上。"
        "已改用 %r。",
        configured,
        BGE_RERANKER_MODEL,
    )

    return BGE_RERANKER_MODEL


def _as_list(raw: Any, count: int) -> list[float] | None:
    """
    把 compute_score 的返回统一成 list[float]。

    它**只传一对时返回标量、多对时返回列表** —— 这个不一致必须在这里
    抹平，否则 count=1 的查询会静默走错分支。

    条数对不上直接返回 None 而不是截断／补齐：错位不会报错，只会让
    分数挂到别的 chunk 上，那是比"没精排"更糟的结果。
    """

    if isinstance(raw, (int, float)):
        values = [float(raw)]

    elif isinstance(raw, (list, tuple)):
        values = [float(value) for value in raw]

    else:
        return None

    if len(values) != count:
        return None

    return values


# ============================================================
# 模型加载
# ============================================================
#
# 加载本身交给 app/common/model_factory.py：那里负责"启动时异步预热、
# 使用时按 key 取、没就绪就阻塞"。
#
# 原来这里是"懒加载 + 模块级字典 + 锁"三件套。它解决了"每个请求加载
# 一次"，但没解决**第一个请求**：那一个请求要独自承担 2.3GB 模型的
# 加载时间。挪到工厂之后加载与请求解耦，服务起来的瞬间就开始加载。


def register_preload(factory: Any) -> None:
    """
    把默认精排模型登记到工厂，由启动阶段统一预热。

    登记的是 BGE_RERANKER_MODEL 这个默认模型；config.model 传了别的
    值时不在这里 —— 那种情况由调用方带 loader 现场登记（见 _score），
    因为启动阶段并不知道运行时会传什么。
    """

    factory.register(
        BGE_RERANKER_MODEL,
        lambda: _build_model(BGE_RERANKER_MODEL),
    )


def _build_model(model_name: str):
    """
    真正构造模型。同步 —— 调用方负责 to_thread。

    走 FlagEmbedding 的 FlagReranker：它就是 bge-reranker 系列的原作者
    实现，比自己拼 transformers 少了"池化方式、分类头取哪一维、
    归一化"这些容易写错又不会报错的细节。
    """

    from FlagEmbedding import FlagReranker

    # fp16 在 CPU 上不支持，交给调用方按设备决定更稳妥 ——
    # 显式 False 而不是让库猜。
    return FlagReranker(model_name, use_fp16=False)


class PassthroughReranker(Reranker):
    """
    占位精排：保持融合给出的顺序、**原样返回**（不截断）。

    现在只服务 **custom** —— `llm` 已经由 `rerank_llm.py` 的
    `LlmReranker` 真正实现（一个类型只能有一个实现，两个都声明会互相
    静默覆盖，注册表会在装配期抛错）。

    它**不改变分数**：分数已在融合阶段归一化，这里再排一次序没有信息
    增益，反而会掩盖"精排其实没跑"这件事。

    ⚠️ 也**不截断**。原本这里返回 `candidates[:top_n]`，与
    `FlagEmbeddingReranker` 的同一个毛病：精排不该顺手决定最终条数 ——
    那由 `cutter.top_k` 定。一个没实现的精排偷偷把结果裁短，比它什么都不
    做更难发现（页面上只是"结果变少了"）。
    """

    reranker_types = (RerankerType.CUSTOM,)

    async def rerank(
            self,
            *,
            query: str,
            candidates: list[RetrieveCandidate],
            top_n: int,
            config: RerankerConfig,
            degraded: dict[str, str] | None = None,
    ) -> list[RetrieveCandidate]:

        logger.warning(
            "精排 %s 未实现，按原序降级返回。model=%s candidates=%d "
            "top_n=%d —— 接入请实现 rerank.py 的一个 Reranker 子类"
            "并注册。",
            config.type,
            config.model,
            len(candidates),
            top_n,
        )

        # 报降级而不是"跑过了"：这一段确实**没有**精排（配置点名了一个
        # 没实现的类型）。不报的话，宏观统计会把它读成"精排跑了但没能
        # 改善排序"，而真相是"这个类型根本没人实现"。
        if degraded is not None:
            degraded["reason"] = f"精排类型 {config.type} 未实现（占位实现）"

        return list(candidates)
