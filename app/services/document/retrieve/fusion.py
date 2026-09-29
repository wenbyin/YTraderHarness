"""
融合实现：RRF 与加权求和。

两者都按 document_chunk_id 归并 —— 同一个 chunk 被多路召回到是
常态，融合的对象正是"同一份内容被几路、排在第几"。

分数一律归一化后返回（见 base.py 的不变量）。
"""

from __future__ import annotations

from ..domain.schemas import FusionerConfig
from ..domain.schemas.enum import FusionerType
from .base import (
    Fusioner,
    RetrieveCandidate,
    merge_highlights,
    merge_metadata,
    merge_routes,
    normalize_scores,
)

__all__ = [
    "ReciprocalRankFusioner",
    "WeightedSumFusioner",
]


class ReciprocalRankFusioner(Fusioner):
    """
    Reciprocal Rank Fusion。

        score(chunk) = Σ 1 / (rrf_k + rank_in_list)

    只用**排名**，不用原始分 —— 这正是 RRF 的价值：不同召回路的
    分数尺度不可比（BM25 是几十，余弦是 0..1），排名可比。

    rrf_k 越大，排名差异被压得越平（k=60 是常见默认）。
    """

    fusioner_type = FusionerType.RRF

    async def fuse(
            self,
            *,
            query: str,
            result_sets: list[list[RetrieveCandidate]],
            top_k: int,
            config: FusionerConfig,
    ) -> list[RetrieveCandidate]:

        rrf_k = config.rrf_k

        scores: dict = {}
        merged: dict = {}
        routes: dict = {}
        highlights: dict = {}
        metadata: dict = {}

        for result_set in result_sets:

            for rank, candidate in enumerate(
                    result_set, start=1
            ):

                chunk_id = candidate.document_chunk_id

                scores[chunk_id] = (
                    scores.get(chunk_id, 0.0)
                    + 1.0 / (rrf_k + rank)
                )

                # 首次出现时留下来源与 metadata；
                # 后续命中只累加分、并**并进来源列表** ——
                # 覆盖来源会让它变成"最后一路"而不是"全部路"。
                merged.setdefault(chunk_id, candidate)
                merge_routes(routes, candidate)
                merge_highlights(highlights, candidate)
                merge_metadata(metadata, candidate)

        fused = [
            merged[chunk_id].fused(
                score,
                routes[chunk_id],
                highlights.get(chunk_id, {}),
                metadata.get(chunk_id, {}),
            )
            for chunk_id, score in scores.items()
        ]

        fused.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return normalize_scores(fused[:top_k])


class WeightedSumFusioner(Fusioner):
    """
    加权求和。

        score(chunk) = Σ weight(source) * score_in_list

    权重取自 config.weights，键是 retrieval_source
    （即 index_identifier）。没给权重时该路按 1.0 计。

    前提是各路的分数已经落在同一尺度上 —— 这正是管线在召回后
    就按路归一化的原因（见 base.py 的不变量）。

    同一 chunk 被多路命中时累加各路的加权分；来源按**每一次出现**
    各自取其所属路（权重也是按那一路取），最终并成 pipeline_keys
    列出全部路。retrieval_source / pipeline_key 只是首次命中那一路的
    代表值 —— 要判断"被哪几路命中"请读 pipeline_keys。
    """

    fusioner_type = FusionerType.WEIGHTED_SUM

    async def fuse(
            self,
            *,
            query: str,
            result_sets: list[list[RetrieveCandidate]],
            top_k: int,
            config: FusionerConfig,
    ) -> list[RetrieveCandidate]:

        weights = config.weights or {}

        scores: dict = {}
        merged: dict = {}
        routes: dict = {}
        highlights: dict = {}
        metadata: dict = {}

        for result_set in result_sets:

            for candidate in result_set:

                chunk_id = candidate.document_chunk_id

                # 权重按**这一次出现**所属的路取 —— 同一条被两路命中时，
                # 两次出现各自按自己那一路的权重加权，而不是整条按某一路。
                weight = weights.get(
                    candidate.retrieval_source, 1.0
                )

                scores[chunk_id] = (
                    scores.get(chunk_id, 0.0)
                    + weight * candidate.score
                )

                merged.setdefault(chunk_id, candidate)
                merge_routes(routes, candidate)
                merge_highlights(highlights, candidate)
                merge_metadata(metadata, candidate)

        fused = [
            merged[chunk_id].fused(
                score,
                routes[chunk_id],
                highlights.get(chunk_id, {}),
                metadata.get(chunk_id, {}),
            )
            for chunk_id, score in scores.items()
        ]

        fused.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return normalize_scores(fused[:top_k])
