"""
查询期阶段的注册表。

与 pipeline/registry.py 的关系
------------------------------
写入链的能力目录（PipelineRegistry）由 main.py 在 lifespan 里构造
一次并挂到 app.state —— 因为它要注入 ES / Milvus 的客户端。

查询期这四个阶段**没有外部依赖**：融合、过滤、裁剪都是纯计算，
精排将来会需要模型客户端，届时按同样的方式注入即可。因此这里
不做 app.state 接线，用一个进程内单例（见 get_registry）。

注册方式：反射发现
------------------
与 PipelineRegistry 同一套思路 —— 遍历子类，按各自声明的类型键
注册。新增一个融合算法只需在 fusion.py 里加一个类并给出
fusioner_type，不必改本文件。

不在能力目录里
--------------
规格明确查询期能力（fuse / rerank）不进 /document-processor-capabilities
—— 那是**写入链**的能力目录。这里另开一套，不要混。
"""

from __future__ import annotations

from functools import lru_cache

from app.common.utils import (
    import_package_modules,
    leaf_subclasses,
)

from ..domain.schemas.enum import FusionerType, RerankerType
from .base import Cutter, Fusioner, Reranker, ResultFilter

__all__ = [
    "RetrieveProcessorRegistry",
    "RetrieveStageNotFound",
    "get_registry",
]


class RetrieveStageNotFound(LookupError):
    """
    配置点名了一个没注册的查询期实现。

    与 PipelineRegistry 的 CapabilityNotFound 同样的立场：
    实现缺失时**显式失败**，而不是悄悄跳过该阶段 ——
    "配置写了精排但没精排"是最难排查的一类问题。
    """


class RetrieveProcessorRegistry:
    """
    查询期四阶段的实现表。

    Filter 与 Cutter 不在这里按名字索引 —— 规格里它们的
    Config 没有 type 字段，各自只有一个实现，直接持有。
    """

    def __init__(self) -> None:

        # 非递归 + 跳过 _ 开头的模块：本包的实现都在这一层
        # （fusion.py / rerank.py / filter.py / cut.py），子包留给
        # 将来按类型拆，那时改成 recursive=True 即可。
        import_package_modules(
            __package__, recursive=False, skip_private=True
        )

        self._fusioners = self._collect_fusioners()
        self._rerankers = self._collect_rerankers()
        self._filter = self._collect_single(
            ResultFilter, "ResultFilter"
        )
        self._cutter = self._collect_single(
            Cutter, "Cutter"
        )

    # ========================================================
    # 取用
    # ========================================================

    def fusioner(
            self,
            fusioner_type: FusionerType,
    ) -> Fusioner:

        return self._require(
            self._fusioners, fusioner_type, "Fusioner"
        )

    def reranker(
            self,
            reranker_type: RerankerType,
    ) -> Reranker:

        return self._require(
            self._rerankers, reranker_type, "Reranker"
        )

    @property
    def filter(self) -> ResultFilter:
        return self._filter

    @property
    def cutter(self) -> Cutter:
        return self._cutter

    # ========================================================
    # 发现
    # ========================================================

    @staticmethod
    def _collect_fusioners() -> dict[FusionerType, Fusioner]:

        collected: dict[FusionerType, Fusioner] = {}

        for fusioner_cls in leaf_subclasses(Fusioner, skip_abstract=True):

            fusioner_type = getattr(
                fusioner_cls, "fusioner_type", None
            )

            if fusioner_type is None:
                continue

            collected[fusioner_type] = fusioner_cls()

        return collected

    @staticmethod
    def _collect_rerankers() -> dict[RerankerType, Reranker]:

        collected: dict[RerankerType, Reranker] = {}

        for reranker_cls in leaf_subclasses(Reranker, skip_abstract=True):

            reranker_types = getattr(
                reranker_cls, "reranker_types", ()
            )

            if not reranker_types:
                continue

            # 一个实例服务多个类型 —— 精排实现通常无状态，
            # 没必要按类型各建一份。
            instance = reranker_cls()

            for reranker_type in reranker_types:

                # 一个类型只能有一个实现。冲突时**显式失败**而不是让
                # 后注册的静默顶掉先注册的 —— 那种表现是"某个实现凭空
                # 消失"，没有任何报错，极难查（PipelineRegistry 对
                # processors 也是这么处理的）。
                existing = collected.get(reranker_type)

                if existing is not None:
                    raise RetrieveStageNotFound(
                        f"RerankerType.{reranker_type.name} 同时被 "
                        f"{type(existing).__name__} 与 "
                        f"{reranker_cls.__name__} 声明。"
                        "一个类型只能有一个实现，否则先注册的会被"
                        "静默顶掉。"
                    )

                collected[reranker_type] = instance

        return collected

    @staticmethod
    def _collect_single(
            base: type,
            kind: str,
    ):
        """
        过滤与裁剪各自只有一个实现，多了就是设计出了问题。
        """

        leaves = leaf_subclasses(base, skip_abstract=True)

        if not leaves:
            raise RetrieveStageNotFound(
                f"没有注册任何 {kind} 实现。"
            )

        if len(leaves) > 1:
            raise RetrieveStageNotFound(
                f"{kind} 只应有唯一实现，实际发现 "
                f"{len(leaves)} 个："
                f"{sorted(cls.__name__ for cls in leaves)}。"
                "规格里它没有 type 字段，多实现无法被选中 —— "
                "请合并，或先给规格加 type。"
            )

        return leaves[0]()

    @staticmethod
    def _require(
            table: dict,
            key,
            kind: str,
    ):
        try:
            return table[key]

        except KeyError as exc:
            raise RetrieveStageNotFound(
                f"没有注册 {kind} 实现：{key!r}。"
                f"已注册：{sorted(map(str, table))}。"
            ) from exc


@lru_cache(maxsize=1)
def get_registry() -> RetrieveProcessorRegistry:
    """
    进程内单例。

    四阶段实现无状态、无外部客户端，构造一次即可；反射遍历
    子类没必要每个请求做一遍。测试里要换实现时调
    get_registry.cache_clear()，或在构造 RetrieveProcessor 时
    直接注入自己的 registry。
    """

    return RetrieveProcessorRegistry()
