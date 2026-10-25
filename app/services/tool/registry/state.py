"""
显式禁用的 Tool：进程内的只读快照。

为什么要缓存
-----------
这份数据只在管理接口写入，而读发生在**每个 run 的工具解析**上，
并且主 agent / planner / skill 三条路共用同一次解析。每轮查一次库
没有意义 —— 在内存里存一个 set，让那三条路都变成零 IO。

语义
----
集合里只有**显式禁用**的 tool_key。不在集合里 = 启用。
库里也是这个语义：有行 = 禁用，没有行 = 启用
（见 models/sqlalchemy/tool_definition_state.py）。

失效
----
写路径**提交成功后**调用 `invalidate()`，把快照置空；下一次
`tool_keys_for_capabilities` 发现未加载才重新查库。之所以不在写
路径里直接改 set：写事务可能回滚，而"提交后才失效"由
`ServiceContext.post_commit` 保证。

⚠️ 进程级。`run_server.py --instances N` 是 N 个**独立进程**，
各有一份快照；PATCH 只失效收到请求的那个进程。多实例部署要么
接受"其他进程重启后生效"，要么用 Redis 广播失效，或给这里加一个
短 TTL —— 三种都行，但必须在部署拓扑确定时选一种。

⚠️ 与 `ToolRegistry` 的关系
--------------------------
只是同住一个包，**不并入** `ToolRegistry`。后者是"代码里声明了
哪些工具"，被发现器与同步消费；把 DB 状态塞进去会让 `all()` /
`keys()` 同时混两种事实，而注册表本身也不该有 DB 生命周期。
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "DisabledToolCache",
    "get_disabled_tool_cache",
]


class DisabledToolCache:
    """只装禁用 key 的进程级快照。`None` 表示还没加载。"""

    def __init__(self) -> None:
        self._disabled: frozenset[str] | None = None

        #: 每次 invalidate 自增。加载路径先记下它，写入快照时再比对：
        #: 若期间有写路径提交并失效，这次读到的就是**旧值**，不能写回。
        #: 没有这道闸，一个与 PATCH 并发的加载会把刚禁用的工具又放出去。
        self._generation = 0

    @property
    def loaded(self) -> bool:
        return self._disabled is not None

    @property
    def generation(self) -> int:
        return self._generation

    def replace(
            self,
            keys: Iterable[str],
            *,
            generation: int | None = None,
    ) -> bool:
        """
        用一批全量 key 覆盖快照。

        generation 不是 None 时，只有与当前代际一致才写入 ——
        返回是否真的写进去了。调用方据此决定要不要重读。
        """

        if generation is not None and generation != self._generation:
            return False

        self._disabled = frozenset(keys)
        return True

    def invalidate(self) -> None:
        """丢弃快照。下一次读取时由服务层重新查库。"""

        self._disabled = None
        self._generation += 1

    def clear(self) -> None:
        """与 invalidate 相同；名字更明确地留给测试。"""

        self.invalidate()

    def is_disabled(self, tool_key: str) -> bool:
        """
        是否被显式禁用。

        ⚠️ 未加载时返回 False（fail-open）。调用方必须先确认
        `loaded` —— 正常运行路径由 `ToolQueryServiceImpl` 的
        `_ensure_disabled_keys` 保证。
        """

        return (
            self._disabled is not None
            and tool_key in self._disabled
        )

    def disabled_keys(self) -> frozenset[str]:
        """当前快照。未加载时是空集合（语义见 is_disabled 的警告）。"""

        return self._disabled or frozenset()


#: 进程级单例。与 get_tool_registry() 同一形状。
_cache = DisabledToolCache()


def get_disabled_tool_cache() -> DisabledToolCache:
    """取进程级禁用快照。"""

    return _cache
