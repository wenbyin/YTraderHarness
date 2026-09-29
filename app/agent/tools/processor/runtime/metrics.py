"""
工具调用的指标收集。

口径
----
    calls        调用**次数**（含被拒绝、含失败）
    errors       其中失败的次数（工具真的跑了但失败）
    denied       其中被拦截器拒绝的次数（工具根本没跑）
    duration     只统计**真的跑了**的那些调用的耗时

为什么 denied 单列而不算进 errors
--------------------------------
"调用失败率 20%"这句话在两种情况下要做的事完全不同：工具在报错
（要去看后端），还是审批把大量调用拦下了（要去看策略）。混成一个
数字之后，这两种情况在监控上长得一模一样。

为什么耗时不含 denied
--------------------
被拒绝的调用没有"耗时"这回事 —— 它可能只花了 0.1 ms（一次字典
查表）。把它算进去会把平均耗时拉低，于是"接口变慢了"这个信号
被稀释。被拒绝的占比要看覆盖率，不是看耗时。

⚠️ 进程内、重启清零
------------------
项目没有 Prometheus（依赖里没有）。真要做持久化监控，把
`snapshot()` 的输出定期推给 Pushgateway，或者把本模块换成官方
客户端 —— 处理器只依赖 `record()`，换实现不用动它们。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime

__all__ = [
    "ToolMetrics",
    "ToolMetricsRegistry",
    "get_tool_metrics",
]


@dataclass
class ToolMetrics:
    """一个工具的累计指标。"""

    tool_key: str

    calls: int = 0
    errors: int = 0
    denied: int = 0

    #: 只累计**真的执行了**的调用（见模块 docstring）。
    total_duration_ms: float = 0.0
    min_duration_ms: float | None = None
    max_duration_ms: float | None = None

    last_called_at: datetime | None = None

    #: 异常类型名 -> 次数。带类型的错误分布比一个总数有用得多。
    errors_by_type: dict[str, int] = field(default_factory=dict)

    # ========================================================
    # 派生值
    # ========================================================

    @property
    def executed(self) -> int:
        """真的跑了的次数 = 总数 - 被拒绝的。"""

        return self.calls - self.denied

    @property
    def successes(self) -> int:
        """成功的次数。"""

        return self.executed - self.errors

    @property
    def error_rate(self) -> float:
        """
        失败率 = errors / calls。

        分母用 **calls** 而不是 executed：被拒绝也是被观测到的
        一次请求，把它从分母里拿掉会让"一半调用被拦下"这种
        情况看起来毫无异常。
        """

        return self.errors / self.calls if self.calls else 0.0

    @property
    def success_rate(self) -> float:
        """成功率 = successes / calls。"""

        return self.successes / self.calls if self.calls else 0.0

    @property
    def denied_rate(self) -> float:
        """被拒绝的占比。审批策略生效程度看这个。"""

        return self.denied / self.calls if self.calls else 0.0

    @property
    def avg_duration_ms(self) -> float:
        """平均耗时。没有执行过的调用时是 0.0。"""

        return (
            self.total_duration_ms / self.executed
            if self.executed
            else 0.0
        )


class ToolMetricsRegistry:
    """按 tool_key 累计指标。线程安全（一把锁护住整张表）。"""

    def __init__(self) -> None:
        self._metrics: dict[str, ToolMetrics] = {}
        self._lock = threading.Lock()

    def record(
            self,
            tool_key: str,
            *,
            duration_ms: float | None = None,
            error: BaseException | None = None,
            denied: bool = False,
            at: datetime | None = None,
    ) -> ToolMetrics:
        """
        记一次调用。

        参数是**可选**的：ASYNC 观察者拿不到可靠的耗时（它只能看
        墙钟，而墙钟包含它自己之前的处理器开销），那时传 None ——
        宁可少一个数，也不要一个会误导人的数。
        """

        with self._lock:

            metrics = self._metrics.get(tool_key)

            if metrics is None:
                metrics = ToolMetrics(tool_key=tool_key)
                self._metrics[tool_key] = metrics

            metrics.calls += 1

            if at is not None:
                metrics.last_called_at = at

            if denied:
                metrics.denied += 1
                return metrics

            if error is not None:
                metrics.errors += 1

                name = type(error).__name__
                metrics.errors_by_type[name] = (
                    metrics.errors_by_type.get(name, 0) + 1
                )

            if duration_ms is not None:
                metrics.total_duration_ms += duration_ms

                if (
                        metrics.min_duration_ms is None
                        or duration_ms < metrics.min_duration_ms
                ):
                    metrics.min_duration_ms = duration_ms

                if (
                        metrics.max_duration_ms is None
                        or duration_ms > metrics.max_duration_ms
                ):
                    metrics.max_duration_ms = duration_ms

            return metrics

    def snapshot(self) -> list[ToolMetrics]:
        """全部指标，按 tool_key 排序。"""

        with self._lock:
            return [
                self._metrics[key]
                for key in sorted(self._metrics)
            ]

    def for_tool(self, tool_key: str) -> ToolMetrics | None:
        """一个工具的指标；从没调用过时返回 None。"""

        with self._lock:
            return self._metrics.get(tool_key)

    def reset(self) -> None:
        """清零。**只给测试用** —— 生产上清零等于丢掉观测历史。"""

        with self._lock:
            self._metrics.clear()


#: 进程级单例。
_metrics: ToolMetricsRegistry | None = None


def get_tool_metrics() -> ToolMetricsRegistry:
    """取进程级指标表。"""

    global _metrics

    if _metrics is None:
        _metrics = ToolMetricsRegistry()

    return _metrics
