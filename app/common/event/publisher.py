"""
事件发布端的**抽象契约**（没有实现）。

实现见 `app/agent/event`（Redis Stream）。

⚠️ 为什么是 Protocol 而不是 ABC
------------------------------
本仓库两套都有，选哪套看**实现是怎么被选中的**：

    ABC（BaseService / BaseRepository）
        实现按契约**反射**找出来（同目录的 X_impl.py，
        见 common/introspect.py 的 resolve_impl）
    Protocol（ProviderAdapter / ModelRegistry / 这里）
        实现由**组合根注入** —— 拿到的就是现成对象，
        没有命名约定，也不需要 resolve_impl

这里属于后者：`app/main.py` 看数据源管理器里**有没有 redis**
（`required: false`，本地连不上是常态），决定装 Redis 实现还是一个
空对象。既没有 X_impl 约定可依，也就没有反射的必要。

⚠️ 实现必须自己吞掉异常 —— 这是契约的一部分
------------------------------------------
`publish` **不得**抛出。事件是旁路：一次存储抖动不该让整个 agent
运行失败，更不该让用户看到一句"模型不可用"。

把这条写进契约（而不是只写在某个实现的注释里），是因为**调用方会按
它写代码**：节点与派发 handler 直接 `await publish(...)`，不包
try/except。契约不说清，后来人就不知道该信谁 —— 一个"可能会抛"的
发布端会逼着每个调用点都加兜底，而那些兜底会慢慢把真正的错误吃掉。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .event import RunEvent
from .trace import TraceEvent

__all__ = ["EventPublisher"]


@runtime_checkable
class EventPublisher(Protocol):
    """
    把事件投递出去的东西。

    只有两个成员，因为调用方只需要这两件事：**有没有通道**，以及
    **投一条**。读取端不在这个契约里 —— 生产与消费是两条独立的路径
    （订阅端见 app/agent/event 的 RunStreamRegistry）。
    """

    @property
    def enabled(self) -> bool:
        """
        是否**真的**有通道（配置开着，且底层存储可用）。

        调用方用它做**接入决策**，而不是探活。最典型的用法在 agent
        节点：没有通道时不把流式回调接进模型层 —— 没有消费者还走流式
        是纯亏，而且那会让"已经推给消费端内容了"这条判断误触发
        （详见 agent_node.py 的说明）。
        """

        ...

    async def publish(self, event: RunEvent | TraceEvent) -> bool:
        """
        投递一条事件，返回是否**已接收**。**绝不抛异常。**

        ⚠️ 返回值是"收下了吗"，**不是**"写进存储了吗"
        ------------------------------------------
        实现可以把写入排到后台（Redis 实现就是这么做的，见
        app/agent/event/publisher.py），此时"已接收"与"已落库"是两件事：
        返回 True 之后写入仍可能失败，那只会体现在实现自己的留痕/计数上。

        之所以允许这样：要交付真实写入结果就必须等它，而"等"会把写入的
        延迟原样搬回调用方（本仓库的调用方在模型流式输出的 token 路径
        上）。二者只能取一个，这里取"不拖慢调用方"。

        因此返回值只适合表达"通道还在不在、还收不收"：False = 没通道、
        没启用、或通道已经不堪重负。调用方完全可以忽略它 —— 那正是这条
        契约存在的意义。
        """

        ...
