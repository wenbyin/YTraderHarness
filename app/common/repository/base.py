"""
仓储基类：session 来源的唯一约定。

仓储是整个分层里最深的一层 —— 它直接持有 AsyncSession，上面所有
服务都通过它碰数据库。因此"session 从哪来"必须有**一个**说法，
而不是每个仓储各写一遍 __init__。

两种来源，都从本基类走：

    请求路径    调用方把请求级 session 传进来。
                它与业务写在同一事务里，仓储不自行 commit / rollback。

    后台路径    只给 session_factory，由 new_session() 开一段短事务。
                流水线节点、观测写入、后台任务都走这条 —— 它们不持有
                任何请求事务。

不允许"两样都没有"：那样的仓储根本没法工作，却要等到第一次查询才
以一句与原因无关的 AttributeError 暴露出来。构造时就报错，报错点
离原因最近。

为什么保留 self._session 这个名字
--------------------------------
所有既有仓储的方法体都是 self._session.execute(...) 这种写法。
基类沿用同一个属性名，各仓储就只需要删掉自己的 __init__，
方法体一个字都不用改。
"""

from __future__ import annotations

from abc import ABC
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "BaseRepository",
    "SessionFactory",
]

#: 造 session 的工厂。
#:
#: async_sessionmaker 与数据源上的同名属性都符合这个形状。这里不写成
#: 具体的 async_sessionmaker 类型，是因为基类不该假设上层用的是
#: SQLAlchemy —— 换数据源时端口不变。
SessionFactory = Callable[[], Any]


class BaseRepository(ABC):
    """
    所有仓储的基类。

    子类**不需要**再写 __init__：直接继承下面这个签名即可，
    于是"传 session"与"传 session_factory"两条路自动都通。
    """

    def __init__(
            self,
            session: AsyncSession | None = None,
            *,
            session_factory: SessionFactory | None = None,
    ) -> None:

        if session is None and session_factory is None:
            raise ValueError(
                f"{type(self).__name__} 需要 session 或 "
                "session_factory 之一：请求路径传 session，"
                "后台路径传 session_factory。"
            )

        self._session = session
        self._session_factory = session_factory

    # ========================================================
    # session 取用
    # ========================================================

    @property
    def session(self) -> AsyncSession:
        """
        当前绑定的 session。

        只传了 session_factory 时没有"当前 session"可取 —— 那种模式
        下每次操作各自开一段短事务（见 new_session）。这里明确抛错，
        而不是返回 None 让调用方在更远的地方拿到
        "NoneType has no attribute execute"。
        """

        if self._session is None:
            raise RuntimeError(
                f"{type(self).__name__} 是以 session_factory 模式构造的，"
                "没有绑定中的 session。请用 new_session() 开一段短事务，"
                "或改用传入 session 的构造方式。"
            )

        return self._session

    @property
    def session_factory(self) -> SessionFactory | None:
        """造 session 的工厂。请求路径下为 None。"""

        return self._session_factory

    @asynccontextmanager
    async def new_session(
            self,
            *,
            begin: bool = True,
    ) -> AsyncIterator[AsyncSession]:
        """
        自开一段 session（后台路径）。

        begin 区分读与写：

            写    begin=True，退出时 commit / 异常时 rollback
            读    begin=False，不开显式事务

        既有代码里这两种写法都有（recorder 的 summarize 与
        artifacts 的 reference 是只读、不 begin 的），因此这里保留
        这个开关，而不是一律包事务。
        """

        if self._session_factory is None:
            raise RuntimeError(
                f"{type(self).__name__} 没有 session_factory，"
                "无法自开 session —— 它是在传入 session 的模式下构造的。"
            )

        async with self._session_factory() as session:
            if begin:
                async with session.begin():
                    yield session
            else:
                yield session
