"""
仓储工厂：构造仓储的唯一入口。

没有任何登记表
--------------
调用方递进来一个**契约**（如 DocumentRepository），工厂自己去找它的
具体实现 —— 见 common/introspect.py 的 resolve_impl。因此：

    新增一个仓储   只要 Impl 与契约在同一个包、且叫 X_impl.py，就不用
                   改任何中央文件，也不用 register()
    工厂保持纯叶子 它不认识任何具体实现，app.common 因此不必反向
                   import app.services

session 从哪来（对应「自己提供 or 传入」）
-----------------------------------------
工厂可以处在三种状态，create() 按固定顺序解析：

    1. create(..., session=s)      调用方显式传入，优先级最高
    2. with_session(s) 派生的工厂  请求路径：整个请求共用一个事务
    3. 基工厂的 session_factory    后台路径：仓储自持，各自开短事务

第 3 种就是"工厂自己提供"：调用方什么都不用给。第 1、2 种是
"调用方传入"。三种都走同一个 create()，调用处不必分叉。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.introspect import resolve_impl

from .base import BaseRepository, SessionFactory

__all__ = [
    "RepositoryFactory",
    "UNSET",
]


class _Unset:
    """哨兵：区分"没传 session"与"显式传了 session=None"。"""

    def __repr__(self) -> str:  # pragma: no cover - 仅用于报错信息
        return "<UNSET>"


#: create() 的 session 默认值。见 _Unset 的说明。
UNSET = _Unset()


class RepositoryFactory:
    """按契约类型构造仓储。"""

    def __init__(
            self,
            session_factory: SessionFactory | None = None,
            *,
            session: AsyncSession | None = None,
    ) -> None:

        if session_factory is None and session is None:
            raise ValueError(
                "RepositoryFactory 需要 session_factory 或 session "
                "之一：请求路径传 session，后台路径传 session_factory。"
            )

        self._session_factory = session_factory
        self._session = session

    @property
    def session_factory(self) -> SessionFactory | None:
        """
        本工厂持有的造 session 工厂。

        ServiceFactory 复用它来开请求事务，而不是自己再存一份 ——
        "应用的 session_factory"只该有一个来源。
        """

        return self._session_factory

    # ========================================================
    # 构造
    # ========================================================

    def create(
            self,
            contract: type,
            *,
            session: Any = UNSET,
            impl: type[BaseRepository] | None = None,
    ) -> Any:
        """
        按契约造一个仓储。

        实现由 resolve_impl 反射找出（契约的子类 / 约定的 X_impl 模块）；
        确有多个实现时用 impl= 显式指定。

        session 的解析顺序见模块 docstring。给不出 session 也拿不到
        session_factory 时抛错 —— 那是装配期配置漏了，不是运行期
        偶发，越早炸越好。
        """

        target = impl if impl is not None else resolve_impl(contract)

        if not isinstance(session, _Unset):
            # 1. 显式传入，最高优先级。
            return target(session)

        if self._session is not None:
            # 2. 请求路径：本工厂绑定了一个请求级 session。
            return target(self._session)

        # 3. 后台路径：让仓储自持，各自开短事务。
        return target(session_factory=self._session_factory)

    # ========================================================
    # 派生
    # ========================================================

    def with_session(
            self,
            session: AsyncSession,
    ) -> RepositoryFactory:
        """
        派生一个绑定到指定 session 的工厂（请求路径）。

        典型用法是请求进来时派生一次，整个请求内所有仓储共用同一个
        session —— 也就共用同一个事务。
        """

        return RepositoryFactory(
            self._session_factory,
            session=session,
        )

    @asynccontextmanager
    async def scope(
            self,
            *,
            begin: bool = True,
    ) -> AsyncIterator[RepositoryFactory]:
        """
        派生一个绑定到**新开 session** 的工厂（后台路径）。

        给不持有请求事务的调用方用：流水线节点、观测写入、后台任务。
        用法：

            async with factory.scope() as repos:
                await repos.create(DocumentChunkRepository).add_many(...)

        begin 区分读与写，与 BaseRepository.new_session 同一套语义：
        只读路径（metadata_extraction 的两次决策查询、recorder 的
        summarize、artifacts 的 reference）传 begin=False。
        """

        if self._session_factory is None:
            raise RuntimeError(
                "本工厂没有 session_factory，无法开新会话 —— "
                "它是在传入 session 的模式下构造的。"
            )

        async with self._session_factory() as session:
            if begin:
                async with session.begin():
                    yield self.with_session(session)
            else:
                yield self.with_session(session)
