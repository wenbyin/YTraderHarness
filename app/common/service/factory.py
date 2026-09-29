"""
服务工厂：构造服务实例的唯一入口。

没有任何登记表
--------------
调用方递进来一个**契约**（如 DocumentRuntimeService），工厂自己：

    1. 找出它的具体实现   —— 见 common/introspect.py 的 resolve_impl
    2. 读该实现的 __init__ 签名，逐个参数决定喂什么
    3. 实例化

第 2 步就是"识别构造方法"：参数的来源只有三处 ——

    仓储契约注解    -> 交给 RepositoryFactory 构造
    公共池子         -> 按参数名 / 按类型取现成实例
    ctx             -> 本次调用的运行时上下文

因此**不需要任何 register()**：新增一个服务，只要它的 Impl 与契约
同包且叫 X_impl.py，工厂就能直接造出来；新增一个依赖，只要池子里有
对应的实例，也会被自动喂进去。

这也让 app.common 保持纯叶子 —— 工厂不认识任何具体实现。

参数解析顺序（完整规则见 _build_kwargs）
----------------------------------------
    调用方显式覆盖 > ctx > 仓储 > 池子(按名) > 池子(按类型)
    > 实现自己的默认值 > 报错

session 与事务边界
------------------
create() 只造实例，**不碰事务** —— 适合已经持有 session 的调用方。

scope() 才是请求/后台路径的入口：它开 session、开事务、造服务、
交出控制权、等调用方返回、退出事务（提交或回滚），最后才跑
post-commit 钩子。这一串顺序是有讲究的，见 scope() 的 docstring。
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from app.common.introspect import (
    is_broad_type,
    resolve_impl,
    unwrap_annotation,
)
from app.common.repository.base import BaseRepository
from app.common.repository.factory import UNSET, RepositoryFactory

from .base import BaseService, PostCommitHook, ServiceContext

__all__ = [
    "ServiceFactory",
]


class ServiceFactory:
    """
    按契约类型构造服务。

    pool 是"公共池子"：进程级单例（调度器、能力注册表、图注册表、
    产物存储、文件存储……）与模块专属的固定值（图名、输出目录……）
    都放这里，构造服务时按需取用。

    池子是**实例级**的：它是这一份运行时依赖，随应用启动而建，
    而不是类级别的全局状态。
    """

    def __init__(
            self,
            repositories: RepositoryFactory,
            *,
            pool: dict[str, Any] | None = None,
    ) -> None:

        self._repositories = repositories

        #: 公共池子。键是取用名，值是实例。
        self._pool: dict[str, Any] = dict(pool or {})

    @property
    def repositories(self) -> RepositoryFactory:
        return self._repositories

    @property
    def pool(self) -> dict[str, Any]:
        """公共池子（只读用途，例如自检脚本）。"""

        return dict(self._pool)

    # ========================================================
    # 能力查询
    # ========================================================

    def needs_session(
            self,
            contract: type,
            *,
            impl: type | None = None,
    ) -> bool:
        """
        这个服务用不用得上 session。

        判据：实现的 __init__ 里有没有**仓储契约**参数。

        仓储是服务碰数据库的唯一入口，因此"没有仓储参数"就等于
        "不需要 session"。应用里唯一一个这样的服务是能力目录
        （ProcessorCapabilityService）—— 它读的是启动时反射出来的
        静态能力表，构造后即冻结，为它开一段会话只会白占一个连接。

        拿不准时返回 True：多开一段会话只是浪费一个连接，
        少开一段却会让服务在运行期才炸 —— 两者代价不对称。
        """

        try:
            target = impl if impl is not None else resolve_impl(contract)
            init = target.__init__
            hints = typing.get_type_hints(init)
            parameters = inspect.signature(init).parameters
        except Exception:
            return True

        return any(
            self._is_repository(
                unwrap_annotation(hints.get(name))
            )
            for name in parameters
            if name != "self"
        )

    # ========================================================
    # 构造
    # ========================================================

    def create(
            self,
            contract: type,
            *,
            session: Any = UNSET,
            ctx: ServiceContext | None = None,
            repositories: RepositoryFactory | None = None,
            impl: type[BaseService] | None = None,
            **overrides: Any,
    ) -> Any:
        """
        造一个服务实例，**不开事务**。

        给已经持有 session 的调用方用（请求依赖、后台任务）。
        session 的解析顺序与 RepositoryFactory.create 一致：
        显式传入 > 本工厂自持 > 让仓储自持。

        overrides 优先级最高，用来覆盖单次构造才不同的参数
        （例如 artifact_store、storage、graph_name）。
        """

        target = impl if impl is not None else resolve_impl(contract)

        # UNSET 是模块级哨兵单例，按 identity 判断即可 ——
        # 因此 create(..., session=None) 与"没传 session"是两回事。
        session_given = session is not UNSET

        if repositories is None:
            repositories = (
                self._repositories.with_session(session)
                if session_given
                else self._repositories
            )

        if ctx is None:
            ctx = self._build_context(
                session=session if session_given else None,
            )

        kwargs = self._build_kwargs(
            target, repositories, ctx, overrides
        )

        return target(**kwargs)

    @asynccontextmanager
    async def scope(
            self,
            contract: type,
            *,
            begin: bool = True,
            **overrides: Any,
    ) -> AsyncIterator[Any]:
        """
        开一段事务 + 造服务 + 交出控制权 + 收尾。

        执行顺序（每一步的位置都是有理由的）：

            async with session_factory() as session:
                async with session.begin():
                    yield service          # 控制权在这里交给调用方
            # <- 事务已提交
            for hook in post_commit:
                await hook()

        为什么钩子必须放在**事务退出之后**：

            - 流水线不能在请求事务里跑（后台任务读不到未提交的行，
              且一次运行几分钟会压垮请求）
            - 也不能"登记完就立刻投出去"：此刻版本行还没提交，
              后台任务立刻去读必然查不到

        为什么异常路径天然安全：

            调用方抛异常时，异常在 yield 点被抛回本生成器，
            穿过 session.begin()（回滚）与 session_factory()，
            于是下面那个 for 循环**整段被跳过** ——
            不会出现"事务回滚了但任务已投出去"。

        begin=False 用于只读路径（不开显式事务）。
        """

        session_factory = self._repositories.session_factory

        if session_factory is None:
            raise RuntimeError(
                "ServiceFactory.scope() 需要仓储工厂持有 "
                "session_factory —— 请用 RepositoryFactory("
                "session_factory=...) 构造，而不是传 session。"
            )

        post_hooks: list[PostCommitHook] = []

        async with session_factory() as session:
            repositories = self._repositories.with_session(session)

            ctx = self._build_context(
                session=session,
                post_commit=post_hooks,
            )

            service = self.create(
                contract,
                ctx=ctx,
                repositories=repositories,
                **overrides,
            )

            if begin:
                async with session.begin():
                    yield service
            else:
                yield service

        # 走到这里 = 事务已提交（回滚路径在 yield 点就抛走了）。
        for hook in post_hooks:
            await hook()

    # ========================================================
    # internals
    # ========================================================

    def _build_context(
            self,
            *,
            session: Any = None,
            post_commit: list[PostCommitHook] | None = None,
    ) -> ServiceContext:
        """按当前工厂的池子与 session 拼一个上下文。"""

        return ServiceContext(
            session=session,
            session_factory=self._repositories.session_factory,
            dispatcher=self._pool.get("dispatcher"),
            post_commit=(
                post_commit if post_commit is not None else []
            ),
            # 池子整份放进 extras：模块专属依赖（图名、输出目录…）
            # 也走这里，避免每加一个就要改 ServiceContext 的字段。
            extras=dict(self._pool),
        )

    def _build_kwargs(
            self,
            impl: type,
            repositories: RepositoryFactory,
            ctx: ServiceContext,
            overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """
        读 impl.__init__ 的签名，逐个参数决定喂什么。

        解析顺序（前面的赢了就轮不到后面）：

            1. overrides              调用方显式给的
            2. ctx / ServiceContext   运行时上下文
            3. 仓储契约注解            交给 RepositoryFactory 构造
            4. 参数名命中池子          最精确的按名匹配
            5. 注解类型命中池子        跳过 str/list 这类宽类型
            6. 实现自己的默认值        不传这个参数
            7. 都没有 -> 报错          装配期漏配，越早炸越好

        第 4 步排在第 5 步前面是有意的：池子里若同时有多个同类型
        实例（例如两个 str），按名匹配才不会有歧义。
        """

        try:
            init = impl.__init__
        except AttributeError:  # pragma: no cover - 不可能发生
            return {}

        # with `from __future__ import annotations`（全项目都是），
        # 注解是**字符串**，必须解析一次才能做类型匹配。
        try:
            hints = typing.get_type_hints(init)
        except Exception:
            # 注解里有解析不了的名字（循环 import 中途等）——
            # 退回按名字匹配，不让整个构造失败。
            hints = {}

        kwargs: dict[str, Any] = {}

        for name, param in inspect.signature(init).parameters.items():

            if name == "self":
                continue

            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            # 1. 调用方显式覆盖
            if name in overrides:
                kwargs[name] = overrides[name]
                continue

            annotation = unwrap_annotation(hints.get(name))

            # 2. 上下文
            if annotation is ServiceContext or name == "ctx":
                kwargs[name] = ctx
                continue

            # 3. 仓储：交给 RepositoryFactory
            if self._is_repository(annotation):
                kwargs[name] = repositories.create(annotation)
                continue

            # 4. 池子：按参数名
            if name in self._pool:
                kwargs[name] = self._pool[name]
                continue

            # 5. 池子：按类型（宽类型跳过，否则 str 会误配）
            if annotation is not None and not is_broad_type(
                annotation
            ):
                matched = self._match_pool_by_type(annotation)

                if matched is not _NO_MATCH:
                    kwargs[name] = matched
                    continue

            # 6. 交给实现自己的默认值
            if param.default is not inspect.Parameter.empty:
                continue

            # 7. 没有来源
            raise TypeError(
                f"无法为 {impl.__name__}.__init__ 的参数 {name!r} "
                f"找到来源（注解 {annotation!r}）。"
                "它既不是仓储契约，也不在池子里，又没有默认值。"
                f"请把它放进 ServiceFactory 的 pool，"
                f"或调用时用 create(..., {name}=...) 显式传入。"
            )

        return kwargs

    def _match_pool_by_type(self, annotation: Any) -> Any:
        """在池子里找一个 annotation 类型的实例。找不到返回哨兵。"""

        for value in self._pool.values():
            try:
                if isinstance(value, annotation):
                    return value
            except TypeError:
                # annotation 不是可 isinstance 的类型（泛型别名等）
                continue

        return _NO_MATCH

    @staticmethod
    def _is_repository(annotation: Any) -> bool:
        """注解是不是一个仓储契约。"""

        return (
            isinstance(annotation, type)
            and issubclass(annotation, BaseRepository)
        )


class _NoMatch:
    """哨兵：池子里没有匹配的值（区别于"匹配到的值是 None"）。"""

    def __repr__(self) -> str:  # pragma: no cover
        return "<NO_MATCH>"


_NO_MATCH = _NoMatch()
