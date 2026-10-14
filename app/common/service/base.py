"""
服务基类与运行时上下文。

为什么需要基类
--------------
8 个应用服务原本各自 `class XxxService(ABC)`，彼此没有共同祖先：
"一个服务是什么"没有单点定义，横切依赖（session / dispatcher /
post-commit 钩子）也只能各写各的构造函数。

现在统一继承 BaseService，横切依赖收进 ServiceContext。

为什么 ServiceContext 的字段是 Any
----------------------------------
app.common 必须保持为纯叶子 —— 见 common/exceptions.py 的约定：
不允许出现 `from app.services...` 的导入。

而上下文里装的东西（GraphDispatcher、PipelineRegistry、
FileStorage…）全是 app.infra / app.services 下的类型。写成具体
类型标注就等于把叶子弄脏了。因此这里只声明"有这么个槽位"，
类型由各业务模块在 composition.py 里自己保证 —— 那里本来就
import 了具体实现，本来就知道类型。

字段是开放的
------------
固定的槽位只覆盖**每个服务都可能用到**的横切依赖。模块自己的
专属依赖（QA 的输出目录、图上注册的图名…）走 extras，
用 get / require 按 key 取。这样加一个模块不用改本文件。
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BaseService",
    "PostCommitHook",
    "RollbackHook",
    "ServiceContext",
]

#: 组合根在事务提交后要执行的钩子。
#:
#: 服务侧只往列表里追加，**不执行** —— 执行时机由事务边界决定。
#: 放错位置会导致"事务回滚了但任务已投出去"，执行时机的完整说明
#: 见 ServiceFactory.scope()。
PostCommitHook = Callable[[], Awaitable[None]]

#: 组合根在事务**没有**提交时要执行的补偿钩子。
#:
#: 与 post_commit 相对：那个只在提交后跑，这个只在回滚 / 异常时跑。
#: 用途是撤销**事务之外**的副作用 —— 进程内锁、已占用的外部配额之类。
#: 事务内的东西不需要它：数据库自己会回滚。
#:
#: ⚠️ 它跑在 `async with session_factory()` 退出**之后** —— session 已经关了，
#: 所以**不能**用它写库。真要写，得用 session_factory 另开一段事务。
RollbackHook = Callable[[], Awaitable[None]]


@dataclass
class ServiceContext:
    """
    服务运行期的横切依赖容器。

    谁构造谁负责填：生产上由 ServiceFactory.scope() 填，
    测试里可以自己 new 一个只填用得到的槽位。
    """

    #: 本服务所在事务的 session。请求路径下有值。
    session: Any = None

    #: 造 session 的工厂。需要另开短事务的服务用它。
    session_factory: Any = None

    #: 图调度端口。只有 DocumentRuntimeService 真正用得上，
    #: 其余服务拿到 None 也不会去看它。
    dispatcher: Any = None

    #: 提交后要跑的钩子。服务往这里追加，组合根在事务提交后执行。
    post_commit: list[PostCommitHook] = field(default_factory=list)

    #: 回滚 / 异常时要跑的补偿钩子。服务往这里追加，组合根在事务
    #: **没有**提交时执行。与 post_commit 恰好互斥 —— 一次调用里
    #: 两者只会跑一边。说明见 RollbackHook。
    rollback: list[RollbackHook] = field(default_factory=list)

    #: 模块专属依赖。见模块 docstring。
    extras: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """按 key 取模块专属依赖，没有就返回 default。"""

        return self.extras.get(key, default)

    def require(self, key: str) -> Any:
        """
        按 key 取模块专属依赖，没有就抛。

        装配期漏填应当在**构造服务时**就暴露，而不是等某个请求走到
        一半才以一句与原因无关的 AttributeError 炸开。
        """

        try:
            return self.extras[key]

        except KeyError as exc:
            raise KeyError(
                f"ServiceContext 缺少依赖 {key!r}。"
                "检查 ServiceFactory 构造时是否把它放进 singletons，"
                "或对应 composition.py 的 builder 是否漏传。"
            ) from exc


class BaseService(ABC):
    """
    所有应用服务契约的基类。

    契约（ABC）继承本类，实现（Impl）再继承契约 —— 于是
    "服务"这个概念有一个单点定义，横切依赖也只有一个入口。

    构造参数 ctx 是可选的：显式装配（测试、冒烟脚本）可以不传，
    此时得到一个空上下文，取用具体槽位会拿到 None。
    """

    def __init__(
            self,
            ctx: ServiceContext | None = None,
    ) -> None:
        self._ctx = ctx if ctx is not None else ServiceContext()

    @property
    def context(self) -> ServiceContext:
        """本服务的运行时上下文。"""

        return self._ctx

    # ========================================================
    # 常用槽位的直通读取
    # ========================================================

    @property
    def session(self) -> Any:
        return self._ctx.session

    @property
    def dispatcher(self) -> Any:
        return self._ctx.dispatcher

    @property
    def post_commit(self) -> list[PostCommitHook]:
        return self._ctx.post_commit

    @property
    def rollback(self) -> list[RollbackHook]:
        return self._ctx.rollback

    def require(self, key: str) -> Any:
        return self._ctx.require(key)
