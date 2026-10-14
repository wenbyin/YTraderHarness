"""
TaskHandlerRegistry —— 把稳定的 handler 名字解析成"调谁、传什么"。

为什么不能用 import path（spec §3）
---------------------------------
两种做法：

    Task(handler="app.services.hitl.application.HitlService.timeout")   ✗
    Task(handler="hitl.timeout")                                        ✓

前者的代价有两层。浅的一层：类改名、模块拆分、实现替换，库里已有的行
全部失效 —— 一次重构变成一次数据迁移。深的一层更严重：**按字符串
import 任意模块本身就是个洞**。谁能写 task.handler（一次提交、甚至
一条被篡改的行）谁就能让 Worker 导入并调用任意代码。任务表是持久化
数据，不该是一条代码执行的入口。

所以名字是**逻辑名**，解析只走这张显式登记过的表。

注册发生在启动时，集中在组合根
----------------------------
本类不做自动发现、不扫包、不读装饰器 —— register() 由
app/main.py（组合根）在启动时逐个调用。理由是依赖方向：注册的内容是
业务 Handler（HitlService、RunService…），而本模块在 infra 层，
**不允许** import 任何业务模块。谁都可以 import 谁的那个地方，
只有组合根。

因此也避免了循环依赖：Task 核心不认识 HITL / Run / Memory，
是它们在启动时把自己挂进来。
"""

from __future__ import annotations

import inspect
import logging

from pydantic import BaseModel

from app.common.task.exceptions import UnknownHandlerError
from app.common.task.protocols import HandlerDefinition

__all__ = ["TaskHandlerRegistryImpl"]

logger = logging.getLogger(__name__)


def _assert_signature(
        service_contract: type,
        method: str,
        handler: str,
) -> None:
    """
    Handler 方法必须能接受 (params, ctx) 两个位置参数。

    注册期就查，而不是等第一条任务真的跑起来才发现 —— 调用点的
    TypeError 会被 Worker 归进"Handler 执行失败"，于是它以
    "重试几次后失败"的形式出现，而真正的原因（签名少了一个参数）
    在日志里只是一句 "takes 2 positional arguments but 3 were given"。

    检查方式是**试着绑定**而不是数参数个数：这样带默认值、关键字参数、
    或 `*args` 的写法都能通过，只有确实装不下两个位置参数的才会被拒。
    """

    target = getattr(service_contract, method)

    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):  # pragma: no cover - 内置/不可内省对象
        return

    # 留出 self 的位置。
    try:
        signature.bind(object(), object(), object())

    except TypeError as exc:
        raise TypeError(
            f"handler {handler} 要调 {service_contract.__name__}.{method}，"
            "但它的签名装不下 (params, ctx) 两个位置参数。"
            "Handler 方法一律写成 "
            f"`async def {method}(self, params, ctx) -> None`"
            "（不需要 ctx 也要声明它）—— 理由见"
            "app/common/task/protocols.py 的 HandlerDefinition.invoke。"
            f"（原始错误：{exc}）"
        ) from exc


class TaskHandlerRegistryImpl:
    """
    HandlerDefinition 的显式登记表。

    刻意**不是**进程级单例：单例会让测试之间互相污染（一个用例注册的
    Handler 泄漏到下一个用例，"这个 handler 已被注册"的报错于是只在
    特定顺序下出现）。实例由组合根建、放进 ServiceFactory 的池子里，
    测试各建各的。
    """

    def __init__(self) -> None:

        self._definitions: dict[str, HandlerDefinition] = {}

    # ========================================================
    # 注册
    # ========================================================

    def register(
            self,
            *,
            handler: str,
            service_contract: type,
            method: str,
            params_type: type[BaseModel],
    ) -> None:
        """
        注册一个 Handler。启动时集中调用。

        参数在**注册期**就校验，理由与"参数在提交期校验"一样：注册错误
        属于部署配置错误，应该在启动时炸出来，而不是等到某条任务真的
        被投递、跑起来、才发现方法名拼错了。

        Raises:
            TypeError: 参数不合法（handler 空、method 为空、params_type
                不是 BaseModel 子类、service_contract 上没有那个方法）。
            ValueError: 同名 handler 已经注册过**另一个**定义。
        """

        if not handler or not handler.strip():
            raise TypeError("handler 不能为空。")

        if handler != handler.strip():
            raise TypeError(
                f"handler 首尾不能有空白：{handler!r} —— "
                "库里的值是按字面比较的，带空白的名字永远匹配不上。"
            )

        if not method or not method.strip():
            raise TypeError(f"handler {handler} 的 method 不能为空。")

        if not (
            isinstance(params_type, type)
            and issubclass(params_type, BaseModel)
        ):
            raise TypeError(
                f"handler {handler} 的 params_type 必须是 Pydantic Model "
                f"的子类（spec §2：不允许用任意 dict 当业务参数类型），"
                f"实际是 {params_type!r}。"
            )

        if not isinstance(service_contract, type):
            raise TypeError(
                f"handler {handler} 的 service_contract 必须是类，"
                f"实际是 {service_contract!r}。"
            )

        # 方法必须在**契约**上存在。
        #
        # 只查实现类是不够的：契约是 Worker 唯一看得见的东西
        # （它只拿到 service_contract，靠 ServiceFactory 反射出实现），
        # 契约上没有的方法在运行期才会以 AttributeError 的形式出现 ——
        # 而那会被当成"Handler 执行失败"、进而触发几次重试。
        if not hasattr(service_contract, method):
            raise TypeError(
                f"handler {handler} 要调 {service_contract.__name__}."
                f"{method}，但那个契约上没有这个方法。"
                "（注册要填的是**服务契约**上的方法名。）"
            )

        _assert_signature(service_contract, method, handler)

        definition = HandlerDefinition(
            handler=handler,
            params_type=params_type,
            service_contract=service_contract,
            method=method,
        )

        existing = self._definitions.get(handler)

        if existing is not None:

            # 同一定义重复注册是**允许**的：开发时的热重载、或同一进程里
            # 走了两次初始化路径，都会发生。只要意图相同就不该报错。
            if existing == definition:
                logger.debug(
                    "[Task] handler %s 重复注册（定义相同，忽略）",
                    handler,
                )
                return

            raise ValueError(
                f"handler {handler} 已经注册过另一个定义："
                f"已有 {existing.service_contract.__name__}."
                f"{existing.method}({existing.params_type.__name__})，"
                f"新的是 {service_contract.__name__}.{method}"
                f"({params_type.__name__})。"
                "同一个逻辑名指向两个实现时，谁生效取决于注册顺序 —— "
                "这类问题只有在运行期才会以「调到了另一个 Handler」的形式"
                "暴露，所以在这里就报错。"
            )

        self._definitions[handler] = definition

    # ========================================================
    # 解析
    # ========================================================

    def resolve(
            self,
            handler: str,
    ) -> HandlerDefinition:
        """
        Raises:
            UnknownHandlerError: 这个名字没注册过。错误里带上已知的全部
                名字 —— 这条路径上最常见的原因是拼写错误，而把可选值
                列出来比让人去翻注册代码快得多。
        """

        definition = self._definitions.get(handler)

        if definition is None:
            known = ", ".join(sorted(self._definitions)) or "（一个都没有）"

            raise UnknownHandlerError(
                f"未注册的 handler：{handler!r}。已注册的有：{known}。"
            )

        return definition

    # ========================================================
    # 自省
    # ========================================================

    def known(self) -> tuple[str, ...]:
        """已注册的全部 handler 名（排序）。"""

        return tuple(sorted(self._definitions))

    def __contains__(
            self,
            handler: object,
    ) -> bool:

        return handler in self._definitions

    def __len__(self) -> int:

        return len(self._definitions)
