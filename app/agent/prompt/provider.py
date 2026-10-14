"""
默认 PromptProvider：节点提示词走包内目录，职责提示词走 PromptService。

两类来源，一个出口
-----------------
    node_prompt   本包注册表（进程级常量，无 I/O，同步）
    duty_prompt   ServiceFactory -> PromptService（业务数据，异步）

⚠️ 为什么职责提示词经 **ServiceFactory.create** 而不是 `scope(...)`
---------------------------------------------------------------
`ServiceFactory.scope()` **无条件**开一段会话（`async with
session_factory()`，见 app/common/service/factory.py），哪怕这个服务一个
仓储参数都没有 —— 它是为"要写库的服务"准备的。职责提示词当前是一张硬编码
表，为它开一个数据库连接是自找的依赖（数据库抖一下，主 agent 的 system
提示词就没了）。

`create()` 只做"解析构造参数 + 造实例"，不开会话、不碰数据库，且 `ctx`
照样注入（见 `_build_kwargs`）。因此这里用 create。

将来提示词进了库（那时 PromptService 会带仓储参数），改成 `scope(...)` 是
**一行**的事 —— `needs_session()` 本来就是为这种判断准备的。
"""

from __future__ import annotations

import logging
from typing import Any

from app.common.exceptions import NotFoundError
from app.common.factories import get_service_factory

from .contracts import (
    PromptNotFoundError,
    PromptProvider,
    PromptUnavailableError,
)
from .registry import found_prompts

__all__ = [
    "DefaultPromptProvider",
    "get_prompt_provider",
    "reset_prompt_provider",
    "set_prompt_provider",
]

logger = logging.getLogger(__name__)


class DefaultPromptProvider:
    """
    `PromptProvider` 的默认实现。

    Parameters
    ----------
    service_factory
        组合根装配期注入的服务工厂（职责提示词的入口）。为 None 时回落到
        进程级 `get_service_factory()` —— 与 model_registry / policy_resolver
        那套"端口为空则回落"的形状一致（见 processor/dependencies.py）。
        两者都没有（脚本、测试没跑过 lifespan）时抛 PromptUnavailableError，
        而不是抛一句与原因无关的 RuntimeError。
    """

    def __init__(self, *, service_factory: Any = None) -> None:
        self._service_factory = service_factory

    # ========================================================
    # 节点功能性提示词
    # ========================================================

    def node_prompt(self, model_role: str, /, **variables: Any) -> str:
        """
        取节点提示词并渲染。

        这里**不做**降级：一份节点提示词是**编译期事实**（文件就在仓库里），
        取不到意味着代码写错了 —— 那是部署/开发错误，不是运行期状况。
        职责提示词相反，它可能来自外部存储，所以调用方要准备降级
        （见 agent_builder_node 的处理）。
        """

        definition = found_prompts().get(model_role)

        if definition is None:
            raise PromptNotFoundError(
                f"没有 model_role={model_role!r} 的节点提示词。"
                f"已登记：{sorted(found_prompts())}。\n"
                "⚠️ 注意 reasoning（主循环）**本来就不该**在这张表里："
                "主 agent 的提示词是**职责**提示词，走 duty_prompt —— "
                "它回答的是「这个模型是干什么的」，而不是「这次调用干什么」。"
            )

        return definition.render(**variables)

    # ========================================================
    # 职责功能性提示词
    # ========================================================

    async def duty_key(self) -> str:
        """
        这次运行该用哪个职责（选择逻辑的唯一落点）。

        建运行那一步取它落库；主循环那边拿到的是**同一个值**（经
        `agent_run.duty_key` → 图 state），不会各自再选一次 —— 一次运行
        里"用的哪份职责"因此只有一个答案。

        Raises:
            PromptNotFoundError: 选出来的职责没登记。
            PromptUnavailableError: 取不到职责表（没跑过 lifespan）。
        """

        try:
            return await self._service().default_duty_key()

        except NotFoundError as exc:
            raise PromptNotFoundError(str(exc)) from exc

    async def duty_prompt(self, duty_key: str | None = None) -> str:
        """
        取主 agent 的职责提示词。

        Args:
            duty_key: 职责标识；None = 默认职责。**主循环传的是运行行上
                      那个 key**（见 `duty_key`），None 只在不该有值的地方
                      出现（老 checkpoint、脚本）。

        Raises:
            PromptNotFoundError: 这个职责没登记。
            PromptUnavailableError: 取不到职责表（没跑过 lifespan）。
        """

        try:
            return await self._service().get_duty_prompt(duty_key)

        except NotFoundError as exc:

            # 把服务层的"资源不存在"翻成本端口自己的词汇：调用方只该认
            # PromptError 这一族，不必知道底下是服务层还是别的东西。
            raise PromptNotFoundError(str(exc)) from exc

    # --------------------------------------------------------

    def _service(self) -> Any:
        """
        造一个 `PromptService`。

        ⚠️ 惰性 import：本模块属于 app.agent，而 PromptService 在
        app.services。放在函数里是为了让"agent 侧 import 期"不经过业务
        模块（app/services/session 反过来 import app/agent/graph，顶层
        import 容易绕成环）。组合根负责在启动期把它接上。

        用 `create` 而不是 `scope`：理由见模块 docstring（不为一段文本
        开数据库会话）。提示词进库后这里换 `scope` 是一行。
        """

        from app.services.prompt.application import PromptService

        return self._factory().create(PromptService)

    def _factory(self) -> Any:
        """本次要用的服务工厂（注入的 > 进程级的）。"""

        if self._service_factory is not None:
            return self._service_factory

        try:
            return get_service_factory()

        except RuntimeError as exc:

            raise PromptUnavailableError(
                f"取职责提示词需要 ServiceFactory，但应用没装配过它（{exc}）。\n"
                "生产路径上组合根会在 lifespan 里装好（见 app/main.py 5.2）；"
                "脚本 / 测试里没有 lifespan 是常态，调用方应当降级处理。"
            ) from exc


# ============================================================
# 进程级默认
# ============================================================
#
# 与 get_model_registry / reset_factories 同一形状：进程级事实一个进程一份，
# 组合根装它、测试整份替换它。

_provider: PromptProvider | None = None


def get_prompt_provider() -> PromptProvider:
    """
    取进程级 PromptProvider。

    没装过就现建一个默认的（不带服务工厂 —— 节点提示词照常工作，职责
    提示词会去回落进程级 ServiceFactory）。**懒加载**是为了让
    "组合根还没跑"的脚本也能取节点提示词：提示词是常量，读它不该要求
    整个应用先启动。
    """

    global _provider

    if _provider is None:
        _provider = DefaultPromptProvider()

    return _provider


def set_prompt_provider(provider: PromptProvider | None) -> None:
    """
    替换进程级 PromptProvider（**组合根与测试**的接缝）。

    传 None 等价于 reset。组合根在 lifespan 里用它把带 ServiceFactory 的
    实现装进来。
    """

    global _provider

    _provider = provider


def reset_prompt_provider() -> None:
    """丢弃进程级 Provider（**只给测试用**）。"""

    global _provider

    _provider = None
