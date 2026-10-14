"""
提示词契约：端口、定义与异常。

两类提示词，一个端口
-------------------
    node_prompt(model_role, **variables)    节点功能性：这次**调模型干什么**
                                            （分类 / 摘要 / 将来的记忆抽取）
    duty_prompt(duty_key)                   职责功能性：主 agent 搭载的模型
                                            **是干什么的**
                                            （stock_research / python_programmer）

前者是**进程级常量**（一个 model_role 一份文本，不查任何外部存储），
后者是**业务数据**（先硬编码在 services/prompt，将来进库）—— 因此两者
的取用形态不同，见下面"为什么一同步一异步"。

为什么一同步一异步
----------------
这不是疏忽，是两侧调用点的形状决定的：

    节点提示词   纯文本、无 I/O。摘要器在**同步**上下文里被构造
                 （dependencies.resolved_compactor），它必须能同步拿到。
    职责提示词   经 ServiceFactory 造 PromptService 再取（将来读库），
                 服务层的方法都是 async —— 只能 await。

不对称只存在于**这一层**：调用方各自只碰自己那一种，谁也看不到另一边的
签名（builder 节点只 await duty_prompt，摘要器只同步调 node_prompt）。

变量契约由 render 的**签名**表达
------------------------------
`PromptDefinition.render(**variables)` 就是"这份提示词需要哪些参数"的
唯一清单：名字写错、少传、多传都是 `TypeError`，不会静默。

刻意**不**做一套占位符校验器：
    - `str.format` 遇到提示词里的 JSON 花括号会炸（提示词经常含示例）
    - `string.Template` 需要额外维护一份占位符清单 —— 而签名本来就是
      清单，还带类型与默认值
    - 运行期反射比对"模板里的 $x 有没有被传进来"是第三份清单
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "PromptDefinition",
    "PromptError",
    "PromptNotFoundError",
    "PromptProvider",
    "PromptRegistryError",
    "PromptUnavailableError",
]


# ============================================================
# 异常
# ============================================================
#
# 都同时继承一个内置异常：`PromptNotFoundError` 是 LookupError、
# `PromptRegistryError` 是 ValueError、`PromptUnavailableError` 是
# RuntimeError —— 这样既有的 `except ValueError` 之类的形状仍然接得住，
# 而调用方想精确区分时用基类 PromptError。


class PromptError(Exception):
    """提示词相关的错误基类。"""


class PromptNotFoundError(PromptError, LookupError):
    """
    要的提示词不存在。

    两种情形，报错信息里会指明是哪一种：

        model_role 在这张表里没有      （如 reasoning：主循环的提示词
                                        走 duty_prompt，不在这张表里）
        duty_key 不在职责表里           （拼错了，或那个职责还没登记）
    """


class PromptRegistryError(PromptError, ValueError):
    """提示词目录本身装配错了（重复 key、文件里没有 PROMPT…）。"""


class PromptUnavailableError(PromptError, RuntimeError):
    """
    职责提示词取不到 —— 通常是"应用没跑过 lifespan"（组合根没装
    ServiceFactory）。脚本与测试里是常态，生产上不该出现。
    """


# ============================================================
# 定义
# ============================================================


@dataclass(frozen=True, slots=True)
class PromptDefinition:
    """
    一份节点提示词。

    `key` 就是 **model_role**：`model.yaml` 里那个绑定用的角色，也是
    `get_runtime(model_role=...)` 的那个值。两者同一个词不是为了好记，
    而是它们本来就是同一件事的两面 —— "这次调用是干什么的"决定了用哪个
    模型，也决定了给它什么提示词。

    `render` 而不是 `text`：分类提示词要把运行时数据（能力候选清单）
    拼进去，摘要提示词一个字都不用改。两种都收进同一个形状，靠的就是
    "给的是可调用对象"—— 静态的那份写成 `lambda: TEXT` 形态的函数即可。

    刻意**不带 version**：文本本身就是身份，而"这一轮用的是哪版提示词"
    在观测里由 `context_hash`（覆盖 system 文本）回答。等提示词要支持
    A/B 或热更新时再加，那时它会有真实的消费者。
    """

    key: str

    description: str

    render: Callable[..., str]

    def __post_init__(self) -> None:

        if not isinstance(self.key, str) or not self.key.strip():
            raise PromptRegistryError(
                f"PromptDefinition.key 必须是非空字符串，收到 {self.key!r}。"
            )

        if not callable(self.render):
            raise PromptRegistryError(
                f"提示词 {self.key!r} 的 render 不可调用"
                f"（{type(self.render).__name__}）。"
            )

    def __call__(self, **variables: Any) -> str:
        """等价于 `render(**variables)` —— 让定义本身可以当函数用。"""

        return self.render(**variables)


# ============================================================
# 端口
# ============================================================


@runtime_checkable
class PromptProvider(Protocol):
    """
    取提示词的**唯一**入口（规格：所有取提示词的地方都从这里拿）。

    ⚠️ 两类提示词都在这一个端口上，但它们的**来源**完全不同（见模块
    docstring）—— 端口合一是为了调用方不必知道"这份提示词的存储在哪"，
    而不是说两类提示词是同一件事。
    """

    def node_prompt(self, model_role: str, /, **variables: Any) -> str:
        """
        节点功能性提示词（分类 / 摘要 / …）。

        Args:
            model_role: 见 `PromptDefinition.key`。
            **variables: 由该提示词的 `render` 签名决定；不匹配即 TypeError。

        Raises:
            PromptNotFoundError: 这个 model_role 没有登记提示词。
        """
        ...

    async def duty_key(self) -> str:
        """
        这次运行该用哪个职责 —— **「选职责」的唯一落点**。

        调用方（建运行那一步）拿它存进 `agent_run.duty_key`：运行行上因此
        能看到"这一轮用的是哪份职责"，写消息时也直接用它填
        `agent_message.business_agent_id`。

        与 `duty_prompt` 的分工：这个给 **key**（要落库、要传给下游），
        那个给 **文本**（喂给模型）。两者都出自同一次选择 —— 拿到 key 的
        调用方应当把它传给 `duty_prompt(key)`，而不是各自再选一次。

        Raises:
            PromptNotFoundError: 选出来的职责没有登记。
            PromptUnavailableError: 取不到职责表（没跑过 lifespan）。
        """
        ...

    async def duty_prompt(self, duty_key: str | None = None) -> str:
        """
        职责功能性提示词（主 agent 搭载的模型是干什么的）。

        Args:
            duty_key: 职责标识；None = 用默认职责（"先只硬编码一个"，
                      因此当前主循环传的就是 None —— 选职责的逻辑还没
                      有，等它会来自会话 / Agent 目录时再填。

        Raises:
            PromptNotFoundError: 这个职责没有登记。
            PromptUnavailableError: 取不到职责表（没跑过 lifespan）。
        """
        ...
