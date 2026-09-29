"""
隐私过滤：把参数与结果里的敏感信息抹掉。

挂两半，各管一头
----------------
    BEFORE  抹**出去的**参数（工具不该看到的东西，比如用户手机号）
    AFTER   抹**回来的**结果（结果不该流向模型/前端的东西）

两者都是 SYNC —— 转换器靠**返回值**交付改过的数据，ASYNC 的返回值
会被丢弃（见 base.py 里那条定义期校验）。

⚠️ 它是"最佳努力"，不是合规手段
------------------------------
正则匹配做不到"保证没有敏感信息"：变形写法、编码、图片里的身份证
都拦不住。它的定位是**降低泄漏面**，不是审计合规的替代品。
链上挂了它，不等于这条链处理过的数据就合规了。

⚠️ 抹掉之后就取不回来了
----------------------
参数是**就地改**的（契约见 context.py：arguments 是本次调用私有的
副本），结果返回的是**新对象**。两者都意味着下游拿到的是抹过的
版本 —— 包括审计与事件订阅者。这是刻意的：如果原件还能在别处取到，
"抹掉"就只是给模型看的障眼法。

需要留原件做排查时，别在这里留，去**调用日志**里留（那是另一个
系统的事，且它有自己的访问控制）。
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

from ..base import ToolTransformer
from ..context import ToolInvocation, ToolOutcome
from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

__all__ = [
    "PrivacyConfig",
    "PrivacyTarget",
    "PrivacyToolProcessor",
    "scrub_text",
    "scrub_value",
]

logger = logging.getLogger(__name__)


class PrivacyTarget(str, Enum):
    """
    要抹掉的敏感信息种类。

    取值进 config_schema 的 enum，前端渲染成多选框。
    """

    PHONE = "PHONE"
    EMAIL = "EMAIL"
    ID_CARD = "ID_CARD"
    BANK_CARD = "BANK_CARD"


#: 每种目标的正则。
#:
#: 两侧的 `(?<!\\d)` / `(?!\\d)` 是**边界保护**：少了它，
#: `13812345678` 会在一串 20 位的订单号里被命中，把订单号挖掉一块 ——
#: 那比不抹更糟，因为结果看起来"正常"却已经损坏。
#:
#: ⚠️ 顺序有意义，见 _ORDERED_TARGETS 的说明。
_PATTERNS: dict[PrivacyTarget, re.Pattern[str]] = {
    PrivacyTarget.ID_CARD: re.compile(
        r"(?<!\d)\d{17}[\dXx](?!\d)"
    ),
    PrivacyTarget.BANK_CARD: re.compile(
        r"(?<!\d)\d{16,19}(?!\d)"
    ),
    PrivacyTarget.PHONE: re.compile(
        r"(?<!\d)1[3-9]\d{9}(?!\d)"
    ),
    PrivacyTarget.EMAIL: re.compile(
        r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
    ),
}

#: 应用顺序：**长模式在前**。
#:
#: 18 位身份证同时满足 ID_CARD 与 BANK_CARD 两条正则。先跑长的，
#: 它被换成掩码之后就不含数字了，后面的短模式自然不会再命中它。
#: 反过来会让同一条信息被抹两次，而且第二次抹的是掩码的一部分 ——
#: 输出仍然是错的，只是错得不明显。
_ORDERED_TARGETS: tuple[PrivacyTarget, ...] = (
    PrivacyTarget.ID_CARD,
    PrivacyTarget.BANK_CARD,
    PrivacyTarget.PHONE,
    PrivacyTarget.EMAIL,
)


class PrivacyConfig(BaseModel):
    """
    隐私过滤参数。

    ⚠️ 刻意**没有** "只过滤某些字段" 这类开关
    ----------------------------------------
    那种开关看起来很有用（"只抹 phone 字段"），但它把"哪里可能有
    敏感信息"这个判断推给了配置的人 —— 而配错的表现是**静默漏掉**，
    正是这个处理器最不该有的失败模式。宁可多抹：抹掉的字段工具可能
    会不高兴，漏掉的字段没人会发现。
    """

    targets: list[PrivacyTarget] = Field(
        default_factory=lambda: [
            PrivacyTarget.PHONE,
            PrivacyTarget.EMAIL,
            PrivacyTarget.ID_CARD,
        ],
        title="过滤项",
        description=(
            "要抹掉的敏感信息种类。**默认不含银行卡号** —— "
            "它的正则（16~19 位连续数字）会命中金额、订单号、"
            "时间戳这类正常数据，误伤面太大。真要抹银行卡，"
            "请确认这条链的参数与结果里不会有长数字串。"
        ),
    )

    mask: str = Field(
        default="[REDACTED]",
        min_length=1,
        max_length=64,
        title="替换文本",
        description=(
            "抹掉之后填什么。留空串是允许的（直接删掉），"
            "但那样会改变文本长度与可读性，排查时看不出这里原本"
            "有东西 —— 默认给一个显眼的占位。"
        ),
    )

    keep_length: bool = Field(
        default=False,
        title="保留长度",
        description=(
            "用等长的 `*` 替换，而不是统一填掩码。"
            "好处是排查时还能看出「这里原本是个手机号」；"
            "代价是**长度本身也是信息**（身份证的长度就能说明"
            "它是身份证）。默认不保留。"
        ),
    )

    max_depth: int = Field(
        default=6,
        ge=1,
        le=32,
        title="最大深度",
        description=(
            "嵌套结构往下走几层。防止自引用的结构把递归拖死；"
            "超过这个深度的那部分**不抹**（不是抹成整块），"
            "因为整块替换会破坏结构，让工具拿到看不懂的东西。"
        ),
    )


class PrivacyToolProcessor(ToolTransformer):
    """
    抹掉参数与结果里的敏感信息。

    只处理**字符串叶子**：dict / list / tuple / pydantic 模型都会
    递归进去，但数字、布尔、None、字节、以及任意对象**原样留着**。
    不猜 —— 对一个不认识的类型做 str() 再还原，只会把对象变成一个
    字符串，那比漏抹更糟。

    dict 的**键不抹**：改键名会让下游按名字取值的代码取不到东西，
    而键里出现手机号的情形远少于值里。
    """

    name = "privacy_transformer"

    display_name = "隐私过滤"

    description = (
        "抹掉工具参数与返回结果中的手机号、邮箱、证件号等敏感信息。"
    )

    config_model: ClassVar[type[BaseModel]] = PrivacyConfig

    supported_hooks: ClassVar[frozenset[ProcessorHook]] = (
        frozenset({
            ProcessorHook.BEFORE,
            ProcessorHook.AFTER,
        })
    )

    #: 只能 SYNC：转换器靠返回值交付结果（见模块 docstring）。
    supported_execution_modes: ClassVar[
        frozenset[ExecutionMode]
    ] = frozenset({ExecutionMode.SYNC})

    # ========================================================
    # BEFORE：抹参数
    # ========================================================

    async def before(
            self,
            invocation: ToolInvocation,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolInvocation:

        parsed = self.config_model.model_validate(config)

        # 就地改：arguments 是本次调用私有的副本（见 context.py），
        # 改它不会影响调用方手里的原始参数，但会让**下游的每一个
        # 处理器**（含事件订阅者）都只看到抹过的版本。
        invocation.arguments.update(
            scrub_value(
                invocation.arguments,
                config=parsed,
            )
        )

        return invocation

    # ========================================================
    # AFTER：抹结果
    # ========================================================

    async def after(
            self,
            invocation: ToolInvocation,
            outcome: ToolOutcome,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolOutcome:

        # 失败的结果没有 value 可抹。错误消息里可能有敏感信息
        # （"查不到用户 13812345678"），但那是**异常对象**，
        # 改它要动异常本身 —— 见下面 _scrub_error 的说明。
        if outcome.error is not None:
            return outcome

        parsed = self.config_model.model_validate(config)

        return ToolOutcome(
            value=scrub_value(outcome.value, config=parsed),
            error=outcome.error,
            duration_ms=outcome.duration_ms,
            cached=outcome.cached,
        )


# ============================================================
# 抹除逻辑（纯函数，与处理器解耦，便于单独验证）
# ============================================================


def scrub_value(value: Any, *, config: PrivacyConfig) -> Any:
    """
    递归抹掉一个值里的敏感信息。

    认识的容器会**产出新对象**（不就地改传进来的那个）：调用方
    可能还持有原对象（例如工具返回的 pydantic 模型），就地改会
    改到它头上，而那是"工具自己的数据"，不该被一个观察链路里的
    处理器动过。
    """

    return _scrub(value, config=config, depth=0)


def scrub_text(text: str, *, config: PrivacyConfig) -> str:
    """抹一段文本。给单测与外部调用用。"""

    redacted = text

    for target in _ORDERED_TARGETS:

        if target not in config.targets:
            continue

        pattern = _PATTERNS[target]

        if config.keep_length:
            redacted = pattern.sub(
                lambda match: "*" * len(match.group(0)),
                redacted,
            )
        else:
            redacted = pattern.sub(
                lambda _: config.mask,
                redacted,
            )

    return redacted


def _scrub(
        value: Any,
        *,
        config: PrivacyConfig,
        depth: int,
) -> Any:

    if depth > config.max_depth:
        # 超深**原样返回**，不抹成整块 —— 那会破坏结构。
        return value

    if isinstance(value, str):
        return scrub_text(value, config=config)

    if isinstance(value, dict):
        # 键不抹（理由见类 docstring）。
        return {
            key: _scrub(
                item, config=config, depth=depth + 1
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return type(value)(
            _scrub(item, config=config, depth=depth + 1)
            for item in value
        )

    if isinstance(value, BaseModel):
        return _scrub_model(value, config=config, depth=depth)

    # 数字 / 布尔 / None / 字节 / 任意对象 —— 原样留着。
    # 不猜：对不认识的类型做 str() 再还原只会把对象变成字符串。
    return value


def _scrub_model(
        value: BaseModel,
        *,
        config: PrivacyConfig,
        depth: int,
) -> BaseModel:
    """
    pydantic 模型：逐个字段抹，产出**新的模型实例**。

    用 model_copy(update=...) 而不是 model_validate(model_dump(...))：
    后者会把嵌套里的自定义类型拍平成 dict，返回的东西类型变了 ——
    而调用方拿到的是"工具的返回值"，形状不该被一个过滤器改掉。

    ⚠️ 只处理声明过的字段。`model_extra`（允许额外字段的模型）
    不碰 —— 那些字段没有类型信息，改它们风险更大。
    """

    updates = {
        name: _scrub(
            getattr(value, name),
            config=config,
            depth=depth + 1,
        )
        for name in type(value).model_fields
    }

    return value.model_copy(update=updates)
