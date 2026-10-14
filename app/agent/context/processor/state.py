"""
State Path 解析：`when` 与 `inputs` 的**唯一**求值入口（规格 §八）。

为什么必须有这一个入口
--------------------
如果每个 Processor 自己 `state.get("context_policy", {}).get(...)`，
那么"字段不存在"的语义就有 N 种写法：

    有的返回 None      -> 看起来像"没配"，静默跳过
    有的返回 {}        -> 更隐蔽，后面的 if 永远为假
    有的抛 KeyError    -> 报错信息指向 Processor 内部，指不到缺哪个字段

而规格 §九要求 MISSING / FALSE / INVALID 三件事严格分开。只有把
求值收敛到一处，那个区分才是**机器保证**的，而不是每个 Processor
各自记得。

MISSING 是一个**单例哨兵**
-------------------------
`None` 不能当"缺失"用：`{"include_memory": None}` 是一个**存在且为
None** 的值，规格把它归到 INVALID（fail），而"键不存在"是 MISSING
（也 fail，但报错信息不同）。两者必须可区分，因此这里用一个哨兵。

哨兵同时也是**假值**（`__bool__` 返回 False）：万一有人漏了
`is MISSING` 的判断，`if value:` 会走"跳过"这条**保守**分支，而不是
凭空执行一个前置条件未知的 Processor。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .spec import StateBinding

__all__ = [
    "MISSING",
    "BoundInputs",
    "Missing",
    "resolve_inputs",
    "resolve_state_path",
]


class Missing:
    """
    "这条路径不存在"的哨兵类型（不要实例化别的 —— 只认模块级 MISSING）。

    单例的判据用 `is`，因此它没有 `__eq__` 重载：重载 `__eq__` 会让
    `value == MISSING` 也成立，而那种写法在 `value` 是 numpy 数组 /
    pydantic 模型时会走到它们的 `__eq__`，得到一堆奇怪的结果。
    """

    __slots__ = ()

    def __bool__(self) -> bool:  # pragma: no cover - 见 docstring
        return False

    def __repr__(self) -> str:
        return "<MISSING>"

    def __str__(self) -> str:
        return "<MISSING>"


#: 唯一实例。
MISSING = Missing()


@dataclass(frozen=True, slots=True)
class BoundInputs:
    """
    一次 `resolve_inputs` 的结果。

    `values` 是交给 Processor 的扁平 dict。**可选的缺失输入会以 None
    出现在里面** —— 那是 §十二 明确允许的形态（"除非 Processor 明确
    声明该 input 为 optional"），也是 Processor 能写
    `inputs["summary"]` 而不是到处 `.get()` 的前提。

    缺失本身仍然单独记一份（`missing`），因为：
        必需的缺失 -> 调用方必须 FAILED，不能继续
        可选的缺失 -> 只是观测信息（"这次没注进去"）
    而"传了 None"与"根本没这个输入"在 Processor 侧应当是同一种处理，
    所以值这一侧不需要再区分。
    """

    values: Mapping[str, Any]

    #: 缺失的输入名 -> 是否必需。
    missing: Mapping[str, bool]

    @property
    def missing_required(self) -> tuple[str, ...]:

        return tuple(
            sorted(name for name, required in self.missing.items() if required)
        )


def resolve_state_path(
        state: Any,
        path: str,
        *,
        default: Any = MISSING,
) -> Any:
    """
    按 dot notation 从 `state` 里取值。

    解析规则（三种形态，依次尝试）：

        Mapping          按键取
        对象属性         取属性（dataclass / pydantic / SimpleNamespace）
        其它             视为"这条路径到此为止" -> MISSING

    中间任何一段取不到（键不存在 / 段不是容器 / 值为 None 且还有
    后续段）都返回 MISSING（或 `default`，如果给了）。

    ⚠️ 值为 None 但**已是最后一段**时返回 None，不是 MISSING ——
    "存在且为 None"与"不存在"必须分得开（见模块 docstring）。

    ⚠️ 不做任何类型转换、不做隐式 bool —— 类型判断是调用方的事
    （gate 必须严格是 bool，见 pipeline.py）。
    """

    if not path:
        raise ValueError("resolve_state_path 的 path 不能为空。")

    segments = path.split(".")

    current: Any = state

    for segment in segments:

        if current is None:
            # 中途遇到 None：后面还有段，而 None 里取不出任何东西。
            return default

        if isinstance(current, Mapping):

            if segment not in current:
                return default

            current = current[segment]
            continue

        if hasattr(current, segment):

            current = getattr(current, segment)
            continue

        return default

    return current


def resolve_inputs(
        state: Any,
        bindings: Mapping[str, StateBinding],
) -> BoundInputs:
    """
    解析一组输入绑定（规格 §十一）。

    ContextBuilder 负责这一步，Processor 只拿到一个扁平 dict ——
    它因此**拿不到完整 State**（规格 §六：Processor 不依赖 Graph State）。
    """

    values: dict[str, Any] = {}
    missing: dict[str, bool] = {}

    for name, binding in bindings.items():

        value = resolve_state_path(state, binding.path)

        # 必需输入拿到 None 也算缺失（见 StateBinding 的 docstring）。
        if value is MISSING or (binding.required and value is None):

            missing[name] = binding.required

            if not binding.required:
                # 可选的缺失：以 None 交付（§十二 允许的形态）。
                values[name] = None

            continue

        values[name] = value

    return BoundInputs(values=values, missing=missing)
