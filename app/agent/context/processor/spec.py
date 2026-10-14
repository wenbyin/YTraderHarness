"""
Context Processor 的**自描述契约**。

规格（`app/services/context/context_builder.yaml`）里有两条硬约束：

    1. 阶段顺序由服务端定义，客户端不可变
    2. 阶段可以被 runtime policy 启用/跳过，但**不能新增或重排**

本模块把"一个 Context Processor 如何声明自己"固定下来，让
ContextBuilder 只需要解析、排序、执行 —— 而不需要认识任何一个具体
Processor（规格 §七：ContextBuilder 里不许出现
`if processor.name == "history"`）。

Processor 自描述四件事（规格 §五）
--------------------------------
    name    我是谁（注册键 / 观测里的 processor 名）
    order   我什么顺序（**必须与固定阶段顺序一致**，见 pipeline.py）
    when    从 State 的**哪个路径**读"我这次要不要执行"
    inputs  我需要从 State 的**哪些路径**取输入，映射到我参数里的哪个名字

三者里的路径都是 dot notation，由 `state.py` 的解析器统一解析 ——
Processor **不**自己 `state.get(...)`（规格 §八）。

⚠️ `when` 的四个语义（规格 §九，强制）
-----------------------------------
    MISSING（路径不存在）      -> ContextBuilder FAILED
    FOUND + False             -> SKIPPED（合法）
    FOUND + True              -> EXECUTE
    FOUND + 非 bool（含 None） -> ContextBuilder FAILED（**不做隐式 bool 转换**）

`None` 被明确归到"非 bool"那一侧：`bool(None) == False` 看起来像
"跳过"，但它的真实含义通常是"上游忘了赋值"，静默跳过正是最难查的
那类问题。

⚠️ `when=None` 的唯一例外
-----------------------
`when=None` 表示**无条件执行**，只允许出现在该 Builder 声明为
`required` 的阶段上（`base.required_stages`）。原因与既有
`BaseContextBuilder` 对 `CONTEXT_RESOLVER` / `PROMPT_ASSEMBLER` 的
处理完全一致：

    入口阶段先有鸡才有蛋 —— 策略与预算由它产出，它不可能由策略门控
    出口阶段是产出的唯一来源 —— 少了它 build() 必然"没有产出任何请求"

两条规则在 pipeline.py 里双向校验（required 阶段必须无条件、
无条件处理器必须在 required 阶段上），因此"配了 gate 却被声明为
required"这种自相矛盾在**启动期**就会炸。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.common.context import StageId

__all__ = [
    "ContextProcessorSpec",
    "StateBinding",
]


@dataclass(frozen=True, slots=True)
class StateBinding:
    """
    "我的某个输入来自 State 的哪条路径"。

    规格 §十三允许两种写法，本模块**两种都收**（在
    `ContextProcessorSpec.__post_init__` 里归一）：

        inputs={"user_id": "user_id"}                             简写
        inputs={"user_id": StateBinding("user_id")}               显式

    简写是常态（绝大多数输入都是必需的），显式只在需要标注
    `required=False` 时用。

    required 的语义（规格 §十二）
    ----------------------------
        True（默认）  路径不存在 -> ContextBuilder FAILED
        False         路径不存在 -> 传 None，Processor 自己决定怎么办

    什么时候该用 `required=False`：**数据本来就可能没有**的输入。
    例如"还没有历史摘要"是完全正常的状态，把 `summary` 声明成必需
    会让每一段新会话的第一次调用直接失败。

    什么时候**不能**用：接线错误。`messages`、`run.phase`、
    `context_policy.include_memory` 这类东西缺失意味着上游漏配，
    继续跑只会让报错离原因更远。

    ⚠️ 必需输入拿到 `None` 也按缺失处理
    ----------------------------------
    一个必需的 `user_id` 拿到 None 与没有它一样没用，两者都该 FAILED。
    （`when` 的 None 是另一回事：那里连 None 都必须是 bool，见 §九。）
    """

    path: str

    required: bool = True

    def __post_init__(self) -> None:

        path = str(self.path).strip()

        if not path:
            raise ValueError("StateBinding.path 不能为空。")

        if path.startswith(".") or path.endswith(".") or ".." in path:
            raise ValueError(
                f"StateBinding.path 不是合法的 dot path：{self.path!r}。"
                "路径形如 context_policy.include_memory，"
                "不能有空段、也不该带前后的点。"
            )

        object.__setattr__(self, "path", path)

    def describe(self) -> dict[str, Any]:

        return {"path": self.path, "required": self.required}


@dataclass(frozen=True, slots=True)
class ContextProcessorSpec:
    """
    一个 Context Processor 的**完整自描述**。

    它是类级常量（`ClassVar`），因此：
        反射发现时不必实例化就能读（注册表与 `/context-builders` 目录）
        实例是共享的、无状态的（见 base.py）

    Parameters
    ----------
    name
        注册键，也是观测里的 `processor` 名。一旦发布**不要改名** ——
        它会进观测行与 Context Snapshot，改名等于让历史运行对不上。
    order
        全局单调的执行序。**必须与固定阶段顺序一致**（pipeline.py 会
        校验"按 order 排"与"按阶段模板排"得到同一个序列），这样
        阶段的不可变契约与 Processor 的自描述不会互相打架。
    stage
        它服务 `StageId` 里的哪一个阶段。它是两套契约的桥：
        Builder 的 `enabled_stages` / `required_stages` 按阶段说话，
        Processor 按自己的 spec 说话。
    when
        `StateBinding` 或路径字符串；**None = 无条件执行**（只允许
        required 阶段，见模块 docstring）。
    inputs
        `参数名 -> StateBinding | 路径`。
    optional
        这个 Processor 失败是否**不算致命**（规格 §二十八）。
        True 时失败记 FAILED + 一条 warning，流水线继续跑。
        典型用途：Memory / 外部检索这类"拿不到也能答"的旁路信息。
    description
        给人看的一句话（目录接口用）。
    """

    name: str

    order: int

    stage: StageId

    when: StateBinding | str | None = None

    inputs: Mapping[str, StateBinding | str] = field(
        default_factory=dict
    )

    optional: bool = False

    description: str = ""

    def __post_init__(self) -> None:

        name = str(self.name).strip()

        if not name:
            raise ValueError("ContextProcessorSpec.name 不能为空。")

        if isinstance(self.order, bool) or not isinstance(self.order, int):
            raise ValueError(
                f"ContextProcessorSpec({name}).order 必须是整数，"
                f"得到 {self.order!r}。"
            )

        if self.order < 0:
            raise ValueError(
                f"ContextProcessorSpec({name}).order 不能为负。"
            )

        if not isinstance(self.stage, StageId):
            raise ValueError(
                f"ContextProcessorSpec({name}).stage 必须是 StageId，"
                f"得到 {self.stage!r}。阶段是服务端定义且不可变的"
                "（见 services/context/context_builder.yaml）。"
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "when", _as_binding(self.when, name, "when"))
        object.__setattr__(
            self,
            "inputs",
            {
                str(key): _as_binding(value, name, f"inputs.{key}")
                for key, value in (self.inputs or {}).items()
            },
        )

    # --------------------------------------------------------

    @property
    def when_path(self) -> str | None:
        """gate 的 State 路径；None = 无条件执行。"""

        return self.when.path if self.when is not None else None

    @property
    def unconditional(self) -> bool:
        return self.when is None

    def input_names(self) -> tuple[str, ...]:

        return tuple(sorted(self.inputs))

    def required_inputs(self) -> tuple[str, ...]:

        return tuple(
            sorted(
                name
                for name, binding in self.inputs.items()
                if binding.required
            )
        )

    def describe(self) -> dict[str, Any]:
        """
        给目录 / 观测用的一份描述。

        字段名刻意贴 `context_builder.yaml` 的 `StagePolicy`：
        `execute_when` / `input_contract` / `required` —— 将来
        `/context-builders` 接口从代码派生那份目录时不必再猜形状。
        """

        return {
            "name": self.name,
            "order": self.order,
            "stage": self.stage.value,
            "required": self.unconditional,
            "execute_when": (
                [self.when.path] if self.when is not None else []
            ),
            "input_contract": {
                "fields": [
                    {
                        "name": name,
                        "type": "state_path",
                        "required": binding.required,
                        "description": binding.path,
                    }
                    for name, binding in sorted(self.inputs.items())
                ]
            },
            "optional": self.optional,
            "description": self.description,
        }


# ============================================================
# internals
# ============================================================


def _as_binding(
        value: StateBinding | str | None,
        processor: str,
        field_name: str,
) -> StateBinding | None:
    """把路径字符串归一成 StateBinding（None 保持 None）。"""

    if value is None:
        return None

    if isinstance(value, StateBinding):
        return value

    if isinstance(value, str):
        return StateBinding(path=value)

    raise ValueError(
        f"ContextProcessorSpec({processor}).{field_name} 只能是 "
        f"StateBinding / 路径字符串 / None，得到 {type(value).__name__}。"
    )
