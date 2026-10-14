"""
Model Resolver：`(agent_type, model_role)` -> Model Profile。

Resolver 的职责**非常克制**（规格 §5）
------------------------------------
它是：

    一个查表 + 等值匹配

它**不是**：

    ❌ LLM（不调用模型）
    ❌ 自然语言理解（不看用户输入）
    ❌ 语义分类（不判断"这句话该走哪条路"）
    ❌ Capability Selection（不决定 Agent 该有什么能力）
    ❌ Agent Builder / JEV 的第二份实现

一句话：**Agent Type 已经被别人决定好了**，Resolver 只回答"那它该用
哪套模型策略"。

与 Agent Builder / JEV 的边界（规格 §6）
--------------------------------------
    Agent Builder / JEV
        ↓ 决定使用什么 Agent / Capability / Agent Type
    Model Resolver
        ↓ 根据**已经确定的** Agent Type + Model Role 决定 Model Profile

因此本模块里没有、也不该有"再问一次 JEV"或"读一下用户消息"的代码。
一旦有了，Resolver 就会开始与 Agent Builder 抢同一个决定权 —— 症状是
"有时候走 A 模型有时候走 B 模型"，而两处都觉得自己是对的。

为什么返回 ModelProfile 而不是 profile id
---------------------------------------
规格里两种形态都提过（`reasoning_default` 或直接 `ModelProfile`）。
返回对象：Runtime 需要 primary / fallbacks / policy 三样东西，返回
字符串会让调用方再查一次表，而多一次查表就多一处"查不到怎么办"。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .errors import ModelConfigurationError, ModelResolutionError
from .types import ModelProfile, ModelResolveContext

__all__ = [
    "CatalogModelResolver",
    "ModelResolver",
    "ModelRoleBinding",
]


#: 通配符：`agent_type` / `model_role` 里的"任意"。
#:
#: 用它而不是 None，是因为它会进配置（YAML 里的 `"*"`）与日志 ——
#: 一个能写进 YAML 的可见符号比"省略这个键"更难被误读成漏填。
WILDCARD = "*"


@dataclass(frozen=True, slots=True)
class ModelRoleBinding:
    """
    一条绑定：`(agent_type, model_role)` -> profile。

    attributes 是**可选的收窄条件**：只有全部等值命中才用这条。它让
    同一个 (agent_type, model_role) 能按部署差异选到不同 profile
    （`{"tenant": "acme"}` -> 专用的高配 profile）。

    ⚠️ attributes 只做**等值匹配**，不做表达式 / 优先级 / 条件语言
    -----------------------------------------------------------------
    一个能算条件的小语言看起来很方便，但它是 Resolver 滑向"第二个
    Agent Builder"的入口。要按业务条件选模型，那是 Agent Builder 的
    决定，应当体现为**不同的 agent_type**（或不同的 model_role），
    而不是在这里写 if。
    """

    agent_type: str
    model_role: str

    #: profile 的 id 或**别名**（`reasoning_default`）。
    profile: str

    attributes: Mapping[str, str] = field(default_factory=dict)

    description: str = ""

    def __post_init__(self) -> None:

        if not self.agent_type.strip():
            raise ValueError("ModelRoleBinding.agent_type 不能为空（用 '*' 表示任意）。")

        if not self.model_role.strip():
            raise ValueError("ModelRoleBinding.model_role 不能为空（用 '*' 表示任意）。")

        if not self.profile.strip():
            raise ValueError(
                f"绑定 {self.agent_type}+{self.model_role} 没有给出 profile。"
            )

        object.__setattr__(
            self,
            "attributes",
            {str(key): str(value) for key, value in (self.attributes or {}).items()},
        )

    # --------------------------------------------------------

    def matches(
            self,
            agent_type: str,
            model_role: str,
            attributes: Mapping[str, str],
    ) -> bool:
        """三件事都命中才算匹配。"""

        if self.agent_type not in (WILDCARD, agent_type):
            return False

        if self.model_role not in (WILDCARD, model_role):
            return False

        return all(
            attributes.get(key) == value
            for key, value in self.attributes.items()
        )

    @property
    def specificity(self) -> int:
        """
        有多具体。命中多条时取最大的那条。

        计数方式：非通配的 agent_type / model_role 各算 1 分，每个
        属性约束算 1 分。于是

            ("react", "reasoning", {})                      2 分
            ("react", "reasoning", {"tenant": "a"})         3 分
            ("*",     "*",         {})                      0 分

        这正是"更具体的绑定优先"这条直觉。**平局不是靠顺序解决的** ——
        见 CatalogModelResolver.resolve。
        """

        return (
            (0 if self.agent_type == WILDCARD else 1)
            + (0 if self.model_role == WILDCARD else 1)
            + len(self.attributes)
        )

    def describe(self) -> dict[str, object]:

        return {
            "agent_type": self.agent_type,
            "model_role": self.model_role,
            "profile": self.profile,
            "attributes": dict(self.attributes),
            "description": self.description,
        }


@runtime_checkable
class ModelResolver(Protocol):
    """
    `(agent_type, model_role)` -> ModelProfile。

    实现只允许做**确定性的映射**：同样的输入给同样的输出，不查网络、
    不调模型、不看时间。理由是它跑在每次模型调用之前（Registry 的
    `get_runtime`），任何不确定性都会变成"同一个节点有时用这个模型
    有时用那个"，且无法从日志复现。
    """

    def resolve(
            self,
            *,
            agent_type: str,
            model_role: str,
            context: ModelResolveContext | None = None,
    ) -> ModelProfile:
        ...


class CatalogModelResolver:
    """
    由一份**绑定表**驱动的 Resolver —— 默认实现。

    `profile_lookup` 是 Registry 交进来的查表函数（`ref -> ModelProfile`，
    含别名解析）。用函数而不是复制一份 profile 字典，是为了：

        注册顺序无关    先注册 resolver 再注册 profile 也不会失效
        单一副本        profile 只在 Registry 里存一份，不会两边漂移
        无循环依赖      resolver 模块不 import registry 模块

    ⚠️ 注册期只做**重复检测**，不做"能不能解析"的检查
    --------------------------------------------------
    `(agent_type, model_role, attributes)` 完全相同的两条绑定在这里就
    报错。而"profile 存不存在"留给第一次解析 —— 那可能是 profile 还没
    注册完（组合根按 providers -> models -> profiles -> bindings 的顺序
    装配，绑定放最后本来就不会撞上）。
    """

    def __init__(
            self,
            *,
            bindings: Sequence[ModelRoleBinding],
            profile_lookup: Callable[[str], ModelProfile],
    ) -> None:

        self._bindings = tuple(bindings)
        self._lookup = profile_lookup

        self._assert_unique()

    # ========================================================
    # 解析
    # ========================================================

    def resolve(
            self,
            *,
            agent_type: str,
            model_role: str,
            context: ModelResolveContext | None = None,
    ) -> ModelProfile:
        """
        定出这次该用的 Model Profile。

        解析不出来就**抛错**，不回落到"某个默认模型"：静默回落会让
        一次配置遗漏表现为"模型答得不太对"（而且很可能一直没人发现），
        而报错直接指出缺哪条绑定、现有的是哪些。
        """

        attributes = dict(context.attributes) if context else {}

        matched = [
            binding
            for binding in self._bindings
            if binding.matches(agent_type, model_role, attributes)
        ]

        if not matched:
            raise ModelResolutionError(
                f"没有可用于 agent_type={agent_type!r} + "
                f"model_role={model_role!r} 的模型绑定。\n"
                f"当前登记的绑定：{self._known()}\n"
                "两条路：在 model.yaml 的 bindings: 里加一条，或在代码里调 "
                "ModelRegistry.register_binding()。\n"
                "（注意：Resolver **不会**替你挑一个默认模型 —— 静默回落会"
                "让一次配置遗漏表现为“模型答得不太对”。）"
            )

        best = max(binding.specificity for binding in matched)

        finalists = [
            binding for binding in matched if binding.specificity == best
        ]

        if len(finalists) > 1:
            # 平局必须报错：挑一个的后果是"行为取决于绑定表的书写顺序"，
            # 而顺序在 YAML 里看起来无关紧要。
            raise ModelConfigurationError(
                f"agent_type={agent_type!r} + model_role={model_role!r} "
                f"（attributes={attributes!r}）命中了多条同等具体的绑定："
                f"{[b.describe() for b in finalists]}。\n"
                "请消歧：把其中一条写得更具体（加 attributes），或删掉"
                "多余的 —— Resolver 不会按书写顺序猜。"
            )

        binding = finalists[0]

        return self._resolve_profile(binding, agent_type, model_role)

    # ========================================================
    # 目录
    # ========================================================

    def bindings(self) -> tuple[ModelRoleBinding, ...]:
        """全部绑定（目录接口与测试用）。"""

        return self._bindings

    def describe(self) -> list[dict[str, object]]:

        return [
            {
                **binding.describe(),
                "specificity": binding.specificity,
            }
            for binding in self._bindings
        ]

    # ========================================================
    # internals
    # ========================================================

    def _resolve_profile(
            self,
            binding: ModelRoleBinding,
            agent_type: str,
            model_role: str,
    ) -> ModelProfile:

        try:
            return self._lookup(binding.profile)

        except ModelResolutionError:
            raise

        except Exception as exc:

            raise ModelResolutionError(
                f"绑定 {agent_type!r}+{model_role!r} 指向的 profile "
                f"{binding.profile!r} 无法解析：{exc}",
                details={
                    "agent_type": agent_type,
                    "model_role": model_role,
                    "profile_ref": binding.profile,
                },
            ) from exc

    def _assert_unique(self) -> None:
        """
        同一个 `(agent_type, model_role, attributes)` 只允许一条绑定。

        重复的后果是**不确定**（两条一样具体 -> 平局报错，于是问题
        直到运行期才暴露，且报错说的是"歧义"而不是"你写了两遍"）。
        """

        seen: dict[tuple[str, str, tuple[tuple[str, str], ...]], int] = {}

        for index, binding in enumerate(self._bindings):

            key = (
                binding.agent_type,
                binding.model_role,
                tuple(sorted(binding.attributes.items())),
            )

            if key in seen:
                raise ModelConfigurationError(
                    f"绑定重复：{binding.agent_type!r} + "
                    f"{binding.model_role!r} + "
                    f"{dict(binding.attributes)!r} 在第 "
                    f"{seen[key]} 与第 {index} 条各出现一次。\n"
                    "同一个键只该有一条绑定 —— 两条一样的绑定会让解析"
                    "在运行期报“歧义”，而真正的原因是这里写重了。"
                )

            seen[key] = index

    def _known(self) -> list[dict[str, object]] | str:
        """
        报错里列出现有绑定。

        带 specificity 与 attributes：看日志的人要能一眼判断"是不是
        我这条被另一条更具体的挡住了"。
        """

        if not self._bindings:
            return "（无）"

        return [
            {
                "agent_type": binding.agent_type,
                "model_role": binding.model_role,
                "profile": binding.profile,
                "attributes": dict(binding.attributes),
                "specificity": binding.specificity,
            }
            for binding in self._bindings
        ]
