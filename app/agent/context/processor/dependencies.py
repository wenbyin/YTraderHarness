"""
本次构建用到的**外部依赖**（规格 §十七、§二十九）。

为什么依赖走 request 而不是构造参数
--------------------------------
`BaseContextBuilder` 的实例是**共享的**，而且它的 `__setattr__` 被
封死（`base.py` 里那段"这不是风格要求，是并发正确性的保证"）。
因此依赖没有地方可以"存"：

    ❌ self._registry = ...        实例属性被禁（而且是共享实例）
    ❌ 类属性放可变依赖            所有实例共享，测试之间会互相污染

而构造注入也不可能：注册表按类反射、**零参构造**实例（见
context/registry.py 的说明）。因此依赖随**请求**进来：

    ContextBuildRequest(dependencies=ContextDependencies(model_registry=...))

缺省（None）时用进程级默认 —— 与 `get_model_registry()` /
`get_tool_registry()` / `get_service_factory()` 那套约定一致：进程级的
东西由进程级的取用函数给，不必层层透传。

⚠️ Processor 从 `context.dependencies` 取依赖
-------------------------------------------
Processor 自己**不**去摸全局变量（那样测试无法注入假实现，而且
"这次用的是哪个注册表"在观测里看不出来）。工作集带着依赖，Processor
就只认工作集。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.model import ModelRole
from app.agent.prompt import get_prompt_provider

from .budget import HeuristicTokenCounter, TokenCounter
from .history import HistorySelector, TurnWindowHistorySelector
from .policy import ContextPolicyResolver, DefaultContextPolicyResolver
from .sources import (
    CapabilityResolver,
    KnowledgeSource,
    MemorySource,
    ProfileSource,
    SkillSource,
    ToolExposureSource,
)
from .summarization import ModelRegistrySummaryCompactor, SummaryCompactor

__all__ = [
    "ContextDependencies",
    "dependencies_of",
    "get_context_dependencies",
    "reset_context_dependencies",
    "set_context_dependencies",
]


@dataclass(frozen=True, slots=True)
class ContextDependencies:
    """
    一次上下文构建的外部依赖束。

    全部可空，且**空有明确的含义**：

        model_registry 为空    -> 用进程级 `get_model_registry()`
        policy_resolver 为空   -> 用 `DefaultContextPolicyResolver()`
        token_counter 为空     -> 用 `HeuristicTokenCounter()`
        history_selector 为空  -> 用 `TurnWindowHistorySelector()`
        各 context source 为空 -> 用 State 里已经有的数据
        summary_compactor 为空 -> 按 model_registry + agent_type 现建一个

    这些都是"端口为空则回落"的形态，而不是"空就报错"：ContextBuilder
    在一条最小路径上（State 里已经有 messages + policy）必须能独立工作
    —— 否则"接一个新 Agent"的第一步就变成"先配齐五个服务"。
    """

    model_registry: Any = None

    policy_resolver: ContextPolicyResolver | None = None

    token_counter: TokenCounter | None = None

    history_selector: HistorySelector | None = None

    profile_source: ProfileSource | None = None
    memory_source: MemorySource | None = None
    knowledge_source: KnowledgeSource | None = None
    tool_source: ToolExposureSource | None = None

    #: task 模式的两条取数依赖。
    #:
    #: skill_source         skill_key -> SkillSnapshot
    #: capability_resolver  capability codes -> tool_keys
    #:
    #: 缺省时 SkillLoader 会降级：没有 skill 就用通用任务提示词，
    #: 没有 resolver 就给空工具集（不是回落到主 agent 的工具）。
    skill_source: SkillSource | None = None
    capability_resolver: CapabilityResolver | None = None

    summary_compactor: SummaryCompactor | None = None

    # ========================================================
    # 解析（缺省 -> 默认实现）
    # ========================================================

    def resolved_model_registry(self) -> Any:
        """
        模型注册表。

        缺省取**进程级**注册表（`app.agent.model.get_model_registry()`）
        —— 与工具注册表同一形状。它是懒加载的：真去读 model.yaml 发生在
        第一次调用时，因此"没有配模型目录"的部署不会在 import 阶段炸。
        """

        if self.model_registry is not None:
            return self.model_registry

        from app.agent.model import get_model_registry

        return get_model_registry()

    def resolved_policy_resolver(self) -> ContextPolicyResolver:

        return self.policy_resolver or DefaultContextPolicyResolver()

    def resolved_token_counter(self) -> TokenCounter:

        return self.token_counter or HeuristicTokenCounter()

    def resolved_history_selector(self) -> HistorySelector:

        return self.history_selector or TurnWindowHistorySelector()

    def resolved_skill_source(self) -> SkillSource:
        """
        skill 源。

        缺省用 `ServiceSkillSource()`（按需查 SkillService）—— 与
        tool_source 的默认实现同一形状：它只在 task 模式的 SkillLoader
        里被调用，普通对话不会碰数据库。
        """

        if self.skill_source is not None:
            return self.skill_source

        from app.infra.agent.skill_exposure import ServiceSkillSource

        return ServiceSkillSource()

    def resolved_capability_resolver(self) -> CapabilityResolver:
        """能力 code -> tool_keys；缺省用服务实现。"""

        if self.capability_resolver is not None:
            return self.capability_resolver

        from app.infra.agent.capability_resolver import (
            ServiceCapabilityResolver,
        )

        return ServiceCapabilityResolver()

    def resolved_compactor(
            self,
            *,
            agent_type: str,
    ) -> SummaryCompactor:
        """
        摘要器（名字里的 compactor 是历史命名 —— 它现在的调用方是
        SUMMARY_LOADER 阶段，见 processors/summary.py）。

        没注入就按 ModelRegistry 现建一个 —— 注意它**只持有注册表**，
        真正的模型调用发生在 `compact()` 里（那时才 `get_runtime`）。
        因此"建摘要器"不产生任何网络或客户端构造。

        ⚠️ 提示词在这里取，然后**交给**摘要器（而不是让它自己去摸）
        ------------------------------------------------------
        `node_prompt(ModelRole.SUMMARIZATION)` 是同步的（提示词是进程级
        常量，见 app/agent/prompt/contracts.py），所以这行不引入任何异步
        成本。提示词只有一个来源，摘要器自带正文的那个默认值已经删掉 ——
        否则"哪一份生效"就取决于是谁构造的它。
        """

        if self.summary_compactor is not None:
            return self.summary_compactor

        return ModelRegistrySummaryCompactor(
            model_registry=self.resolved_model_registry(),
            agent_type=agent_type,
            system_prompt=get_prompt_provider().node_prompt(
                ModelRole.SUMMARIZATION
            ),
        )

    def describe(self) -> dict[str, Any]:
        """这份依赖束里**配了哪些**（进日志/快照，不含任何秘密）。"""

        return {
            "injected": sorted(
                name
                for name in (
                    "model_registry",
                    "policy_resolver",
                    "token_counter",
                    "history_selector",
                    "profile_source",
                    "memory_source",
                    "knowledge_source",
                    "tool_source",
                    "skill_source",
                    "capability_resolver",
                    "summary_compactor",
                )
                if getattr(self, name) is not None
            )
        }

    @classmethod
    def default(cls) -> "ContextDependencies":
        """全空的依赖束（一切走进程级默认）。"""

        return cls()


# ============================================================
# 进程级默认
# ============================================================
#
# 与 set_model_registry / reset_factories 同一形状：进程级事实一个进程
# 一份，测试可以整份替换。

_dependencies: ContextDependencies | None = None


def get_context_dependencies() -> ContextDependencies:
    """取进程级依赖束（没有就用全空的默认）。"""

    global _dependencies

    if _dependencies is None:
        _dependencies = ContextDependencies.default()

    return _dependencies


def set_context_dependencies(
        dependencies: ContextDependencies | None,
) -> None:
    """
    替换进程级依赖束。

    组合根用它注入定制依赖（真的 ModelRegistry / 外部上下文源）；
    测试用它注入假的。传 None 等价于 reset。
    """

    global _dependencies

    _dependencies = dependencies


def reset_context_dependencies() -> None:
    """丢弃进程级依赖束（**只给测试用**）。"""

    global _dependencies

    _dependencies = None


def dependencies_of(context: Any) -> ContextDependencies:
    """
    从工作集里取依赖束（没有就用进程级默认）。

    这是 Processor 唯一的取依赖入口 —— 它们因此不必（也不该）import
    进程级单例。
    """

    injected = getattr(context, "dependencies", None)

    if isinstance(injected, ContextDependencies):
        return injected

    return get_context_dependencies()
