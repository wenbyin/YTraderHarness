"""
阶段 1：CONTEXT_RESOLVER —— 入口处理器（规格 §五 的"参数解析"）。

它跑在**没有任何 gate**的情况下（`when=None`），因为：

    策略与预算由它产出，它不可能由策略门控（先有鸡才有蛋）

这与既有 `BaseContextBuilder` 对 CONTEXT_RESOLVER 的处理逐字一致
（那边它是唯一有真实默认值的阶段，且永远先跑），也是规格里
`required_stage` 的含义。

它做两件事，都写进 overlay（于是后续处理器可以用 State Path 读到）：

    1. `context_policy` = 基线策略 ⊕ 图 state 的覆盖项（**不转换类型**）
    2. `context_budget` = 按模型窗口与策略上限算出来的预算

模型窗口来自 **ModelRegistry**（规格 §二十一）：拿这次要用的 profile 的
主模型定义，读它的 `context_window`。拿不到时回落到
`policy.default_context_window` 并记一条 warning —— 兜底是**可见**的
（写进快照的 `context_window_source`），而不是静默发生。

⚠️ 为什么预算失败不算致命
--------------------
真正"无法获得有效 Model"的失败发生在 **Model Runtime**（那里会以
`ModelResolutionError` 明确失败）。在预算这一步把整个上下文构建打断，
只会让一个诊断信息更差的错误先冒出来。因此这里的选择是：兜底窗口 +
warning + 快照里注明来源。
"""

from __future__ import annotations

from typing import Any

from app.agent.model import ModelError, ModelRole
from app.common.context import ContextBuilderType, StageId

from ..base import ContextProcessor
from ..budget import TokenBudget
from ..context import ContextBuildContext
from ..dependencies import dependencies_of
from ..policy import ContextPolicy, merge_policy_state
from ..spec import ContextProcessorSpec, StateBinding

__all__ = ["ContextResolveProcessor"]


class ContextResolveProcessor(ContextProcessor):
    """解析有效策略与 token 预算，并把它们放进 overlay。"""

    spec = ContextProcessorSpec(
        name="context_resolver",
        order=10,
        stage=StageId.CONTEXT_RESOLVER,
        when=None,  # 入口：无条件（见模块 docstring）
        inputs={
            # 覆盖项：Agent Builder / 调用方在图 state 里给的策略差异。
            # optional —— 没给就用基线策略。
            "policy_overrides": StateBinding(
                "context_policy", required=False
            ),
            "agent_type": StateBinding("run.agent_type", required=False),
            "builder_type": StateBinding("run.builder_type", required=False),
            "model_role": StateBinding("run.model_role", required=False),
        },
        description=(
            "解析 Context Policy（基线 ⊕ 覆盖）与 token 预算，"
            "作为后续处理器的输入写进 overlay。"
        ),
    )

    async def process(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
    ) -> ContextBuildContext:

        deps = dependencies_of(context)

        builder_type = _builder_type(
            inputs.get("builder_type") or context.builder_type
        )

        agent_type = str(
            inputs.get("agent_type") or context.agent_type or ""
        )

        model_role = str(
            inputs.get("model_role") or context.model_role or ModelRole.REASONING
        )

        # 解析器只做确定性映射（不查网络、不调模型）。
        baseline = deps.resolved_policy_resolver().resolve(
            agent_type=agent_type,
            builder_type=builder_type,
        )

        overrides = inputs.get("policy_overrides")

        merged = merge_policy_state(
            baseline,
            overrides if isinstance(overrides, dict) else None,
        )

        # overlay 里放**合并后的原始结构**：类型不对的覆盖项要留给 gate
        # 去报 INVALID（规格 §九），而不是在这里被悄悄修正。
        context.set("context_policy", merged)

        # 工作集上放**有效且类型正确**的版本（供处理器读数字）。
        effective = ContextPolicy.from_state(merged, base=baseline)

        context.policy = effective
        context.model_role = model_role
        context.agent_type = agent_type

        window, source, model_id = _context_window(
            deps, agent_type=agent_type, model_role=model_role, context=context
        )

        budget = TokenBudget.allocate(
            context_window=window,
            policy=effective,
            window_source=source,
            model_id=model_id,
        )

        context.budget = budget
        context.set("context_budget", budget.as_state())

        context.set(
            "context_resolution",
            {
                "source": effective.source,
                "policy_override_keys": sorted(merged)
                if isinstance(overrides, dict)
                else [],
                "agent_type": agent_type,
                "builder_type": builder_type.value,
                "model_role": model_role,
                "window_source": source,
            },
        )

        return context


# ============================================================
# internals
# ============================================================


def _context_window(
        deps: Any,
        *,
        agent_type: str,
        model_role: str,
        context: ContextBuildContext,
) -> tuple[int, str, str | None]:
    """
    这次调用会用到的模型上下文窗口（规格 §二十一）。

    返回 `(window, source, model_id)`；source 取：

        model_registry   从模型定义读到（优先）
        policy_default   回落到策略里的兜底窗口（并记 warning）
    """

    fallback = (
        context.policy.default_context_window
        if context.policy is not None
        else ContextPolicy().default_context_window
    )

    try:
        registry = deps.resolved_model_registry()

        runtime = registry.get_runtime(
            agent_type=agent_type or "agent",
            model_role=model_role,
        )

        definition = registry.definition(runtime.profile.primary)

        window = getattr(definition, "context_window", None)

        if isinstance(window, int) and window > 0:
            return window, "model_registry", definition.id

        context.warn(
            f"模型 {definition.id} 没有声明 context_window，"
            f"预算回落到策略默认值 {fallback}。"
        )

        return fallback, "policy_default", definition.id

    except ModelError as exc:

        # 模型解析失败**不在这里升级为致命错误** —— 真正需要有效模型的
        # 是 Model Runtime，那里会以明确的错误失败。这里只兜底 + 留痕。
        context.warn(
            f"无法从 ModelRegistry 解析模型窗口（{type(exc).__name__}: "
            f"{exc}），预算按策略默认值 {fallback} 估算。"
        )

        return fallback, "policy_default", None

    except Exception as exc:  # noqa: BLE001 - 兜底：预算是旁路，不该拦调用

        context.warn(
            f"解析模型窗口时出错（{type(exc).__name__}: {exc}），"
            f"预算按策略默认值 {fallback} 估算。"
        )

        return fallback, "policy_default", None


def _builder_type(value: Any) -> ContextBuilderType:

    if isinstance(value, ContextBuilderType):
        return value

    try:
        return ContextBuilderType(str(value))

    except ValueError:
        # 认不出来时不猜：用通用类型（agent_invoke 是默认语义）。
        return ContextBuilderType.AGENT_INVOKE
