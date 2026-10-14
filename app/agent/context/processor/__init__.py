"""
Context Processor：上下文构建的**执行单元**（规格 §五 - §十四）。

职责分界
--------
    ContextProcessor      一个处理器该做什么（业务逻辑）
    ContextProcessorSpec  它如何自描述（name / order / stage / when / inputs）
    ContextProcessorPipeline  排序、gate、绑定、执行、观测、累积
    BaseContextBuilder    既有契约：阶段参与、固定顺序、产出 ModelRequest

一句话：**Processor 自描述依赖，ContextBuilder 统一解析 State，产物交给
ModelRegistry**。

模块地图
--------
    spec.py          StateBinding / ContextProcessorSpec（自描述）
    state.py         dot-path 解析 + MISSING 哨兵（§八 / §九）
    base.py          ContextProcessor 契约
    context.py       工作集 / 执行记录 / 快照 / 结果（§六 / §十九 / §廿六 / §廿七）
    values.py        ContextSection / SummarySnapshot
    messages.py      消息工具（唯一知道"消息长什么样"的地方）
    policy.py        ContextPolicy + resolver（§二十）
    budget.py        TokenBudget + 计数器（§二十一）
    history.py       历史选择器 + 检索器端口（§十五）
    turns.py         轮次 / 窗口 / 摘要节奏的全部数学（三个阶段共用）
    sources.py       外部上下文源端口（画像 / 记忆 / 知识 / 工具）
    summarization.py 摘要器（**唯一**调模型的地方，走 ModelRegistry）
    dependencies.py  依赖束（进程级默认 + 可注入）
    pipeline.py      执行引擎（§十）
    registry.py      反射发现
    errors.py        异常层级（复用 ApplicationError，§二十八）
    processors/      11 个具体实现

⚠️ 本包必须有 __init__.py
------------------------
注册表用 `pkgutil.walk_packages` 反射遍历 `processors/`，而它**不会
走进没有 `__init__.py` 的目录**。删掉那个文件会让所有 Processor 无声
消失（发现结果为空，而不是报错）。
"""

from .base import ContextProcessor
from .budget import HeuristicTokenCounter, TokenBudget, TokenCounter
from .context import (
    BuildPhase,
    ContextBuildContext,
    ContextBuildResult,
    ContextSnapshot,
    ProcessorExecution,
    ProcessorStatus,
)
from .dependencies import (
    ContextDependencies,
    dependencies_of,
    get_context_dependencies,
    reset_context_dependencies,
    set_context_dependencies,
)
from .errors import (
    ContextBudgetError,
    ContextBuildError,
    ContextPolicyError,
    ContextProcessorError,
    ContextStateError,
)
from .history import (
    HistorySelection,
    HistorySelector,
    TurnWindowHistorySelector,
)
from .messages import message_text
from .pipeline import PIPELINE_STAGE_ORDER, ContextProcessorPipeline
from .policy import (
    ContextPolicy,
    ContextPolicyResolver,
    DefaultContextPolicyResolver,
)
from .registry import create_pipeline, found_processors
from .spec import ContextProcessorSpec, StateBinding
from .state import MISSING, Missing, resolve_state_path
from .values import ContextSection, SummarySnapshot

__all__ = [
    # 契约
    "ContextProcessor",
    "ContextProcessorSpec",
    "StateBinding",
    # 执行
    "ContextProcessorPipeline",
    "PIPELINE_STAGE_ORDER",
    "create_pipeline",
    "found_processors",
    # 工作集与产物
    "ContextBuildContext",
    "ContextBuildResult",
    "ContextSnapshot",
    "ContextSection",
    "BuildPhase",
    "ProcessorExecution",
    "ProcessorStatus",
    "SummarySnapshot",
    # 策略与预算
    "ContextPolicy",
    "ContextPolicyResolver",
    "DefaultContextPolicyResolver",
    "TokenBudget",
    "TokenCounter",
    "HeuristicTokenCounter",
    # 历史
    "HistorySelection",
    "HistorySelector",
    "TurnWindowHistorySelector",
    # 依赖
    "ContextDependencies",
    "dependencies_of",
    "get_context_dependencies",
    "set_context_dependencies",
    "reset_context_dependencies",
    # 工具
    "MISSING",
    "Missing",
    "resolve_state_path",
    "message_text",
    # 异常
    "ContextBuildError",
    "ContextStateError",
    "ContextProcessorError",
    "ContextBudgetError",
    "ContextPolicyError",
]
