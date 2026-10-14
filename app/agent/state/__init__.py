"""
Agent 的**公共**状态词汇。

放这里的都是"不止一张图、不止一个节点要用"的东西：

    agent_state.py   输入分类的结果与它的取值域

图自己的 state（比如 ReAct 那张图）仍然定义在各自的图包里
（app/agent/graph/react/state.py），它们从这里**引用**公共类型，而不是
各自再定义一份"行为分类"的枚举 —— 两份词汇表迟早会漂移成两套取值，
而下游（路由、模型选择）会按其中一套写死。
"""

from .agent_state import (
    MODULE_OTHER,
    AgentClassification,
    BehaviorCategory,
    ExecutionPlan,
    coerce_behavior,
    coerce_execution_plan,
    empty_classification,
)

__all__ = [
    "MODULE_OTHER",
    "AgentClassification",
    "BehaviorCategory",
    "ExecutionPlan",
    "coerce_behavior",
    "coerce_execution_plan",
    "empty_classification",
]
