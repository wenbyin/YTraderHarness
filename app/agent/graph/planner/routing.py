"""Planner 子图的走向判断。

    planner_llm -> planner_tool × N（Send 扇出） / task_generator

工具结果必须回到产生 tool_call 的模型；没有 tool_calls 才进入
task_generator。与子代理路由同一条形状，只把目标换成 planner 自己的
节点，并复用主图的 fanout trace 与 call_scope。

⚠️ 绝不能返回空列表
------------------
LangGraph 对空 destinations 不报错，而是让整次运行静默结束。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from langgraph.types import Send

from app.agent.graph.react.context import ReactContext
from app.agent.graph.react.routing import (
    _emit_fanout_trace,
    call_scope,
    resolve_tool_keys,
)

from .catalog import PLANNER_LLM_NODE, TASK_GENERATOR_NODE, TOOL_NODE
from .state import PlannerState

__all__ = ["route_after_planner_llm"]


def route_after_planner_llm(
        state: PlannerState,
        runtime: Runtime[ReactContext],
) -> str | list[Send]:
    """
    模型要求调工具就扇出；否则交给 task_generator。

    只看**最后一条** AIMessage：历史里的旧 tool_calls 已经处理过，
    遍历全部会让 planner 永远在循环里。
    """

    messages = state.get("messages") or ()

    if not messages:
        return TASK_GENERATOR_NODE

    last = messages[-1]

    if not (isinstance(last, AIMessage) and last.tool_calls):
        return TASK_GENERATOR_NODE

    calls = list(last.tool_calls)

    # 与主图 route_after_agent 同一条 trace：确定性 event id 在
    # _emit_fanout_trace 里生成，resume/replay 不会产生重复记录。
    _emit_fanout_trace(runtime, last, calls)

    tool_keys = resolve_tool_keys(state)

    return [
        Send(
            TOOL_NODE,
            {
                "call": call,
                "tool_keys": tool_keys,
                "call_scope": call_scope(last, index, call),
                "return_to": PLANNER_LLM_NODE,
                "is_controller": index == 0,
            },
        )
        for index, call in enumerate(calls)
    ]
