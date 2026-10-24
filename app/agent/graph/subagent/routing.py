"""子代理子图的走向判断。

两条边：

    task_listener -> llm_node（领到任务） / END（没有任务）
    llm_node      -> tool_node × N（Send 扇出） / task_listener（执行完毕）

工具结果**必须回到产生 tool_call 的模型**，不能直接汇入 listener ——
否则模型看不到 ToolMessage，推理循环就断了（design_planner/task.md
§7.1 第 7 条）。

⚠️ `route_after_llm` 绝不能返回空列表
------------------------------------
LangGraph 对空 destinations 不报错，而是让这次运行**静默结束**。
本函数的形状天然保证非空：没有 tool_calls 就回 listener。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import Send

from app.agent.graph.react.routing import call_scope, resolve_tool_keys

from .catalog import LLM_NODE, TASK_LISTENER_NODE, TOOL_NODE
from .state import SubAgentState

__all__ = ["route_after_listener", "route_after_llm"]


def route_after_listener(state: SubAgentState) -> str:
    """
    listener 领到任务就去模型；没有任务就结束。

    判据是 `current_task` 非空 —— listener 在领不到任务时显式清空它，
    并把任务列表更新为终态。没有 ready task 但仍有未完成任务的情形
    也走 END（worker 不 spin；下一波由父图重新派发）。
    """

    if state.get("current_task"):
        return LLM_NODE

    return END


def route_after_llm(state: SubAgentState) -> str | list[Send]:
    """
    模型要求调工具就扇出；否则回 listener 结算任务。

    只看**最后一条** AIMessage（与主图同一条理由：遍历全部消息会让
    "上一轮调过工具"变成"这一轮也要调"，图永远在循环里）。

    `invalid_tool_calls` 不算 —— 参数不合 schema 的调用执行不了。
    """

    messages = state.get("messages") or ()

    if not messages:
        return TASK_LISTENER_NODE

    last = messages[-1]

    if not (isinstance(last, AIMessage) and last.tool_calls):
        return TASK_LISTENER_NODE

    tool_keys = resolve_tool_keys(state)
    calls = list(last.tool_calls)

    return [
        Send(
            TOOL_NODE,
            {
                "call": call,
                "tool_keys": tool_keys,
                "call_scope": call_scope(last, index, call),
                # 工具结果回到产生它的模型。
                "return_to": LLM_NODE,
                "is_controller": index == 0,
            },
        )
        for index, call in enumerate(calls)
    ]
