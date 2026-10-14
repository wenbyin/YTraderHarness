"""
"这次工具调用是谁发起的" —— 一个进程内的、随调用生效的载体。

为什么需要它（以及为什么不是"再往工具参数里塞两个字段"）
--------------------------------------------------
工具的入参 schema 是**暴露给模型**的，而模型不能知道"我是谁" ——
`user_id` 一旦进 schema，模型就能填一个别人的 id，那是一条跨租户读。

所以身份只能**在旁边**递给服务方法，而不是走 parameters。两条路：

    A  拦截器把它写进 `ToolInvocation.arguments`
       问题在于 `MemorySearchRequest` 是 `extra="forbid"` 的 ——
       多一个键就被拒。要让它成立，得再给模型加一层"从 LLM schema 里
       藏掉注入字段"的机制（`synthesize_schema` 目前只会把签名原样
       翻译成 schema）。那是为一个集成点造一台机器。
    B  一个随调用生效的 ContextVar（本模块）
       服务方法自己读。没有 schema 要藏，没有模型要改。

**这里选 B**，理由是 A 在本仓库的请求模型约定下要多做一件事，而 B 不用。
将来若真要换成 A（身份显式出现在方法签名上），把那层"藏字段"补上、
再把下面这个读取口换成方法参数即可 —— 本模块是唯一要改的地方。

⚠️ 为什么是 ContextVar 而不是全局变量
---------------------------------
工具执行是**并发**的（tools 节点按调用扇出，一个 run 里多个调用各自
一个 task）。全局变量会让 A 的调用读到 B 的身份 —— 而那是越权。
ContextVar 按 task 隔离，且 await 之间保持，正好是这里要的语义。

⚠️ 拿不到时是 `None`，而且**必须 fail-closed**
------------------------------------------
写入点只有一处（react 图的 tools 节点，执行工具之前）。拿不到只有两种
可能：调用不来自 agent 那条链（脚本、冒烟），或者那条链没接上。
两种情况下调用者都是"不知道是谁" —— 服务必须拒绝，而不是退化成
"不过滤"（那等于把全库摊开）。见 memory 服务里那个包装方法的判断。

⚠️ 为什么放在 app/common
--------------------
写入方在 agent 侧（`app/agent/graph/react/tools.py`），读取方在服务侧
（`app/services/memory/...`）。唯一能同时被两边 import 而不制造
"服务依赖 agent" 这种反向依赖的地方，就是中立层。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import UUID

__all__ = [
    "ToolCaller",
    "current_tool_caller",
    "tool_caller_scope",
]


@dataclass(frozen=True, slots=True)
class ToolCaller:
    """
    一次工具调用的调用者身份快照。

    三个字段都可为 None，含义不同：

        user_id      调用者 —— 决定"能搜到谁的数据"。None = 不知道是谁
        session_id   这次调用发生在**哪个会话**里（ReactContext 从 run
                     交进来）。None = 这个调用不挂在会话上
        agent_id     这次调用**在哪份职责下**发生（= 运行的 `duty_key`）

    ⚠️ 为什么 `agent_id` 也在这里
    --------------------------
    L2/L3（场景 / 画像）的 scope 是 **(agent_id, user|org)** —— 同一个用户
    在不同职责下是两份记忆（见 `SceneScope`）。而往上下文里注入那两段时
    用的就是 `duty_key`（`context_policy.memory_retrieve.agent_id`，
    由 agent_builder_node 从 `state["duty_key"]` 写入）。

    工具这边取不到它，就只能从会话行猜一个 agent_id —— 而两者一旦不同
    （一个会话跑多份职责时就会），模型会**看到索引里的场景名却取不到那一篇**：
    它照着注入的清单调工具，拿回"场景不存在"。不报错、只是答不上来。

    冻起来（frozen）：它是一次调用的环境，处理器与服务都只读。
    """

    user_id: str | None
    session_id: UUID | None
    agent_id: str | None = None


#: 当前调用者。默认 None = "不在任何工具调用里"。
_current: ContextVar[ToolCaller | None] = ContextVar(
    "ytrader_tool_caller", default=None
)


def current_tool_caller() -> ToolCaller | None:
    """
    取当前这次工具调用的调用者；不在工具调用里时返回 None。

    **不要**把 None 当成"不过滤" —— 那是拿"我不知道是谁"当"谁都可以"。
    调用方要么拒绝，要么只读写明确无归属的数据。
    """

    return _current.get()


@contextmanager
def tool_caller_scope(
        caller: ToolCaller,
) -> Iterator[ToolCaller]:
    """
    在 `with` 范围内把调用者设为 `caller`，退出时**还原**（不是清空）。

    还原而不是清空，是因为工具执行可能嵌套（工具内部再派发一次调用，
    将来出现 A2A 那种形态时）。`ContextVar.set` 返回的 token 正是为
    这件事准备的；用 `set(None)` 复位会在嵌套时把外层的身份抹掉。
    """

    token = _current.set(caller)

    try:
        yield caller

    finally:
        _current.reset(token)
