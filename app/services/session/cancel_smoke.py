"""
取消一次正在跑的运行：冒烟。

跑法：
    python -m app.services.session.cancel_smoke

它验的是"取消"这件事从 HTTP 那一端到 checkpoint 那一端的整条路：

    dispatcher.cancel(run_id)
      -> 取消**那一次运行的子任务**（不是 worker）
      -> handler 捕获 CancelledError，走取消收尾
      -> 状态记 cancelled + 放掉 thread 锁
      -> agent_message 里留下痕迹（有产出写产出，一条都没产出就补公告）

⚠️ 为什么必须真的经过调度器
------------------------
`agent_run_smoke` 是直接调 `handler(envelope)` 的，那条路上**没有 task
可取消** —— 取消的机制（子任务、登记表、CancelledError 的归属判断）
一个都验不到。本文件因此把真正的 `InProcessGraphDispatcher` 拉进来，
并且把图拖慢到"确实还在跑"，取消才有对象。

⚠️ 拖慢图的手法
--------------
在 `ToolExecutor.execute` 上挂一个 `asyncio.sleep`。它是**真的**取消点
（await 点上抛 CancelledError），而且落在图的深处 —— 比在桩节点里 sleep
更接近真实（真模型调用、真工具调用都在那类 await 上）。

⚠️ 锁由本文件自己模拟
------------------
`_seed_run` 绕开了 `create_run`（见 agent_run_smoke 的说明），所以 thread
锁不会有谁去取。取消要验的恰恰是"锁有没有被放掉"，因此这里**手动取一次**
（模拟 create_run 那一步），再断言取消之后它是空的。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from typing import Any
from uuid import uuid4

from langgraph.store.memory import InMemoryStore

import app.services.session.agent_run_smoke as _smoke
from app.infra.agent import tool_executor as _tool_executor_module
from app.infra.agent.tool_executor import ToolExecutor
from app.infra.graph import (
    DispatchEnvelope,
    get_thread_run_lock,
    reset_thread_run_lock,
)
from app.infra.graph.cancellation import (
    get_run_cancellation_registry,
    reset_run_cancellation_registry,
)
from app.infra.graph.dispatcher_impl import InProcessGraphDispatcher
from app.services.session.agent_run_smoke import (
    _Harness,
    _seed_run,
    check,
    section,
)
from app.services.session.domain.schemas import (
    MessageKind,
    MessageStatus,
    RunStatus,
)

#: 默认只报错；排查时 `SMOKE_LOG_LEVEL=INFO`（或 DEBUG）能把
#: `_record_produced_messages` 那条"吞异常只记日志"的路径照出来 ——
#: 否则"消息一条没写"在冒烟里只有一个空列表，看不出为什么。
logging.basicConfig(
    level=os.environ.get("SMOKE_LOG_LEVEL", "ERROR")
)

#: 图要"跑"多久才够我们插进去取消。够长就行，反正取消会把它砍断。
SLOW_SECONDS = 30.0

#: 等状态变化的上限。超了就当失败 —— 免得冒烟挂死而不是报错。
WAIT_SECONDS = 10.0


#: 真正跑到 `execute` 的工具调用，按顺序记下来。
#:
#: ⚠️ 这是本文件里最要紧的一个断言工具。取消**不一定**能让这次调用消失：
#! 它可能只是"这次没跑成"，而那个 tool_call 还留在 checkpoint 上，
#: 下一轮把那条 thread 接着跑时**顺手执行掉**。所以必须能看见"谁真的跑了"。
EXECUTED: list[tuple[str, dict]] = []


class _SlowToolExecutor(ToolExecutor):
    """
    在真正执行之前先 `await asyncio.sleep(...)`。

    这个 await 就是取消点：`task.cancel()` 会让它抛 `CancelledError`，
    一路穿过图与 handler —— 与"取消一次真模型调用"是同一条路径。
    """

    async def execute(
            self,
            tool_key: str,
            arguments: Any = None,
            **kwargs: Any,
    ) -> Any:

        await asyncio.sleep(SLOW_SECONDS)

        EXECUTED.append((tool_key, dict(arguments or {})))

        return await super().execute(tool_key, arguments, **kwargs)


class _NeverQuery:
    """工具目录查询：永远不返回（图会一直卡在这一步，正好当取消点）。"""

    async def tool_ids_for_keys(self, keys: Any) -> Any:
        await asyncio.sleep(SLOW_SECONDS)
        return {}


class _NeverFactory:

    @contextlib.asynccontextmanager
    async def scope(self, contract: Any, **kwargs: Any):
        yield _NeverQuery()


async def _wait_for(
        harness: _Harness,
        run_id: Any,
        status: str,
) -> bool:
    """轮询等运行走到某个状态。超时返回 False（让用例报失败，不挂死）。"""

    deadline = asyncio.get_running_loop().time() + WAIT_SECONDS

    while asyncio.get_running_loop().time() < deadline:

        if harness.rows[run_id].status == status:
            return True

        await asyncio.sleep(0.01)

    return False


def _thread_id(harness: _Harness) -> str:
    return f"agentsess:{harness.session_id}"


def _envelope(harness: _Harness, run_id: Any) -> DispatchEnvelope:
    return DispatchEnvelope(
        graph_name=harness.definition.name,
        pipeline_run_id=run_id,
        thread_id=_thread_id(harness),
        payload={"session_id": str(harness.session_id)},
    )


async def main() -> int:

    # 每个用例都从干净的进程级状态开始（锁、取消登记表都是单例）。
    reset_thread_run_lock()
    reset_run_cancellation_registry()

    # --------------------------------------------------------
    section("拼装：真的调度器 + 一个跑得很慢的图")

    _tool_executor_module._executor = _SlowToolExecutor(
        chain_loader=lambda tool_id: _no_chain()
    )

    import app.agent.graph.react.tools as _tools_module

    _tools_module.get_service_factory = lambda: _NeverFactory()

    _store = InMemoryStore()
    _original_register = _smoke.register_react_graph

    def _register_with_store(**kwargs: Any) -> Any:
        kwargs.setdefault("store", _store)
        return _original_register(**kwargs)

    _smoke.register_react_graph = _register_with_store

    harness = _Harness(
        agent_tool_calls=True,
        tool_keys=["session_smoke.echo"],
        tool_call_name="session_smoke_echo",
    )

    dispatcher = InProcessGraphDispatcher()
    dispatcher.register(harness.definition.name, harness.handler)
    await dispatcher.start()

    # --------------------------------------------------------
    section("取消一次正在跑的运行")

    run = _seed_run(harness.rows, "跑起来然后取消掉", session_id=harness.session_id)

    thread_id = _thread_id(harness)

    # 模拟 create_run 那一步取的锁。
    # ⚠️ `try_acquire` 的约定与直觉相反：**成功返回 None**，失败才返回
    # 当前持有者（见 thread_lock.py —— 调用方要的是"谁在占着"，那才是能给
    # 用户看的 409 信息）。
    holder = get_thread_run_lock().try_acquire(thread_id, run_id=run.run_id)

    check("锁先被这次运行拿住了（模拟 create_run）", holder is None)

    handle = await dispatcher.dispatch(_envelope(harness, run.run_id))

    check("派发被受理", handle.accepted)

    check(
        "图确实跑起来了（置成 running）",
        await _wait_for(harness, run.run_id, RunStatus.RUNNING.value),
    )

    registry = get_run_cancellation_registry()

    check(
        "取消之前登记表是空的",
        not registry.is_requested(run.run_id),
    )

    accepted = await dispatcher.cancel(run.run_id)

    check("取消被受理（调度器认得这个 run）", accepted)

    check(
        "登记表留下了痕迹（handler 靠它区分「我被取消」与「应用关闭」）",
        registry.is_requested(run.run_id),
    )

    check(
        "运行最终记成 cancelled（不是 failed、也不是 completed）",
        await _wait_for(harness, run.run_id, RunStatus.CANCELLED.value),
    )

    # --------------------------------------------------------
    # 落库：agent_run 与 agent_message 两张表都要写
    # --------------------------------------------------------

    cancelled_row = harness.rows[run.run_id]

    check(
        "agent_run 写回 status=cancelled",
        cancelled_row.status == RunStatus.CANCELLED.value,
    )

    check(
        "agent_run 的 error_* 是空的（用户主动取消不是「失败」）",
        cancelled_row.error_type is None
        and cancelled_row.error_message is None,
    )

    # `_seed_run` 绕过了 create_run，所以这个 run 在 agent_message 里的行
    # **只有本轮图产出的那些**（没有接收时写下的用户消息）—— 断言因此干净。
    written = [
        row for row in harness.messages.rows if row.run_id == run.run_id
    ]

    check(
        "agent_message 写进了本轮产出的消息（不是一条都不写）",
        len(written) >= 1,
    )

    check(
        "那些消息的 status 是 cancelled",
        all(
            row.status == MessageStatus.CANCELLED.value
            for row in written
        ),
    )

    print(
        f"    （agent_message 写了 {len(written)} 条："
        f"{[(row.role, row.status) for row in written]}）"
    )

    # ⚠️ 取消**当下**那条 thread 上还剩什么 —— 这是"被取消的那次工具调用会
    # 不会被下一轮顺手执行掉"的直接判据（必须在这一刻读：下一轮跑完之后
    # 什么都排空了，读不出所以然）。
    after_cancel = await harness.definition.compiled.aget_state(
        {"configurable": {"thread_id": thread_id}}
    )

    print(f"    （取消当下：next={after_cancel.next}）")

    # ⚠️ 更要紧的是 messages 里留下了什么形状：`tools.py` 的 docstring 警告过
    # "每个 tool_call 都必须有一条对应的 ToolMessage，少一条会让下一轮 agent
    # 看到悬空的 tool_call，部分 provider 直接拒绝这次请求"。
    # 取消正好可能留下那个形状，所以直接数一数。
    _msgs = after_cancel.values.get("messages") or []
    _calls = sum(len(getattr(m, "tool_calls", None) or ()) for m in _msgs)
    _results = sum(1 for m in _msgs if type(m).__name__ == "ToolMessage")

    print(
        f"    （取消当下：消息 {len(_msgs)} 条，"
        f"tool_calls {_calls} 个，ToolMessage {_results} 条）"
    )

    # ⚠️ 这里**故意不是 check**
    # ----------------------
    # 不强行中断 + 不回滚时，取消会留下 `next=('tools',)` 与一条**悬空的
    # tool_call**（有调用、没有对应的 ToolMessage）。这是已知的、量出来的
    # 形状，不是本冒烟要判定的失败 —— 判定它要么改实现（写一条"已取消"的
    # ToolMessage 把调用了结），要么接受它（下一轮部分 provider 会拒）。
    # 在决定之前把它打印出来，别让它静默溜过去。
    if _calls != _results:
        print(
            "    ⚠️ 取消留下了悬空的 tool_call（有调用、无 ToolMessage）。"
            "tools.py 的 docstring 警告过这个形状：部分 provider 会直接"
            "拒绝下一次请求。"
        )

    # 同样注意约定：拿到 None 才是"取成功了"，也就是锁确实空着。
    #
    # ⚠️ 这次试取是**真的取上了**（没有 peek API），所以拿一个已知的令牌
    # 取、验完立刻还回去 —— 不还的话下面那个用例会被自己挡成 409。
    probe = uuid4()

    check(
        "thread 锁被放掉了（取消意味着这次运行真的结束了）",
        get_thread_run_lock().try_acquire(thread_id, run_id=probe) is None,
    )

    get_thread_run_lock().release(thread_id, run_id=probe)

    check(
        "登记表被清掉了（不清理就是内存泄漏）",
        not registry.is_requested(run.run_id),
    )

    check(
        "worker 还活着 —— 取消子任务没有把调度器一起干掉",
        dispatcher.queue_size == 0,
    )

    # --------------------------------------------------------
    section("取消之后那条 thread 仍然可用")

    second = _seed_run(harness.rows, "取消之后的下一轮", session_id=harness.session_id)

    get_thread_run_lock().try_acquire(thread_id, run_id=second.run_id)

    # 下一轮不再拖慢：把执行器换成能记录执行的那个，让它能跑完。
    EXECUTED.clear()

    class _RecordingExecutor(ToolExecutor):
        async def execute(
                self,
                tool_key: str,
                arguments: Any = None,
                **kwargs: Any,
        ) -> Any:

            EXECUTED.append((tool_key, dict(arguments or {})))

            return await super().execute(tool_key, arguments, **kwargs)

    _tool_executor_module._executor = _RecordingExecutor(
        chain_loader=lambda tool_id: _no_chain()
    )

    handle = await dispatcher.dispatch(_envelope(harness, second.run_id))

    check("取消之后仍能派发新运行", handle.accepted)

    check(
        "新运行跑到了 completed（thread 没有被半截状态卡住）",
        await _wait_for(harness, second.run_id, RunStatus.COMPLETED.value),
    )

    print(f"    （这一轮真的执行了的工具调用：{EXECUTED}）")

    # 下一轮跑完之后应该是空的 —— 这条验的是"thread 最终是可用的"，
    # 与上面那条"取消当下有没有留下 pending"是两回事。
    after_second = await harness.definition.compiled.aget_state(
        {"configurable": {"thread_id": thread_id}}
    )

    check(
        "两个运行都结束之后 checkpoint 上没有遗留 pending 任务",
        not after_second.next,
    )

    # --------------------------------------------------------
    section("排队期间就被取消 -> 也要留下一条 message")

    queued = _seed_run(
        harness.rows, "还没轮到就被取消", session_id=harness.session_id
    )

    # ⚠️ 这里**直接调 handler**，而不是走调度器
    # --------------------------------------
    # 要构造的是"**还没轮到它**就被取消"，也就是 handler 开工**之前**的
    # 那个时刻。走 dispatcher 得先让 worker 忙起来、把它挤在队列里 ——
    # 那要依赖时序；而这里判定的分支（handler 开头那句 `is_requested`
    # 短路）本来就与调度器无关。
    #
    # 上面两个用例仍然走真调度器 —— 取消的**机制**（子任务、登记表、
    # CancelledError 的归属）在那里验，这里只验"排队取消也要落库"。
    get_run_cancellation_registry().request(queued.run_id)

    await harness.handler(_envelope(harness, queued.run_id))

    queued_row = harness.rows[queued.run_id]

    check(
        "运行记成 cancelled（图一次都没跑）",
        queued_row.status == RunStatus.CANCELLED.value,
    )

    check(
        "error_* 是空的（排队取消同样是用户主动取消，不是失败）",
        queued_row.error_type is None
        and queued_row.error_message is None,
    )

    queued_msgs = [
        row for row in harness.messages.rows if row.run_id == queued.run_id
    ]

    # 这一条是本次改动的核心：以前排队取消**一条 message 都不写**，
    # 会话历史里只剩用户那句话，而运行却是 cancelled —— 用户分不清
    # "模型没回"和"我把它停了"。
    check(
        "补了一条 message（不是一条都不写）",
        len(queued_msgs) == 1,
    )

    check(
        "那条是 kind=notice + status=cancelled（不伪装成模型回复）",
        bool(queued_msgs)
        and queued_msgs[0].kind == MessageKind.NOTICE.value
        and queued_msgs[0].status == MessageStatus.CANCELLED.value,
    )

    check(
        "正文是中断公告（不是空内容 —— 空内容在界面上就是什么都没有）",
        bool(queued_msgs)
        and any(
            block.get("type") == "text" and block.get("text")
            for block in (queued_msgs[0].content or [])
        ),
    )

    # --------------------------------------------------------
    section("取消一个不认得的 run -> 不受理")

    fresh = InProcessGraphDispatcher()

    check(
        "没派发过的 run_id 取消不动它",
        not await fresh.cancel(uuid4()),
    )

    await dispatcher.aclose()

    print("\n全部通过。")
    return 0


async def _no_chain() -> list[Any]:
    """没有处理器链 —— 取消冒烟不关心审批，只关心能不能被打断。"""

    return []


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
