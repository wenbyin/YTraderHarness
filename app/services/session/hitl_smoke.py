"""
人工审批（HITL）端到端冒烟：图停得下来，也恢复得起来。

跑法：
    python -m app.services.session.hitl_smoke

它验的是**两端接在一起**的那条路 —— 单独看任何一端都已经有别的冒烟：

    approval.py 的三选项语义      app/agent/tool_processor_smoke.py
    工具节点与允许集              app/services/session/agent_run_smoke.py

本文件补的是它们之间的接缝，也正是这一路上最容易被"结构就位"骗过去的地方：

    建运行 -> 图跑到审批 -> **interrupt** -> waiting_input + hitl.requested
           -> resume -> **Command(resume=...)** -> hitl.resolved -> completed

⚠️ 为什么必须端到端，不能只测处理器
--------------------------------
处理器的三选项可以直接调 `before()` 验（那是 tool_processor_smoke 干的事），
但那**证明不了**它真的能停下来 —— 从处理器到 `interrupt()` 之间隔着三层
接线，任何一层断了都会静默退化：

    tools 节点没造 ToolRuntime    -> `_identifiers` 返回 None
                                    -> 走 fail-closed 的"通道不可用"分支
                                    -> 工具被拒绝，但**运行照常 completed**

最后那个后果是本文件存在的主要理由：一条**看起来成功**的运行，实际上工具
一次都没跑，而且没有任何地方报错 —— 只有 `waiting_input` 与 `hitl.requested`
同时出现，才说明整条链路是通的。

⚠️ 桩的边界
----------
复用 `agent_run_smoke` 的内存仓储与桩节点（它有 `__main__` 守卫，导入安全）。
只有 tools 节点被换成**真货**：审批拦截器、处理器链、ToolRuntime 全是真的。

拼装期替换三处，都是"冒烟没有数据库"的必然结果，不是绕过被测代码：

    处理器链的读法    换成返回一条 approval_interceptor 挂载记录
    工具目录查询      换成返回一个固定的 tool_id（否则走"目录不可用"分支）
    图的 store        换成 InMemoryStore（否则批准账本退回进程内并打 warning）
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from typing import Any
from uuid import UUID, uuid4

from langgraph.store.memory import InMemoryStore

import app.services.session.agent_run_smoke as _smoke
from app.agent.tools.processor.interceptor.approval import ApprovalConfig
from app.common.exceptions import ConflictError
from app.infra.agent import tool_executor as _tool_executor_module
from app.infra.agent.tool_executor import ToolExecutor
from app.infra.graph import DispatchEnvelope
from app.services.session.agent_run_smoke import (
    _FakeRepositoryFactory,
    _Harness,
    _InMemoryMessageRepository,
    _InMemoryRunRepository,
    check,
    section,
)
from app.services.session.application.agent_run_service_impl import (
    AgentRunServiceImpl,
)
from app.services.session.domain.schemas import (
    ResumeAnswer,
    ResumeRequest,
    RunStatus,
)

#: 处理器链要用的挂载参数。`always_allowed=True` 才提供「始终允许」按钮。
APPROVAL_CONFIG: dict[str, Any] = {
    "always_allowed": True,
    "prompt": "允许这次付款吗？",
}

logger = logging.getLogger(__name__)


# ============================================================
# 拼装期的替身
# ============================================================


class _ApprovalRow:
    """
    一条挂载记录：BEFORE / SYNC 的 approval_interceptor。

    `build_chain` 按鸭子类型读这几个属性（见 infra/agent/chain.py），
    因此不需要真的构造 `ToolProcessor` 实体 —— 那会把一次数据库往返
    拖进冒烟。
    """

    processor_key = "approval_interceptor"
    hook = "BEFORE"
    execution_mode = "SYNC"
    config = APPROVAL_CONFIG
    sequence = 1
    enabled = True


class _FakeToolQuery:
    """只实现 tools 节点要的那一个方法。"""

    async def tool_ids_for_keys(
            self,
            tool_keys: list[str],
    ) -> dict[str, Any]:

        return {key: uuid4() for key in tool_keys}


class _FakeToolFactory:
    """
    `get_service_factory()` 的替身。

    ⚠️ 它挡掉的是**"工具目录不可用"那条降级路径**，不是被测代码。
    真实现要连库，而这条路径一旦走到，`execute` 根本不会被调用、审批
    也就永远不会触发 —— 冒烟会绿得毫无意义（这正是 agent_run_smoke 里
    带 tool_calls 那个用例的现状：它断言的就是"降级为目录不可用"）。
    """

    @contextlib.asynccontextmanager
    async def scope(self, contract: Any, **kwargs: Any):
        yield _FakeToolQuery()


class _SessionStub:
    """
    `AgentSessionRepository` 的最小替身 —— `resume_run` 只用到一个 `get`。

    `user_id` 不参与判定：归属校验是**仓储**的事（真实现里过滤在 SQL 上），
    而这一节验的是 `answers` 的集合校验，不是归属。
    """

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id

    async def get(
            self,
            session_id: Any,
            *,
            user_id: str | None = None,
    ) -> "_SessionStub":

        return self


class _Recorder:
    """接住 `publish_run_event`，好让用例断言事件而不碰 Redis。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    async def __call__(
            self,
            event_type: Any,
            topic_id: str,
            *,
            data: Any = None,
            node: str | None = None,
    ) -> bool:

        self.events.append(
            (
                getattr(event_type, "value", str(event_type)),
                str(topic_id),
                dict(data or {}),
            )
        )

        return True

    def types(self) -> list[str]:
        return [name for name, _, _ in self.events]

    def of(self, event_type: str) -> list[dict[str, Any]]:
        return [data for name, _, data in self.events if name == event_type]

    def clear(self) -> None:
        self.events.clear()


# ============================================================
# 一条装配好的链路
# ============================================================


class _HitlHarness(_Harness):
    """
    `_Harness` + 三处替身（见模块 docstring）。

    图与 handler 都是真的（继承来的），因此 `_dispatch` / `run` 的行为与
    生产一致 —— 包括 handler 里那些锁与终态的决策。
    """

    def __init__(self, *, tool_call_name: str = "session_smoke_echo") -> None:

        self.events = _Recorder()

        super().__init__(
            agent_tool_calls=True,
            tool_keys=["session_smoke.echo"],
            tool_call_name=tool_call_name,
        )

    # ---- 恢复 --------------------------------------------------

    async def resume(
            self,
            run_id: UUID,
            *,
            answers: dict[str, dict[str, Any]],
    ) -> None:
        """
        走一次**恢复**派发 —— 与 `AgentRunServiceImpl._schedule_resume`
        写进 envelope 的形状逐字一致（那边是唯一的写入方）：`resume` 是
        一个 `{interrupt_id: data}` 映射，正好是 `Command(resume=...)` 要的。

        ⚠️ 这里**绕过**了服务层的"必须一次带全"校验（那在
        `_require_complete_answers` 里，走 HTTP 才有）。所以本文件里
        "少给一个 id"那个用例验的是**引擎**的行为，不是服务端会放过它。
        """

        await self.handler(
            DispatchEnvelope(
                graph_name=self.definition.name,
                pipeline_run_id=run_id,
                thread_id=f"agentsess:{self.session_id}",
                payload={
                    "session_id": str(self.session_id),
                    "resume": {
                        interrupt_id: dict(data)
                        for interrupt_id, data in answers.items()
                    },
                },
            )
        )

    def run_row(self, run_id: UUID) -> Any:
        return self.rows[run_id]

    def tool_messages(self) -> list[str]:
        """工具那一轮回到对话里的文本（ToolMessage 的正文）。"""

        out: list[str] = []

        for row in self.messages.rows:
            for block in getattr(row, "content", None) or ():
                text = block.get("text", "") if isinstance(block, dict) else ""

                if text.startswith(("调用被拒绝", "工具目录")):
                    out.append(text)

        return out


# ============================================================
# 用例
# ============================================================


async def _interrupt(
        harness: _HitlHarness,
) -> tuple[Any, str]:
    """跑一轮，断言它停在审批上，返回 `(run, interrupt_id)`。"""

    harness.events.clear()

    run = await harness.run()

    check(
        "终态是 waiting_input（不是 completed）",
        run.status == RunStatus.WAITING_INPUT.value,
    )

    # ⚠️ 这一条是本文件的核心断言：只要 approval 走了 fail-closed 的
    # "通道不可用"分支，运行就会以 completed 收尾，而上面那条会先炸。
    # 两条一起看，才排除"跑完了但其实没停过"。
    check(
        "没有发终止事件（运行没结束，SSE 流不该关）",
        not [
            name for name in harness.events.types()
            if name in ("run.completed", "run.failed")
        ],
    )

    requested = harness.events.of("hitl.requested")

    check("发了一条 hitl.requested", len(requested) == 1)

    payload = requested[0]

    check("带 interrupt_id（回答时要用它）", bool(payload.get("interrupt_id")))

    check(
        "带三个按钮（配置里 always_allowed=True）",
        payload.get("options") == ["approve", "reject", "always"],
    )

    check(
        "带被审批的工具与参数（不能盲批）",
        payload.get("tool_key") == "session_smoke.echo"
        and "arguments" in payload,
    )

    # ⚠️ 不是 thread_id：审批记录按**运行**归档。agent 那条路的 thread 是
    # 会话级稳定的，拿它当作用域会让「始终允许」泄漏到同会话的后续运行。
    check(
        "作用域是 run_id（不是会话级的 thread_id）",
        payload.get("run_id") == str(run.run_id),
    )

    # 客户端要靠它画倒计时。⚠️ 断言"是个正数"而不是"等于 300"：这里要钉的是
    # **契约**（字段在、形状对），默认值是可以改的配置，拿它当断言会在改默认值时
    # 变成一条与本次改动无关的红。
    check(
        "带等待时长（审批界面用它画倒计时，不是判定）",
        isinstance(payload.get("wait_seconds"), int)
        and payload.get("wait_seconds") > 0,
    )

    return run, payload["interrupt_id"]


async def main() -> int:

    # --------------------------------------------------------
    section("拼装：把审批链的读法换成固定的一条挂载记录")

    # 处理器链：读库换成读固定的一条。
    _tool_executor_module._executor = ToolExecutor(
        chain_loader=lambda tool_id: _load_approval_chain()
    )

    # 工具目录查询：连库换成固定返回。
    import app.agent.graph.react.tools as _tools_module

    _tools_module.get_service_factory = lambda: _FakeToolFactory()

    # 图编译时给一个 store（审批账本要落它）。
    _store = InMemoryStore()
    _original_register = _smoke.register_react_graph

    def _register_with_store(**kwargs: Any) -> Any:
        kwargs.setdefault("store", _store)
        return _original_register(**kwargs)

    _smoke.register_react_graph = _register_with_store

    # 事件：接住，不碰 Redis。
    import app.services.session.application.agent_run_dispatch as _dispatch_module

    recorder = _Recorder()
    _dispatch_module.publish_run_event = recorder

    # --------------------------------------------------------
    section("图停在审批上：waiting_input + hitl.requested")

    harness = _HitlHarness()
    harness.events = recorder

    run, interrupt_id = await _interrupt(harness)

    check("运行行落库的状态也是 waiting_input", run.status == "waiting_input")

    # --------------------------------------------------------
    section("恢复：同意 -> hitl.resolved + 跑完")

    recorder.clear()

    await harness.resume(
        run.run_id,
        answers={interrupt_id: {"decision": "approve"}},
    )

    check(
        "发了 hitl.resolved",
        "hitl.resolved" in recorder.types(),
    )

    check(
        "事件里的 interrupt_id 与请求一致",
        recorder.of("hitl.resolved")[0].get("interrupt_id") == interrupt_id,
    )

    check(
        "跑完了（resume 确实把图驱动到了 END）",
        harness.run_row(run.run_id).status == RunStatus.COMPLETED.value,
    )

    check(
        "工具没有被拒绝（approve 走的是放行那条）",
        not harness.tool_messages(),
    )

    # --------------------------------------------------------
    section("恢复：拒绝 -> 工具不执行，但整轮照常完成")

    deny_harness = _HitlHarness()
    deny_harness.events = recorder

    recorder.clear()

    deny_run, deny_interrupt = await _interrupt(deny_harness)

    recorder.clear()

    await deny_harness.resume(
        deny_run.run_id,
        answers={deny_interrupt: {"decision": "reject"}},
    )

    check(
        "拒绝之后运行仍然走到 completed（拒绝不是失败）",
        deny_harness.run_row(deny_run.run_id).status
        == RunStatus.COMPLETED.value,
    )

    check(
        "工具确实没跑：回到对话里的是一条「调用被拒绝」",
        any(
            text.startswith("调用被拒绝")
            for text in deny_harness.tool_messages()
        ),
    )

    # --------------------------------------------------------
    section("恢复：interrupt_id 给错 -> 空操作（重新停住，绝不误批）")

    # ⚠️ 实测：id 匹配不上时 langgraph **不认这个回答**，pending 任务原样
    # 留着，图重新停在同一个中断上 —— 处理器根本没有被叫醒（`parse` 都没
    # 调到）。所以这条断言的是"什么都没发生"，不是"报错"、也不是"拒绝"。
    #
    # 为什么这条比"报错"重要：这条路上唯一不可接受的失败是**放行**。
    # 停住是最安全的那一边，代价只是客户端得自己发现回答没生效。
    wrong_harness = _HitlHarness()
    wrong_harness.events = recorder

    recorder.clear()

    wrong_run, _ = await _interrupt(wrong_harness)

    recorder.clear()

    await wrong_harness.resume(
        wrong_run.run_id,
        answers={
            "00000000000000000000000000000000": {"decision": "approve"}
        },
    )

    check(
        "给错 id 之后运行**仍然**停在 waiting_input",
        wrong_harness.run_row(wrong_run.run_id).status
        == RunStatus.WAITING_INPUT.value,
    )

    check(
        "工具没有被放行（没有走到 completed）",
        wrong_harness.run_row(wrong_run.run_id).status
        != RunStatus.COMPLETED.value,
    )

    # --------------------------------------------------------
    section("服务层校验：answers 必须凑齐当前所有 pending")

    # 这一节**不走 HTTP**，直接构造服务调 `resume_run` —— 校验发生在请求期，
    # 与是否经过 FastAPI 无关（不这么做的话客户端会拿到 200 却什么都没推进，
    # 而那正是这条校验存在的理由）。
    guarded = _HitlHarness()
    guarded.events = recorder
    recorder.clear()

    guarded_run, guarded_interrupt = await _interrupt(guarded)

    service = AgentRunServiceImpl(
        agent_run_repository=guarded.runs,
        agent_session_repository=_SessionStub(
            f"agentsess:{guarded.session_id}"
        ),
        message_repository=guarded.messages,
        # 校验靠它读 checkpoint 上**真实**的待处理集合。
        react_definition=guarded.definition,
        # 不接 dispatcher：这一节只验"校验过不过"，不验真的派发
        # （`_schedule_resume` 因此在 dispatcher 为 None 时早退）。
        post_commit=[],
    )

    async def _try(answers: list[ResumeAnswer]) -> str | None:
        try:
            await service.resume_run(
                guarded_run.run_id,
                ResumeRequest(answers=answers),
                user_id=None,
            )
        except ConflictError as exc:
            return str(exc)
        return None

    wrong = await _try(
        [ResumeAnswer(interrupt_id="0" * 32, data={"decision": "approve"})]
    )

    check("给一个不存在的 id -> ConflictError（少了也多了）", wrong is not None)

    check(
        "错误信息里列出了当前待处理的那个 id（调用方据此直接改）",
        guarded_interrupt in (wrong or ""),
    )

    extra = await _try([
        ResumeAnswer(
            interrupt_id=guarded_interrupt, data={"decision": "approve"}
        ),
        ResumeAnswer(interrupt_id="1" * 32, data={"decision": "reject"}),
    ])

    check("多带一个 → ConflictError", extra is not None)

    complete = await _try([
        ResumeAnswer(
            interrupt_id=guarded_interrupt, data={"decision": "approve"}
        )
    ])

    check(
        "凑齐了就不报错（本节 dispatcher 为空，所以只校验、不派发）",
        complete is None,
    )

    print("\n全部通过。")
    return 0


async def _load_approval_chain() -> list[Any]:
    """`ToolExecutor` 的 chain_loader：一条固定的审批挂载记录。"""

    return [_ApprovalRow()]


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
