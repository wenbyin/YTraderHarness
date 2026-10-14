"""
派发链路的冒烟测试：从运行行到图，再回到运行行与 agent_message 表。

不需要数据库，也**不调真模型** —— 用 `graph.py` 特意留出的 `NODE_FUNCTIONS`
那条缝把节点换成桩（与 agent_smoke.py 同一手法），仓储换成一个内存实现。

验的是这条链路上最容易静默断掉的几段：

    运行行读得到          handler 拿 run_id 从库里取到 input
    图的 config 对        缺 thread_id 时 agent 节点会直接抛
    终态真的写回去        成功 -> completed，异常 -> failed + error_message
    消息真的落下来        且**只落本轮产出的那些**

最后一条是重点。图的 messages 通道里是整条 thread 的历史，而
"本轮新增"是按条数切出来的 —— 切错了不会报错，只会静静地把前几轮重复写
一遍、或者把本轮的 HumanMessage 当成 assistant 回复写进去。因此下面有一组
用例专门盯它：多轮之后只写新的一条、且一条 user 行都不多写。

跑法：
    python -m app.services.session.agent_run_smoke
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph.react.graph import NODE_FUNCTIONS, build_react_graph
from app.agent.graph.react.runtime import register_react_graph
from app.infra.graph import DispatchEnvelope, GraphRegistry
from app.services.tool.registry import tool

from .application.agent_run_dispatch import (
    NodeLogRecorder,
    build_dispatch_handler,
)
from .domain.entities import AgentMessage, AgentRun
from .domain.schemas import RunStatus
from app.services.memory.repository.atom import MemoryAtomRepository

from .repository.agent_run import AgentRunRepository
from .repository.message import MessageRepository

_USER_TEXT = "帮我查一下这个季度的财报"


# ============================================================
# 桩
# ============================================================


class _SmokeService:
    """
    一个**真**工具，只为了让注册表里有一条带点号的 key。

    `tool_key` 由 `source_ref` 推导（这里是 `session`，见 spec.py 的
    `derive_source_ref`），所以它是 `session.<方法名>` —— **带点**，
    正是 provider 不接受的那种。用来验 tools 节点能否把模型回传的
    provider 安全名翻回来（见 `_tool_key_of`）。
    """

    @tool(key="session_smoke.echo")
    def echo(self, text: str = "") -> str:
        """原样回显。"""

        return text


class _FakeModelRegistry:
    """节点被换成桩之后根本不会被问到，给个占位即可。"""

    def get_runtime(self, **kwargs: Any) -> Any:
        raise AssertionError(
            "冒烟测试不该真的去要模型 —— 节点应当已被替换成桩。"
        )


class _InMemoryRunRepository(AgentRunRepository):
    """
    内存版 AgentRunRepository。

    只实现派发链路用到的三个方法（get / mark_running / finish）——
    这正是 `AgentRunRepository` 里那三个的签名，接口对不上会当场炸。
    """

    def __init__(self, rows: dict[UUID, AgentRun]) -> None:
        self.rows = rows
        #: 按发生顺序记下每一次状态变更，用例据此断言"确实经过了 running"。
        self.transitions: list[str] = []

    async def add(self, run: AgentRun) -> AgentRun:
        self.rows[run.run_id] = run
        return run

    async def get(self, run_id: UUID) -> AgentRun | None:
        return self.rows.get(run_id)

    async def mark_running(self, run_id: UUID) -> None:
        self.rows[run_id].status = RunStatus.RUNNING.value
        self.transitions.append(RunStatus.RUNNING.value)

    async def finish(
            self,
            run_id: UUID,
            status: str,
            *,
            error_type: str | None = None,
            error_message: str | None = None,
    ) -> None:
        row = self.rows[run_id]
        row.status = status
        row.error_type = error_type
        row.error_message = error_message
        self.transitions.append(status)


class _InMemoryMessageRepository(MessageRepository):
    """
    内存版 MessageRepository。

    **只实现写路径用到的那几个**（next_seq / add_many / user_message_id）——
    写路径的冒烟测试不翻页，`list_page` 因此明确抛而不是返回空：返回空会让
    "翻页没接上"看起来像"这个会话没有消息"。翻页有它自己的
    message_smoke.py。
    """

    def __init__(self) -> None:
        self.rows: list[AgentMessage] = []

    async def next_seq(self, session_id: UUID) -> int:
        return max(
            (
                row.seq
                for row in self.rows
                if row.session_id == session_id
            ),
            default=0,
        ) + 1

    async def add_many(
            self,
            messages: Sequence[AgentMessage],
    ) -> list[AgentMessage]:
        self.rows.extend(messages)
        return list(messages)

    async def user_message_id(self, run_id: UUID) -> UUID | None:
        for row in self.rows:
            if row.run_id == run_id and row.role == "user":
                return row.message_id
        return None

    async def count_user_turns(
            self,
            session_id: UUID,
            *,
            after_seq: int | None = None,
    ) -> int:
        """
        该会话的用户轮次（`role='user'` 且 `kind='chat'`）。

        `after_seq` 给定时只数它**之后**的 —— L1 的触发要的是"自上次抽取
        以来攒了几轮"（水位是会话内的 `seq`，只当过滤条件用，**不能**拿
        总数去减它：两者单位不同）。

        与下面 `list_by_run` 同一条理由：L1 抽取的**触发**会在收尾路径上
        调它，所以这里真的实现，而不是抛 AssertionError —— 手工改一条
        消息再手工触发时，拿到一个说得通的计数比拿到一个断言有用。
        """

        return sum(
            1
            for row in self.rows
            if row.session_id == session_id
            and row.role == "user"
            and row.kind == "chat"
            and (after_seq is None or row.seq > after_seq)
        )

    async def list_chat_after(
            self,
            session_id: UUID,
            seq: int,
    ) -> list[AgentMessage]:
        """该会话 seq 之后的全部 chat 消息，按 seq 升序（L1 抽取的输入）。"""

        return sorted(
            (
                row
                for row in self.rows
                if row.session_id == session_id
                and row.seq > seq
                and row.kind == "chat"
            ),
            key=lambda row: row.seq,
        )

    async def list_by_run(self, run_id: UUID) -> list[AgentMessage]:
        """
        这一轮写下的全部消息，按 seq 升序。

        与 `list_page` 不同，这里**真的实现**（而不是抛）：它是
        `memory.extract` 任务读消息的形状，而那个任务是本冒烟脚本的
        handler 收尾时会**提交**的东西 —— 手工触发它时（补索引那类操作）
        拿到一个说得通的答案，比拿到一个 AssertionError 有用。

        内存版的排序按 seq 就地排 —— 真实现里 ORDER BY 由数据库做，
        这里 Fake 的职责只是"形状对得上"。
        """

        return sorted(
            (row for row in self.rows if row.run_id == run_id),
            key=lambda row: row.seq,
        )

    async def list_page(
            self,
            session_id: UUID,
            *,
            limit: int,
            before_seq: int | None = None,
            after_seq: int | None = None,
    ) -> Any:
        raise AssertionError(
            "写路径的冒烟测试不翻页 —— 翻页见 message_smoke.py。"
        )

    # ---- 下面三个是 memory 那侧新加的读方法（L2 浏览 / 按 id 溯源 / 画像计数）
    #
    # ⚠️ 它们**曾经缺失**，而症状是这条冒烟**连装都装不起来**：
    # MessageRepository 是抽象基类，少实现一个就没法实例化。报出来的是
    # `TypeError: Can't instantiate abstract class ... without an
    # implementation for abstract methods ...` —— 看起来像"冒烟测试本身坏
    # 了"，而真正的原因是别人往契约上加了一个方法，这个 Fake 没跟上。
    #
    # 这里按上面 `list_page` 的先例**明确抛**（而不是实现一个假的读）：
    # 写路径的用例不会调它们，而实现了却不对的读会让"某天真的调到了"表现
    # 成一条看起来正常的错误答案。
    #
    # 契约再加方法时，这三个也是照着补的位置。

    async def list_by_ids(
            self,
            message_ids: Sequence[UUID],
            *,
            session_id: UUID,
    ) -> list[AgentMessage]:
        raise AssertionError(
            "写路径的冒烟测试不按 id 溯源 —— 它属于 memory 的读路径。"
        )

    async def browse(self, **kwargs: Any) -> Any:
        raise AssertionError(
            "写路径的冒烟测试不浏览消息 —— 它属于 memory 的读路径。"
        )

    async def count_user_turns_in_scope(self, **kwargs: Any) -> int:
        raise AssertionError(
            "写路径的冒烟测试不跨会话计数 —— 它属于画像的触发路径。"
        )


class _FakeRepositories:
    """
    `RepositoryFactory.scope()` 交出来的那个东西。

    ⚠️ 按契约分发的字典，不是"只有一个仓储"
    -------------------------------------
    以前这里写的是 `assert contract is AgentRunRepository`。handler 加了
    写入消息之后会要第二个仓储，那种写法会当场炸 —— 但报出来的是
    AssertionError，看起来像"冒烟测试本身坏了"，而不是"handler 多要了一个
    依赖"。字典 + 一句说得清的报错，让下次加仓储时知道该改哪里。
    """

    def __init__(self, repositories: dict[type, Any]) -> None:
        self._repositories = repositories

    def create(self, contract: type) -> Any:
        repository = self._repositories.get(contract)

        if repository is None:
            raise AssertionError(
                f"冒烟测试没有为 {contract.__name__} 准备内存实现 —— "
                "handler 每多要一个仓储，这里就要多给一个。"
            )

        return repository


class _FakeMemoryAtomRepository:
    """
    L1 触发要的那**一个方法**：读水位。

    ⚠️ 本脚本其实走不到它
    -----------------
    `submit_atom_extract` 第一件事是 `get_service_factory()`，而本脚本不装
    生命周期（纯内存桩），那一步抛 RuntimeError 之后它直接返回 None ——
    所以这个假仓储平时不会被问到。留着它是为了两件事：契约上一旦有人给
    本脚本装上工厂，这里的报错会是"少给了一个仓储"而不是别的东西；以及
    "handler 每多要一个仓储，这里就要多给一个"这句话有个落点。

    水位给 0 = "这个会话还没抽过"，于是触发算出的待抽轮次就是全部用户轮次。
    """

    async def get_cursor(self, session_id: UUID) -> Any:
        return SimpleNamespace(last_seq=0, last_scene_name="")


class _FakeRepositoryFactory:
    """
    冒充 RepositoryFactory。

    handler 只用到 `scope(begin=...)` 这一个入口，因此这里只需要把它做成
    一个异步上下文管理器 —— 生产环境交给它的那个会自己开短事务。
    """

    def __init__(self, repositories: dict[type, Any]) -> None:
        self._repos = _FakeRepositories(repositories)

    @contextlib.asynccontextmanager
    async def scope(self, *, begin: bool = True) -> Any:
        yield self._repos


# ============================================================
# 断言工具（与 agent_smoke.py 同一套）
# ============================================================


def check(label: str, condition: Any) -> None:
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}")
    if not condition:
        raise AssertionError(label)


def section(title: str) -> None:
    print(f"\n{title}")


def _has_tool_round(state: Any) -> bool:
    """
    state 里是否已经出现过"要求调工具"的 AIMessage。

    桩 agent 靠它只发起一次工具轮 —— 本图是循环，每次都要调工具的
    桩会一路转到 recursion_limit。
    """

    from langchain_core.messages import AIMessage

    return any(
        isinstance(message, AIMessage) and message.tool_calls
        for message in state.get("messages") or ()
    )


# ============================================================
# 用例
# ============================================================


def _seed_run(
        rows: dict[UUID, AgentRun],
        text: str,
        *,
        session_id: UUID | None = None,
) -> AgentRun:
    """
    直接往"库"里塞一行 queued 的运行 —— 绕开 create_run。

    因此这里**没有用户消息**（那条是 create_run 写的，见
    agent_run_service_impl._record_user_message）。写路径的用例据此断言
    "handler 不会自己造一条 user 行"。
    """

    now = datetime.now(timezone.utc)
    run = AgentRun(
        run_id=uuid4(),
        session_id=session_id or uuid4(),
        status=RunStatus.QUEUED.value,
        input={"text": text},
        meta={},
        created_at=now,
        updated_at=now,
    )
    rows[run.run_id] = run
    return run


async def _dispatch(
        *,
        agent_raises: bool = False,
        text: str = _USER_TEXT,
) -> tuple[AgentRun, _InMemoryRunRepository, list[str]]:
    """
    跑一次完整的派发：建链路 -> 建运行 -> 调 handler。

    单次用例的便捷入口。多轮的用例直接建 `_Harness`（同一个链路跑两次），
    见下面那一节。
    """

    harness = _Harness(agent_raises=agent_raises)

    run = await harness.run(text)

    return run, harness.runs, harness.passed


class _Harness:
    """
    一条装配好的派发链路：图 + handler + 内存仓储 + 节点走位记录。

    做成类而不是一个函数，是因为**多轮**用例要在同一条 thread 上连着跑两次
    —— 那必须复用同一个 checkpointer 与同一个 handler。每次重建会得到两个
    互不相干的会话，而"会不会把上一轮的消息重复写一遍"恰恰只在共享 thread
    的前提下才暴露得出来。
    """

    def __init__(
            self,
            *,
            agent_raises: bool = False,
            agent_tool_calls: bool = False,
            tool_keys: list[str] | None = None,
            tool_call_name: str = "demo_tool",
            reasoning: str | None = None,
    ) -> None:

        self.passed: list[str] = []
        self.rows: dict[UUID, AgentRun] = {}

        self.runs = _InMemoryRunRepository(self.rows)
        self.messages = _InMemoryMessageRepository()

        #: 会话 id。多次 run() 共用它 —— 也就是共用一条 thread，
        #: 与生产一致（thread_id 按**会话**稳定，见 agent_run_service_impl）。
        self.session_id = uuid4()

        self.recorder = NodeLogRecorder()

        self.definition = self._build_graph(
            agent_raises=agent_raises,
            agent_tool_calls=agent_tool_calls,
            tool_keys=tool_keys,
            tool_call_name=tool_call_name,
            reasoning=reasoning,
        )

        self.handler = build_dispatch_handler(
            definition=self.definition,
            repository_factory=_FakeRepositoryFactory(  # type: ignore[arg-type]
                {
                    AgentRunRepository: self.runs,
                    MessageRepository: self.messages,
                    MemoryAtomRepository: _FakeMemoryAtomRepository(),
                }
            ),
            recorder=self.recorder,
        )

    def _build_graph(
            self,
            *,
            agent_raises: bool,
            agent_tool_calls: bool,
            tool_keys: list[str] | None,
            tool_call_name: str,
            reasoning: str | None,
    ) -> Any:
        """把节点换成桩再装配图。只有 build_dispatch_handler 与图本身是真的。"""

        passed = self.passed

        async def stub_builder(state: Any, runtime: Any) -> dict:
            passed.append("builder")

            if tool_keys is None:
                return {}

            return {"context_policy": {"tool_keys": list(tool_keys)}}

        async def stub_agent(state: Any, runtime: Any) -> dict:
            passed.append("agent")

            if agent_raises:
                raise NotImplementedError("假装工具执行入口没接通")

            from langchain_core.messages import AIMessage

            # ⚠️ 只在**还没有过工具轮**时要求调工具。
            #
            # 本图是 `agent ->(有 tool_calls)-> tools -> agent` 的循环，
            # 一个每次都被调、每次都要求调工具的桩会让它转到
            # recursion_limit（25 次）才报 GraphRecursionError ——
            # 那验的是"循环上限"，不是工具那条路。真模型的行为是
            # "调完工具、拿到结果、给出答复"，桩照这个来。
            if agent_tool_calls and not _has_tool_round(state):
                # 桩 builder 没写 tool_keys 时允许集是空的 —— 那样
                # `tool_call_name` 属于"没暴露"，会被拒绝执行并回一条
                # error ToolMessage，然后带着它回到这里。
                #
                # 传了 tool_keys 时允许集非空，`tool_call_name` 用的是
                # **provider 安全名**（下划线形式）—— 那一组用例验的是
                # tools 节点能不能把它翻回 tool_key。
                return {
                    "messages": [
                        AIMessage(
                            content="我来查一下",
                            tool_calls=[
                                {
                                    "name": tool_call_name,
                                    "args": {"q": 1},
                                    "id": "call_1",
                                }
                            ],
                        )
                    ]
                }

            # ⚠️ 思维链走 additional_kwargs，**不进 content** —— 本仓库实测的
            # DeepSeek 形状（见 transformer.log）。只在这一处注入，用来验
            # "它到底有没有落进 agent_message"。
            kwargs: dict = {}

            if reasoning is not None:
                kwargs["additional_kwargs"] = {"reasoning_content": reasoning}

            return {"messages": [AIMessage("答复", **kwargs)]}

        async def stub_planner(state: Any, runtime: Any) -> dict:
            passed.append("planner")
            return {}

        original = dict(NODE_FUNCTIONS)

        try:
            # ⚠️ 必须在 build_react_graph() **之前**替换（装配时按 node_id 取函数）。
            NODE_FUNCTIONS["builder"] = stub_builder
            NODE_FUNCTIONS["agent"] = stub_agent
            NODE_FUNCTIONS["planner"] = stub_planner
            # tools 刻意**不换** —— 带 tool_calls 那个用例要的就是真实的
            # tools 节点（它现在能跑通了，见那一节的断言）。

            return register_react_graph(
                registry=GraphRegistry(),
                checkpointer=InMemorySaver(),
                recorder=self.recorder,
                model_registry=_FakeModelRegistry(),  # type: ignore[arg-type]
            )

        finally:
            NODE_FUNCTIONS.clear()
            NODE_FUNCTIONS.update(original)

    async def run(self, text: str = _USER_TEXT) -> AgentRun:
        """
        在这个会话上再跑一轮。

        thread_id 只由 session_id 决定（不带 run_id）—— 与生产一致：
        thread 按会话稳定，多轮因此累积在同一条 thread 上。
        """

        run = _seed_run(self.rows, text, session_id=self.session_id)

        await self.handler(
            DispatchEnvelope(
                graph_name=self.definition.name,
                pipeline_run_id=run.run_id,
                thread_id=f"agentsess:{self.session_id}",
                payload={"session_id": str(self.session_id)},
            )
        )

        return self.rows[run.run_id]

    # ---- 断言辅助 --------------------------------------------------

    def assistant_rows(self) -> list[AgentMessage]:
        """写下来的 assistant 消息，按 seq 排序。"""

        return sorted(
            (row for row in self.messages.rows if row.role == "assistant"),
            key=lambda row: row.seq,
        )

    def user_rows(self) -> list[AgentMessage]:
        return sorted(
            (row for row in self.messages.rows if row.role == "user"),
            key=lambda row: row.seq,
        )


async def main() -> int:

    # --------------------------------------------------------
    section("成功：queued -> running -> completed")

    run, repo, passed = await _dispatch()

    check(
        "终态是 completed",
        run.status == RunStatus.COMPLETED.value,
    )
    check(
        "确实经过 running（不是 queued 直接跳终态）",
        repo.transitions == [
            RunStatus.RUNNING.value,
            RunStatus.COMPLETED.value,
        ],
    )
    check("成功时不带错误信息", run.error_type is None)
    check(
        "图从 builder 走到 agent（入口没有被绕过）",
        passed[:2] == ["builder", "agent"],
    )

    # --------------------------------------------------------
    section("成功：本轮产出的 assistant 消息落库")

    harness = _Harness()

    run = await harness.run()

    assistant = harness.assistant_rows()

    check(
        "写下了恰好一条 assistant 消息",
        len(assistant) == 1,
    )
    check(
        "role / kind 对",
        assistant[0].role == "assistant"
        and assistant[0].kind == "chat",
    )
    check(
        "status 跟着运行的终态（completed）",
        assistant[0].status == "completed",
    )
    check(
        "run_id 指向本次运行",
        assistant[0].run_id == run.run_id,
    )
    check(
        "内容被包成内容块数组",
        assistant[0].content == [{"type": "text", "text": "答复"}],
    )
    check(
        "seq 从 1 起",
        assistant[0].seq == 1,
    )

    # ⚠️ 这一条专盯最容易静默出错的那个坑
    # ------------------------------------
    # 图交给 handler 的 state 里包含**本轮输入的那条 HumanMessage**
    # （_initial_state 放的），而"本轮新增"是按 invoke 之前的条数切出来的
    # —— 不把它滤掉，它会被当成 assistant 回复再写一条，而这一行不会报错，
    # 只是前端会看到用户的话被重复渲染成模型的回复。
    check(
        "没有把本轮的 HumanMessage 当 assistant 写下来",
        harness.user_rows() == [],
    )

    # --------------------------------------------------------
    section("思维链：落进独立的列，不进 content 的块数组")

    # ⚠️ 这一条专盯"静默丢字段"
    # ----------------------
    # 从消息到接口之间有三处**逐字段**拷贝（to_entities / add_many /
    # build_message），漏一处都不报错：漏在写库那一处是库里恒 NULL（用户
    # 永远看不到推理），漏在读那一处是库里有值但接口不返回（更难查）。
    # 所以这里要同时看两件事 —— 那一位有值，而且 content 里**没有**推理块
    # （两个容器重叠会让前端把同一段渲染两遍）。
    harness = _Harness(reasoning="先看要不要查工具，再答。")

    run = await harness.run()

    assistant = harness.assistant_rows()

    check(
        "思维链写进了 reasoning_content 那一列",
        assistant[0].reasoning_content == "先看要不要查工具，再答。",
    )
    check(
        "content 里没有推理块（只放可见内容）",
        [block["type"] for block in assistant[0].content] == ["text"],
    )

    # 对照组：模型没给思维链时是 None —— 不是空串（空串会让前端渲染一个
    # 永远空着的推理框，见 tests/agent/context/test_message_reasoning.py）。
    plain = _Harness()

    await plain.run()

    check(
        "没有思维链时是 None，不是空串",
        plain.assistant_rows()[0].reasoning_content is None,
    )

    # --------------------------------------------------------
    section("多轮：只写本轮新增的那条")

    # 同一个 harness 跑第二轮 —— 共享 thread，checkpoint 里现在有 3 条消息
    # （H1 A1，加本轮的 H2）。切错就会把第一轮重复写一遍。
    run = await harness.run(text="接着说")

    assistant = harness.assistant_rows()

    check(
        "第二轮之后仍然只有两条 assistant 消息（第一轮没被重复写）",
        len(assistant) == 2,
    )
    check(
        "两条的 seq 连续递增",
        [row.seq for row in assistant] == [1, 2],
    )
    check(
        "第二轮的 run_id 指向第二次运行，不是第一次",
        assistant[-1].run_id == run.run_id
        and assistant[0].run_id != run.run_id,
    )

    # --------------------------------------------------------
    section("失败：异常被写回运行行")

    harness = _Harness(agent_raises=True)

    run = await harness.run()
    repo = harness.runs
    passed = harness.passed

    check(
        "终态是 failed",
        run.status == RunStatus.FAILED.value,
    )
    check(
        "异常类名落了库",
        run.error_type == "NotImplementedError",
    )
    check(
        "异常文本落了库（规格不回这个字段，只能从库里看）",
        "工具执行入口没接通" in (run.error_message or ""),
    )
    check(
        "节点确实跑过 —— 失败是图里发生的，不是没接上",
        "agent" in passed,
    )
    check(
        "模型还没产出就抛了 —— 因此没有 assistant 消息",
        harness.messages.rows == [],
    )

    # --------------------------------------------------------
    section("工具调用没被暴露：拒绝执行，但整轮照常完成")

    # 这条用例走的是**真实的** tools 节点（上面两个都是桩，它刻意不换）。
    # 桩 builder 返回空的 state，于是没有任何 tool_keys -> 允许集为空
    # -> 模型要求的 demo_tool 属于"没暴露"那一种，被拒。
    #
    # 它验的是那条立场：**拒绝是诚实的工具结果，不是运行失败**。
    # 模型看到一条 error ToolMessage 之后还能继续（这里因为桩 agent
    # 只有一条固定回复，路由直接走到 END）。整轮记成 completed，
    # 而"工具没跑"这件事写在那条 ToolMessage 的 status 上。
    harness = _Harness(agent_tool_calls=True)

    run = await harness.run()

    assistant = harness.assistant_rows()

    check(
        "整轮完成（一个工具调用被拒不该把整轮判死）",
        run.status == RunStatus.COMPLETED.value,
    )
    check(
        "三条都落库了：要求调工具 -> 工具结果 -> 拿着结果的答复",
        len(assistant) == 3,
    )
    check(
        "kind 依次是 tool_activity / tool_activity / chat",
        [row.kind for row in assistant]
        == ["tool_activity", "tool_activity", "chat"],
    )
    check(
        "第一条是 AIMessage：文本块与 tool_call 块都在",
        [block["type"] for block in assistant[0].content]
        == ["text", "tool_call"],
    )
    check(
        "tool_call 块带上了工具名与参数",
        assistant[0].content[1]["name"] == "demo_tool"
        and assistant[0].content[1]["args"] == {"q": 1},
    )
    check(
        "第二条是 ToolMessage：正文说明了是哪个工具被拒",
        [block["type"] for block in assistant[1].content] == ["text"]
        and "demo_tool" in assistant[1].content[0]["text"],
    )
    check(
        "第三条是带着工具结果的答复",
        assistant[2].content[0]["text"] == "答复",
    )

    # --------------------------------------------------------
    section("模型回传的是 provider 安全名，tools 节点要认得")

    # ⚠️ 这一条盯的是转义的**入站那一半**。
    #
    # tool_key 形如 `session_smoke.echo`（带点），provider 只接受
    # `^[a-zA-Z0-9_-]+$`，所以出站时转成了 `session_smoke_echo`
    # （见 infra/agent/tool_exposure.py）。模型回传的自然是转过的名字，
    # 而 tools 节点的允许集是**按 tool_key 算的** —— 翻不回去就会被
    # 当成"模型幻觉出一个没暴露的工具"而拒绝。
    #
    # 那个症状极难查：模型明明调了它看得见的工具，却被告知"不在本次
    # 可用的集合里"，而两侧各自都"工作正常"。
    #
    # 本 smoke 没有 ServiceFactory，执行那一步会降级成"工具目录不可用"
    # —— 那没关系：要验的就是**过了允许集那一关**，不再回"未暴露"。
    harness = _Harness(
        agent_tool_calls=True,
        tool_keys=["session_smoke.echo"],
        tool_call_name="session_smoke_echo",
    )

    run = await harness.run()

    assistant = harness.assistant_rows()

    check(
        "整轮照常完成",
        run.status == RunStatus.COMPLETED.value,
    )
    check(
        "三条都落库了",
        len(assistant) == 3,
    )
    check(
        "tool_call 块里是**模型回传的那个名字**（原样保留，不就地改写）",
        assistant[0].content[1]["name"] == "session_smoke_echo",
    )
    check(
        "没有被当成「未暴露」拒绝",
        "不在本次可用的工具集合中" not in assistant[1].content[0]["text"],
    )
    check(
        "而是走到了执行那一步（本 smoke 无 ServiceFactory，降级为目录不可用）",
        "工具目录当前不可用" in assistant[1].content[0]["text"],
    )

    # 反证：同一个工具，模型回一个**没转义**的名字 —— 那是注册表里
    # 真实存在的 tool_key，但**不在本次暴露的那一批的安全名里**，
    # 所以照样要被拒。这条盯的是"翻译不能宽到把原始 key 也放进来"。
    harness = _Harness(
        agent_tool_calls=True,
        tool_keys=["session_smoke.echo"],
        tool_call_name="session_smoke.echo",
    )

    await harness.run()

    assistant = harness.assistant_rows()

    check(
        "回原始 tool_key（没转义）-> 同样按未暴露拒绝",
        "不在本次可用的工具集合中" in assistant[1].content[0]["text"],
    )

    # --------------------------------------------------------
    section("空白输入：queued -> failed，且不进图")

    run, repo, passed = await _dispatch(text="   ")

    check(
        "终态是 failed",
        run.status == RunStatus.FAILED.value,
    )
    check(
        "没有先亮一下 running —— 这次运行根本没有开始过",
        repo.transitions == [RunStatus.FAILED.value],
    )
    check(
        "图一次都没跑：空输入在进图之前就被拒了",
        passed == [],
    )
    check(
        "原因写的是输入为空",
        run.error_type == "ValueError",
    )

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
