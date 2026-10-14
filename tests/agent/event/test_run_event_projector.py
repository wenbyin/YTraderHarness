"""
投影层的两道门：**节点** + **角色**。

为什么必须有角色这道门
--------------------
一次运行里的模型调用不止"给用户写答复"一种。上下文压缩
（`model_role=summarization`）就跑在 agent 节点**里面**，它的
`langgraph_node` 也是 `"agent"` —— 按节点过滤拦不住它。实测（Redis 里的
运行流）：压缩结果以 2400 字的形式被当成助手正文推给了用户。

⚠️ 这个文件必须跑**真的图**
--------------------------
`langgraph_node` / `model_role` 都住在协议事件的 metadata 里，而 metadata
是 langchain 的 run 对象带来的 —— 手搓事件 dict 喂给投影层，测不出
"metadata 里到底有没有那个键"。所以这里跑真的 `astream_events(version="v3")`
+ 真的投影层 + 真的 `LangChainProviderAdapter`，只把 chat 客户端换成假模型
（`chat_model_factory` 那个注入口，见 adapters.py）。整条链是：

    ModelRuntime（role 注入）-> adapters（config={"metadata": …}）
        -> langchain on_chat_model_start -> 协议事件 -> 投影层
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from fakes import make_definition, make_registry

from app.agent.event import run_event_projector as projector_module
from app.agent.event.run_event_projector import projector_factory
from app.agent.model.errors import ModelProviderError
from app.agent.model.leaked_tool_call import LEAK_MARKER_TOKEN
from app.agent.model import (
    ModelProfile,
    ModelRequest,
    ModelRole,
    ModelRoleBinding,
)
from app.agent.model.adapters import LangChainProviderAdapter


#: 模型名 -> 那条假客户端会回复的正文。**按名字取**而不是按调用顺序：
#: 顺序依赖会让"哪次调用是哪一次"变成测试里一个没人看着的假设。
_TEXTS = {
    "summarizer": "SUMMARY-TEXT",
    "main": "FINAL-ANSWER",
}

#: 同上，但每个模型都**带思维链**（走 DeepSeek 的 additional_kwargs 形状）。
#: 两个都有是刻意的：可见性那道门只有在"内部调用也产出了同一种东西"时才
#: 被真正验到 —— 只让主循环有推理，漏发与不漏发看起来一模一样。
_THINKING = {
    "summarizer": "SUMMARY-THINKING",
    "main": "MAIN-THINKING",
}


# ============================================================
# 装置
# ============================================================


def _leak_sample() -> str:
    """
    "模型把工具调用写成正文"的最小形状（用常量拼，理由见 leak 模块）。

    真的那次长得多（两个 invoke 块、参数齐全），但闸认的形状就这些。
    """

    token = LEAK_MARKER_TOKEN

    return (
        "<" + token + " calls>\n"
        + "<" + token + ' invoke name="memory_atom_search">\n'
        + "</" + token + " invoke>\n"
        + "</" + token + " calls>"
    )


def _fake_factory(**kwargs: Any) -> GenericFakeChatModel:
    """
    替掉 `app.common.llm.get_chat_model`（适配器的注入口）。

    ⚠️ 同一个回复给**几份**
    ---------------------
    客户端按模型定义缓存，重试会**再调同一个实例** —— 而
    `GenericFakeChatModel` 的入参是一个迭代器，只给一份的话第二次调用会抛
    `StopIteration`。那看起来像"模型抽风"，实际是替身不够用（本来会掩盖被测
    行为：重试路径根本没被走到）。
    """

    return GenericFakeChatModel(
        messages=iter([AIMessage(content=_TEXTS[kwargs["model"]])] * 4)
    )


def _fake_factory_with_reasoning(**kwargs: Any) -> GenericFakeChatModel:
    """
    同上，但回复里**带思维链**（`additional_kwargs["reasoning_content"]`）。

    这是本仓库真实的 DeepSeek 形状：实测那一轮 `content=''`、推理在
    additional_kwargs 里（见 transformer.log）。它同时也会让 langchain 在
    聚合后的消息里造出一个 `type=reasoning` 的块 —— 两条路都走一遍。
    """

    name = kwargs["model"]

    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content=_TEXTS[name],
                    additional_kwargs={"reasoning_content": _THINKING[name]},
                )
            ]
            * 4
        )
    )


def _registry(factory: Any = _fake_factory) -> Any:
    """两个角色各一个 profile，都走同一个真适配器 + 假客户端。"""

    adapter = LangChainProviderAdapter(chat_model_factory=factory)

    return make_registry(
        models=[make_definition("summarizer"), make_definition("main")],
        profiles=[
            ModelProfile(id="sum-p", primary="fake:summarizer"),
            ModelProfile(id="main-p", primary="fake:main"),
        ],
        bindings=[
            ModelRoleBinding(
                agent_type="react",
                model_role=ModelRole.SUMMARIZATION,
                profile="sum-p",
            ),
            ModelRoleBinding(
                agent_type="react",
                model_role=ModelRole.REASONING,
                profile="main-p",
            ),
        ],
        adapter=adapter,
    )


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _compiled(registry: Any, *, node: str, roles: Sequence[str]) -> Any:
    """
    一张一句话的图：节点按 `roles` 依次调模型，只把**最后一次**的回复
    写回 state（与真图一致 —— 压缩的产物不进 messages）。
    """

    async def _node(state: _State) -> dict[str, Any]:

        response = None

        for role in roles:
            runtime = registry.get_runtime(agent_type="react", model_role=role)
            response = await runtime.ainvoke(
                ModelRequest(messages=(HumanMessage("hi"),))
            )

        return {"messages": [response.message]}

    graph: StateGraph = StateGraph(_State)
    graph.add_node(node, _node)
    graph.add_edge(START, node)
    graph.add_edge(node, END)

    return graph.compile(checkpointer=InMemorySaver())


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """把事件截在投影层的出口，不碰 Redis。"""

    events: list[Any] = []

    async def _capture(
            type: Any,
            topic_id: str,
            *,
            data: Any = None,
            node: str | None = None,
    ) -> bool:

        events.append(
            SimpleNamespace(
                type=getattr(type, "value", str(type)),
                topic_id=topic_id,
                node=node,
                data=dict(data or {}),
            )
        )
        return True

    monkeypatch.setattr(projector_module, "publish_run_event", _capture)

    return events


async def _drive(
        compiled: Any,
        *,
        message_nodes: Sequence[str],
        message_roles: Sequence[str],
) -> None:
    """跑完一条运行 —— transformer 只有被这个循环泵着才会拿到事件。"""

    run = await compiled.astream_events(
        {"messages": [HumanMessage("hello")]},
        {"configurable": {"thread_id": "projector-test"}},
        version="v3",
        transformers=[
            projector_factory(
                run_id="run-under-test",
                message_nodes=set(message_nodes),
                message_roles=set(message_roles),
            ),
        ],
    )

    async with run:
        await run.output()


def _of(events: Sequence[Any], type: str, role: str | None = None) -> list[Any]:
    return [
        event
        for event in events
        if event.type == type and (role is None or event.data.get("role") == role)
    ]


# ============================================================
# 用例
# ============================================================


async def test_only_the_allowed_role_sends_message_content(published: list[Any]) -> None:
    """
    压缩（summarization）与主循环（reasoning）在**同一个节点**里各调一次，
    只有主循环的正文出得去。
    """

    compiled = _compiled(
        _registry(),
        node="agent",
        roles=(ModelRole.SUMMARIZATION, ModelRole.REASONING),
    )

    await _drive(compiled, message_nodes={"agent"}, message_roles={ModelRole.REASONING})

    deltas = _of(published, "message.delta")

    # ---- 正文：一个字都不许来自压缩 ----
    assert "".join(item.data["delta"] for item in deltas) == "FINAL-ANSWER"

    # ---- 顺序：压缩那条的边界**在任何 delta 之前**就发完了 ----
    # 这条钉住的是"不需要缓冲"：不可见与否在 message-start 那一刻就定了，
    # 所以它的正文根本来不及被发出去 —— 不是发出去再撤回（撤不回）。
    message_types = [
        item.type for item in published if item.type.startswith("message.")
    ]
    assert message_types[:2] == ["message.started", "message.completed"]
    assert published[0].type == "node.started"  # 节点边界之后才是消息边界
    assert _of(published, "message.started")[0].data["role"] == ModelRole.SUMMARIZATION

    # ---- 边界事件：两条消息都有，前端靠它转 loading ----
    started = _of(published, "message.started")
    assert [item.data["role"] for item in started] == [
        ModelRole.SUMMARIZATION,
        ModelRole.REASONING,
    ]
    assert [item.data["visible"] for item in started] == [False, True]

    completed = _of(published, "message.completed")
    assert [item.data["role"] for item in completed] == [
        ModelRole.SUMMARIZATION,
        ModelRole.REASONING,
    ]
    assert [item.data["content"] for item in completed] == ["", "FINAL-ANSWER"]
    assert [item.data["visible"] for item in completed] == [False, True]

    # 两条 started 都挂在 agent 节点上 —— 这正是"节点门拦不住它"的证据。
    assert {item.node for item in started} == {"agent"}


async def test_empty_role_whitelist_sends_boundaries_but_no_content(
        published: list[Any],
) -> None:
    """
    白名单为空 = 一条正文都不发（与 `message_nodes` 同口径）。

    这条**不是**洁癖：它钉住的是"注入断掉时的表现"。那种情况下事件流上
    只剩边界，前端会一直转圈、但**不会**把内部消息当答复渲染出来 ——
    少发正文比漏内部消息好。
    """

    compiled = _compiled(
        _registry(),
        node="agent",
        roles=(ModelRole.REASONING,),
    )

    await _drive(compiled, message_nodes={"agent"}, message_roles=set())

    assert _of(published, "message.delta") == []
    assert [item.data["content"] for item in _of(published, "message.completed")] == [""]

    started = _of(published, "message.started")
    assert [item.data["visible"] for item in started] == [False]


async def test_a_leaked_tool_call_never_reaches_the_client(
        published: list[Any],
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    模型把工具调用写成正文时：**一条 delta 都不发**，边界事件照发，运行**失败**。

    实测那次（run 51b31758）是 149 条 delta 与 completed 全文都是那段标记，
    而运行表记的是 `completed` —— 那种"成功"最难查。所以这里三层一起钉住：

        标记不出流（一条 delta 都没有）
        边界事件照发、正文空串（前端不会卡在"以为还有下文"）
        **整次运行抛错**（模型层拒绝把"只有标记"当成一次答复；重试与降级
        因此仍有机会救回来 —— 闸把 delta 都拦下了，所以没有"已经推给用户
        的内容"，retry 是合法的）

    重试会对同一个模型再来一次，所以 started / completed 可能有好几对。
    """

    monkeypatch.setitem(_TEXTS, "main", _leak_sample())

    compiled = _compiled(
        _registry(),
        node="agent",
        roles=(ModelRole.REASONING,),
    )

    with pytest.raises(ModelProviderError):

        await _drive(
            compiled,
            message_nodes={"agent"},
            message_roles={ModelRole.REASONING},
        )

    # 标记一个字符都不出流。区段**之间**那点空白会漏成 delta，那是无害的
    # （completed 的正文是空串，订阅端按对账规则会把它抹掉）。
    assert all(
        item.data["delta"].strip() == ""
        for item in _of(published, "message.delta")
    )

    started = _of(published, "message.started")
    assert started and all(item.data["visible"] is True for item in started)

    completed = _of(published, "message.completed")
    assert completed
    # 正文同样只剩区段之间那点空白（不是""，因为没有哪个字符是"正文"）。
    assert all(item.data["content"].strip() == "" for item in completed)


async def test_other_nodes_stay_silent(published: list[Any]) -> None:
    """
    节点那道门没被角色这道门顶掉：别的节点（builder 的结构化输出）依旧
    一条 message.* 都不发。
    """

    compiled = _compiled(
        _registry(),
        node="builder",
        roles=(ModelRole.REASONING,),
    )

    await _drive(compiled, message_nodes={"agent"}, message_roles={ModelRole.REASONING})

    assert _of(published, "message.started") == []
    assert _of(published, "message.completed") == []


# ============================================================
# 思维链：自己的通道 + 同一套门
# ============================================================


async def test_reasoning_streams_on_its_own_channel_and_respects_the_gates(
        published: list[Any],
) -> None:
    """
    思维链走 `message.reasoning.*`，**不进** `message.delta` / `message.completed`。

    两条断言各挡一种错：

        1. 正文那条通道**一个字都不许混进推理** —— 契约明写它不含思维链，
           订阅端那边是"追字到聊天窗"，混进去就是把草稿纸当答复渲染。
        2. 压缩（summarization）也有思维链，但它**一条都不许出去**。
           这道门比正文那道更要紧：压缩的输入里有用户原文与历史，推理里
           往往还带着内部判据 —— 那是比漏一段正文更难解释的一种泄漏。

    顺序也钉住：全量推理在 `message.completed` **之前**发（前端按到达顺序
    渲染时，推理块因此落在正文上方，不必回头往已收尾的消息里插一段）。
    """

    compiled = _compiled(
        _registry(_fake_factory_with_reasoning),
        node="agent",
        roles=(ModelRole.SUMMARIZATION, ModelRole.REASONING),
    )

    await _drive(compiled, message_nodes={"agent"}, message_roles={ModelRole.REASONING})

    # ---- 1. 正文通道干净 ----
    assert "".join(item.data["delta"] for item in _of(published, "message.delta")) == "FINAL-ANSWER"
    assert [item.data["content"] for item in _of(published, "message.completed")] == [
        "",
        "FINAL-ANSWER",
    ]

    # ---- 2. 推理只来自主循环 ----
    assert "".join(
        item.data["delta"] for item in _of(published, "message.reasoning.delta")
    ) == "MAIN-THINKING"

    completed = _of(published, "message.reasoning.completed")

    assert len(completed) == 1
    # 载荷里**没有** visible：推理只在可见时发，那个字段恒为 True ——
    # 永远不变的字段比缺字段更糟（订阅端会为它写死分支）。见 event.py。
    assert completed[0].data == {
        "content": "MAIN-THINKING",
        "role": ModelRole.REASONING,
    }

    # ---- 3. 顺序 ----
    #
    # ⚠️ 事件流里有**摘要那条**的 completed（不可见、正文空串），它更早 ——
    # 拿"列表里第一个 message.completed"比顺序会比到它头上。所以看尾部：
    # 最后这一对必须是"推理全量 -> 消息收尾"，而且紧挨着。
    assert [item.type for item in published][0] == "node.started"

    order = [
        item.type
        for item in published
        if item.type in ("message.reasoning.completed", "message.completed")
    ]

    assert order[-2:] == ["message.reasoning.completed", "message.completed"]


async def test_reasoning_events_stay_silent_when_the_role_is_not_allowed(
        published: list[Any],
) -> None:
    """
    角色白名单为空时，**推理一条都不发**（与正文同一条门）。

    钉的是"注入整体失效"那种情形：边界事件还在（前端一直转圈），但推理
    与正文都不出去。少了这条，注入漏配会表现成"内部调用边想边说"。
    """

    compiled = _compiled(
        _registry(_fake_factory_with_reasoning),
        node="agent",
        roles=(ModelRole.REASONING,),
    )

    await _drive(compiled, message_nodes={"agent"}, message_roles=set())

    assert _of(published, "message.reasoning.delta") == []
    assert _of(published, "message.reasoning.completed") == []
    assert _of(published, "message.started") != []


async def test_reasoning_completed_is_a_real_reconciliation_point(
        published: list[Any],
) -> None:
    """
    `message.reasoning.completed` 的全文**来自整块**，不是 delta 拼接。

    ⚠️ 为什么这条要用合成 payload
    ---------------------------
    `GenericFakeChatModel` 只吃字符串 content（给它块列表会直接
    `ValueError: Expected content to be a string`），而"整块推理"这种形状
    恰恰要 provider 真的吐出推理块才走得到。所以这里直接喂协议事件的
    dict —— 这条路不涉及 metadata 里那两道门（另有用例在真图上验），
    验的是**投影规则本身**。

    与正文那条对账规则逐字对应（见 event.py）：completed 若是 delta 拼出来
    的，两边永远相等，那条规则就成了空转，"丢了几个 delta"再也检测不出来。
    这里让整块与分片刻意不同，就是为了证明取的是整块。

    顺带钉住 `text` 键的退让：有的 provider 把推理放在 `text` 里。
    """

    projector = projector_module.RunEventProjector(
        (),
        run_id="run-under-test",
        message_nodes={"agent"},
        message_roles={ModelRole.REASONING},
    )

    meta = {
        "langgraph_node": "agent",
        "model_role": ModelRole.REASONING,
        "run_id": "llm-run-1",
    }

    await projector._on_messages(({"event": "message-start", "id": "lc-1"}, meta))

    await projector._on_messages((
        {
            "event": "content-block-delta",
            "delta": {"type": "reasoning-delta", "reasoning": "分片一"},
        },
        meta,
    ))

    await projector._on_messages((
        # 整块用 `text` 键 —— 与 processor/messages.py 的退让同一条判据。
        {
            "event": "content-block-finish",
            "content": {"type": "reasoning", "text": "整块推理"},
        },
        meta,
    ))

    await projector._on_messages(({"event": "message-finish", "metadata": {}}, meta))

    completed = _of(published, "message.reasoning.completed")

    assert [item.data["content"] for item in completed] == ["整块推理"]

    # 分片照发（订阅端先看到它，随后被全量覆盖）。
    assert "".join(
        item.data["delta"] for item in _of(published, "message.reasoning.delta")
    ) == "分片一"

    # 非推理模型（没有思维链）时**一个空壳都不发**：这条用例的对照组。
    published.clear()

    other = projector_module.RunEventProjector(
        (),
        run_id="run-under-test",
        message_nodes={"agent"},
        message_roles={ModelRole.REASONING},
    )

    await other._on_messages(({"event": "message-start", "id": "lc-2"}, meta))
    await other._on_messages(({"event": "message-finish", "metadata": {}}, meta))

    assert _of(published, "message.reasoning.completed") == []
    assert _of(published, "message.reasoning.delta") == []
