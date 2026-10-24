"""
把图产出的消息翻成实体并落库 —— 消息写入的两个点之一。

    用户消息      建运行那一步写（agent_run_service_impl._record_user_message）
    assistant 消息 本模块，由派发 handler 在运行终局确定后调用

⚠️ 为什么不是"运行结束时把所有消息一起写"
--------------------------------------
因为每条消息的落库时机应该是**它自己的结局确定的那一刻**：

    用户消息            接收时就确定了（一句话说完就是完整的）
    assistant 正常回复   整段生成完成时
    assistant 失败/中断   运行终局确定时（此时能拿到多少就是多少）

本轮 assistant 消息的结局与**运行的**结局是同一个时刻 —— 图跑完（或抛了）
那一刻才知道它生成了什么。因此本模块是在 handler 里被调用的，而不是在
建运行、也不是在图节点里。

⚠️ 不 import langchain
-------------------
判角色、取文本、读 tool_calls 都走
`app/agent/context/processor/messages.py` 的鸭子类型助手。那个模块的
docstring 写着它是"本项目里**唯一**一处知道消息长什么样"的地方 ——
在这里 `isinstance(msg, AIMessage)` 会让那句话变成假话，而且 langchain
换消息结构时要改两处。

副作用是冒烟测试可以直接喂 SimpleNamespace，不必造真的 langchain 消息。

⚠️ 已知缺口：模型生成到一半就报错，那段文字会丢
------------------------------------------
`agent_node` 只在整段生成完成后才产出一条 `AIMessage`（它内部把流式分片
聚合成一整条才返回）。模型调用中途抛异常时，已经生成的那段内容随异常一起
消失，没有任何地方存着 —— 于是"失败的 assistant 消息存已生成内容"这条只在
**图确实产出过消息**时才成立：

    带 tool_calls 的 AIMessage 已经落进 checkpoint，随后 tools 节点报错
        -> 那条 AIMessage 会被本模块写下来，status=failed（有内容）
    模型自己生成到一半报错
        -> 本轮没有任何 assistant 消息（内容确实丢了）

要真正做到后者，得改 agent_node 的流式聚合与异常语义（把已聚合的内容在
抛之前存下来）—— 那是另一块改动，不在本次范围内。这里写明，是为了不让
"失败的运行查不到回复"被当成 bug 反复排查。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from app.agent.context.processor.messages import (
    is_human_message,
    is_reasoning_block,
    is_system_message,
    is_tool_message,
    message_kind,
    message_reasoning,
    tool_calls,
)
from app.common.repository.factory import RepositoryFactory

from ..domain.entities import AgentMessage as AgentMessageEntity
from ..domain.schemas import (
    MessageKind,
    MessageRole,
    MessageStatus,
    RunStatus,
)
from ..repository.message import MessageRepository

__all__ = [
    "CANCELLATION_NOTICE_TEXT",
    "message_status",
    "new_assistant_messages",
    "record_cancellation_notice",
    "record_interaction",
    "record_messages",
    "to_entities",
]


#: 取消时补的那条记录的正文。
#:
#: ⚠️ 为什么要有固定文案，而不是留空 content
#: --------------------------------------
#: 这条记录是给**人**看的：它出现在会话历史里，作用就是解释"这里为什么
#: 什么都没有"。留空的话它在界面上通常就渲染成什么都没有 —— 那正是要修的
#: 那个症状。文案故意平淡且不带原因（"为什么取消"是运行生命周期的事，
#: 这里只负责记下"这一轮中断了"）。
CANCELLATION_NOTICE_TEXT = "（本次运行已取消）"


# ============================================================
# 该写哪些 / 写成什么状态
# ============================================================


def new_assistant_messages(
        prior_count: int,
        final_messages: Sequence[Any],
) -> list[Any]:
    """
    从图的终态里挑出**本轮新增、且该落库**的消息。

    ⚠️ 这个函数存在的唯一理由是它会漏掉东西
    -------------------------------------
    两个判断都不直观，而且错了都不会报错：

    1. **按条数切**。`ReactState.messages` 是整条 thread 的历史（checkpointer
       累积的），不是本轮增量 —— 直接全写会把前面每一轮都重复写一遍。
       `prior_count` 是 invoke **之前**数出来的条数，切片因此只剩本轮。

       ⚠️ 前提：`add_messages` 只追加。它其实还支持 RemoveMessage 墓碑与按
       id 替换，那时下标会移位、切片就切错了。今天没有节点发墓碑
       （compaction 只写 summary 通道），所以这条成立；真要用上墓碑时，
       这里必须换成按消息 id 做差集。

    2. **滤掉普通 human 与 system**。本轮的 `HumanMessage` 是
       `_initial_state` 交给图的，而 `prior_count` 是在那之前数的 ——
       不滤的话它一定会落在切片里。用户消息已由建运行那一步写过一条，
       再写一次就是**同一句话两行**，而且这一行会带着
       `role=assistant`（契约里只有 user / assistant 两个取值，
       没有第三个格子给它），前端会把它渲染成模型的回复。

       **例外**：带 `kind=task_instruction` 的 HumanMessage 要写。
       它不是用户说的那句话，是 planner / listener 注入的任务指令；
       前端任务面板需要它。它同样以 role=assistant 落库，靠 kind 区分。

       system 消息今天不在 state 通道上（只出现在组装好的 prompt 里），
       滤它是为了防止将来有人往 state 里塞。
    """

    tail = list(final_messages)[prior_count:]

    return [
        message
        for message in tail
        if not is_system_message(message)
        and (
            not is_human_message(message)
            # 任务指令是 HumanMessage，但它不是用户说的那句话，而是
            # planner / listener 注入的内部指令。它带着 kind 标记，必须
            # 落库（前端任务面板要它）；普通 HumanMessage 仍然滤掉 ——
            # 用户那条已经由建运行的事务写过。
            or message_kind(message) == MessageKind.TASK_INSTRUCTION.value
        )
    ]


def message_status(run_status: RunStatus) -> str:
    """
    运行的终态 -> 消息的终态。

    这个映射必须存在（而不是把 RunStatus 的值直接塞进去）：两个枚举的取值
    类似但**不是同一组**，且列上带 CHECK —— 塞错值的报错会离原因很远。
    逐条理由见 domain/schemas/enum.py 的 MessageStatus。

    ⚠️ `waiting_input` 不是 `interrupted`
    ------------------------------------
    HITL 中断时，assistant 消息是**完整产出**的：AIMessage 已经生成完、
    ToolMessage 也已经返回。暂停的是 **run**，不是某条消息。把消息标成
    interrupted 会让历史里所有 tool_activity 看起来都"被打断"，而它们
    其实都成功完成了。

    用户回答本身由 resume 路径写成一条 `kind=interaction` 的消息；"还在
    等人"这个事实由 `agent_run.status=waiting_input` 表达。

    Raises:
        ValueError: 传进来一个跑不到终态的状态（queued / running）。
                    走到这里说明调用方用错了这个函数，越早炸越好。
    """

    mapping = {
        RunStatus.COMPLETED: MessageStatus.COMPLETED,
        # 图停下来等人，但消息本身是完整的 —— 见 docstring。
        RunStatus.WAITING_INPUT: MessageStatus.COMPLETED,
        RunStatus.FAILED: MessageStatus.FAILED,
        RunStatus.CANCELLED: MessageStatus.CANCELLED,
    }

    resolved = mapping.get(run_status)

    if resolved is None:
        raise ValueError(
            f"{run_status} 不是终态，不能用来推消息的终态。"
            "本函数只接受 completed / waiting_input / failed / cancelled。"
        )

    return resolved.value


# ============================================================
# 实体组装（纯函数）
# ============================================================


def to_entities(
        messages: Sequence[Any],
        *,
        session_id: UUID,
        run_id: UUID,
        start_seq: int,
        status: str,
        business_agent_id: str | None,
        reply_to_message_id: UUID | None,
        created_at: datetime,
) -> list[AgentMessageEntity]:
    """
    消息 -> 实体，seq 从 `start_seq` 起连续分配。

    ⚠️ seq 连续是"这一次写入"的性质，不是整张表的：两次写入之间允许有洞
    （前一次写失败就留了洞）。翻页不依赖连续性。

    ⚠️ 时间戳只取一次
    ---------------
    同一批消息共用一个 `created_at`。逐条取 now() 会让它们差出几毫秒，而
    排序一律按 seq —— 时间戳在这里只是"什么时候写的"一个记录，不是顺序的
    依据（同一次写入里它本来就区分不出先后）。

    `updated_at` 直接填 created_at：消息不可变，写入之后没有代码会改它，
    两个时间戳因此**恒等**（见 models/sqlalchemy/agent_message.py 的说明）。
    这里不取第二次 now()，正是为了不让它们差出几毫秒、看起来像"被改过"。
    """

    return [
        AgentMessageEntity(
            message_id=uuid4(),
            session_id=session_id,
            run_id=run_id,
            seq=start_seq + offset,
            role=MessageRole.ASSISTANT.value,
            kind=_kind(message),
            # 产出这条消息的**职责**（duty_key）—— 由调用方从运行行上取
            # （agent_run.duty_key），这一批消息全部用它。
            business_agent_id=business_agent_id,
            content=_content_blocks(message),
            # 思维链单独一列，**不在** content 的块数组里（见 _content_blocks）。
            # 非推理模型取到 None —— 那一列因此是 NULL 而不是空串。
            reasoning_content=message_reasoning(message),
            status=status,
            # 幂等 ID 要客户端给，而消息不是客户端直接提交的。
            client_message_id=None,
            reply_to_message_id=reply_to_message_id,
            meta=_meta(message),
            created_at=created_at,
            # 同一个值，不取第二次 now() —— 见本函数 docstring。
            updated_at=created_at,
        )
        for offset, message in enumerate(messages)
    ]


# ============================================================
# 落库
# ============================================================


async def record_messages(
        repository_factory: RepositoryFactory,
        *,
        session_id: UUID,
        run_id: UUID,
        status: RunStatus,
        messages: Sequence[Any],
        business_agent_id: str | None = None,
) -> int:
    """
    把本轮产出的消息写进 agent_message 表，返回写了几条。

    ⚠️ 自开一段短事务，不借用任何请求事务
    -----------------------------------
    本函数在派发 handler 里被调用，而 handler 跑在调度器的后台 worker 里 ——
    那时建运行的请求事务早就提交了，连接也还回池子了（与 agent_run 的
    状态写回同一条理由，见 agent_run_repository_impl 的说明）。

    ⚠️ 调用方必须让异常传出去
    ----------------------
    这里**不吞异常**：写失败时调用方需要知道（它会把这次失败记进日志），
    但也不能因此跳过运行的终态写回 —— 那个决定在调用方手里，不在这里。

    Raises:
        ValueError: `status` 不是终态（编程错误）。
    """

    if not messages:
        # 一次运行完全没产出消息（图一进来就失败、或者输入在进图之前就被
        # 拒了）是常见的。为它开一段事务、查两次 max(seq) 纯属浪费。
        return 0

    resolved_status = message_status(status)

    async with repository_factory.scope() as repositories:

        message_repository = repositories.create(MessageRepository)

        # 取号与插入在同一个事务里（scope 包着），这是 next_seq 契约里
        # 要求的 —— 分开成两段就失去串行化的保证。
        start_seq = await message_repository.next_seq(session_id)

        # 回填"回复的是哪一条"。取不到就留空：那条 user 行由**建运行的
        # 请求事务**写下，正常路径上它一定在；不在的场合是"运行被直接
        # 喂进库的"（冒烟测试、将来的补跑工具），那时回复关联本就无从谈起。
        reply_to = await message_repository.user_message_id(run_id)

        entities = to_entities(
            messages,
            session_id=session_id,
            run_id=run_id,
            start_seq=start_seq,
            status=resolved_status,
            business_agent_id=business_agent_id,
            reply_to_message_id=reply_to,
            created_at=datetime.now(timezone.utc),
        )

        await message_repository.add_many(entities)

    return len(entities)


def _interaction_text(
        questions: Sequence[Mapping[str, Any]],
        answers: Mapping[str, str],
) -> str:
    """问答 -> 人能直接读的一段文本（结构化块另存一份）。"""

    lines: list[str] = []

    for question in questions:

        text = str(question.get("question") or "").strip()

        if not text:
            continue

        answer = answers.get(text, "")

        lines.append(f"Q: {text}\nA: {answer}")

    return "\n\n".join(lines)


async def record_interaction(
        repository_factory: RepositoryFactory,
        *,
        session_id: UUID,
        run_id: UUID,
        business_agent_id: str | None = None,
        questions: Sequence[Mapping[str, Any]],
        answers: Mapping[str, str],
        annotations: Mapping[str, Mapping[str, str]] | None = None,
        interrupt_id: str,
) -> int:
    """
    一次 HITL 问答 -> 一条 `role=user`、`kind=interaction` 的消息。

    为什么在 resume 路径写
    -------------------
    用户回答走的是 `Command(resume=...)`，不经过 `_initial_state`，所以
    普通消息记录器看不到这句话。如果不在这里补，历史里只有 assistant 的
    tool_activity，用户的回答只藏在 ToolMessage 的 JSON 里 —— 翻历史时
    "我问了什么、你答了什么"不可读。

    `content` 同时给两样：

        text 块         人直接读的 Q/A
        interaction 块  结构化 questions / answers / annotations，给前端渲染

    Raises:
        底层异常原样抛。调用方（resume 路径）决定是否吞掉 —— 写失败不该
        阻止图 resume，但必须留下日志。
    """

    if not questions:
        return 0

    now = datetime.now(timezone.utc)

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": _interaction_text(questions, answers),
        },
        {
            "type": "interaction",
            "questions": [dict(question) for question in questions],
            "answers": dict(answers),
            "annotations": {
                str(key): dict(value)
                for key, value in (annotations or {}).items()
            },
        },
    ]

    async with repository_factory.scope() as repositories:

        message_repository = repositories.create(MessageRepository)

        start_seq = await message_repository.next_seq(session_id)

        await message_repository.add_many([
            AgentMessageEntity(
                message_id=uuid4(),
                session_id=session_id,
                run_id=run_id,
                seq=start_seq,
                role=MessageRole.USER.value,
                kind=MessageKind.INTERACTION.value,
                business_agent_id=business_agent_id,
                content=content,
                reasoning_content=None,
                status=MessageStatus.COMPLETED.value,
                client_message_id=None,
                reply_to_message_id=None,
                meta={
                    "interrupt_id": interrupt_id,
                    "source": "ask_user_question",
                },
                created_at=now,
                updated_at=now,
            )
        ])

    return 1


async def record_cancellation_notice(
        repository_factory: RepositoryFactory,
        *,
        session_id: UUID,
        run_id: UUID,
        business_agent_id: str | None = None,
) -> int:
    """
    取消一次运行、而本轮**一行消息都没写成**时，补一条中断记录。

    ⚠️ 只在"一条都没写成"的时候补，有产出时不补
    ----------------------------------------
    有产出时那几条**已经**写下了、也已经带 `status=cancelled`（见
    `record_messages`）—— 再补一条只是把同一件事说两遍。

    真正需要它的是两种情形，症状是同一个：**用户在会话历史里只看到自己
    那句话，下面什么都没有。**

        排队期间就被取消    图一次都没跑，按定义没有任何产出
        跑了一半被取消      还没有一条**完整**的 AIMessage
                            （agent_node 整段生成完才产出，见本模块
                            docstring 的"已知缺口"）

    ⚠️ `kind=notice`，不是 `chat`
    ----------------------------
    这条**不是**模型说的话，是运行生命周期写下的公告。契约里 `notice`
    那一格就是为这种记录留的（messages.yaml 的 `Message.kind`），今天还
    没有别的生产者。用 `chat` 会让前端把它渲染成一次模型回复 —— 用户会以为
    模型真的回了这么一句。

    ⚠️ 自开一段短事务（与 `record_messages` 同一条理由）
    ------------------------------------------------
    调用方是调度器后台 worker 里的 handler，那时建运行的请求事务早就提交了。

    Raises:
        任何底层异常**原样抛**。调用方（派发 handler）决定要不要吞 —— 但
        这条记录是"取消可见"的唯一来源，吞掉它等于静默退回改之前的状态。
    """

    now = datetime.now(timezone.utc)

    async with repository_factory.scope() as repositories:

        message_repository = repositories.create(MessageRepository)

        # 取号与插入在同一个事务里（scope 包着）—— next_seq 契约要求。
        start_seq = await message_repository.next_seq(session_id)

        # 回填"回复的是哪一条"。取不到就留空：那条 user 行由建运行的请求
        # 事务写下，正常路径上一定在；不在的场合是运行被直接喂进库的
        # （冒烟测试、将来的补跑工具），那时回复关联本就无从谈起。
        reply_to = await message_repository.user_message_id(run_id)

        await message_repository.add_many([
            AgentMessageEntity(
                message_id=uuid4(),
                session_id=session_id,
                run_id=run_id,
                seq=start_seq,
                role=MessageRole.ASSISTANT.value,
                kind=MessageKind.NOTICE.value,
                # 与这一轮其它消息同一个职责 —— 公告也是这次运行产出的。
                business_agent_id=business_agent_id,
                content=[
                    {"type": "text", "text": CANCELLATION_NOTICE_TEXT}
                ],
                status=MessageStatus.CANCELLED.value,
                # 幂等 ID 要客户端给，而这条不是客户端提交的。
                client_message_id=None,
                reply_to_message_id=reply_to,
                meta={},
                created_at=now,
                # 同一个值：消息不可变，写入之后没有代码会改它
                # （见 to_entities 的说明）。
                updated_at=now,
            ),
        ])

    return 1


# ============================================================
# internals
# ============================================================


#: 任务消息的 kind。它们自带 `additional_kwargs.kind`，优先级高于
#: 工具/对话的推断。
_TASK_KINDS: frozenset[str] = frozenset({
    MessageKind.TASK_INSTRUCTION.value,
    MessageKind.TASK_RESULT.value,
})


def _kind(message: Any) -> str:
    """
    assistant 消息的种类。

        task_instruction / task_result          -> 自带标记优先
        工具结果（ToolMessage）                 -> tool_activity
        要求调用工具（AIMessage 带 tool_calls） -> tool_activity
        其余                                   -> chat

    一条既说了话、又调了工具的消息算 tool_activity —— 它的正文里两块都有
    （文本块 + tool_call 块），前端按 tool_activity 渲染工具卡片时文本
    仍然在旁边，不丢东西。
    """

    tagged = message_kind(message)

    if tagged in _TASK_KINDS:
        return tagged

    if is_tool_message(message) or tool_calls(message):
        return MessageKind.TOOL_ACTIVITY.value

    return MessageKind.CHAT.value


def _meta(message: Any) -> dict[str, Any]:
    """
    任务消息的归属信息。普通消息仍是空对象。

    task_id / plan_id / skill_key 是前端把消息挂到任务卡片所需的最小集；
    它们来自 `additional_kwargs`，不塞进 content —— content 只放正文。
    """

    if message_kind(message) not in _TASK_KINDS:
        return {}

    kwargs = getattr(message, "additional_kwargs", None)

    if not isinstance(kwargs, Mapping):
        return {}

    return {
        key: str(kwargs[key])
        for key in ("task_id", "plan_id", "skill_key")
        if kwargs.get(key)
    }


def _content_blocks(message: Any) -> list[dict[str, Any]]:
    """
    消息 -> 内容块数组。

    langchain 的 content 有三种形态（str / list / dict，见
    app/agent/context/processor/messages.py 的说明），统一成契约要求的数组：

        str            -> [{"type": "text", "text": ...}]
        list           -> 逐个规范化（字符串元素包成文本块，字典原样保留）
        dict           -> 当成单个块

    最后再把 tool_calls 附成 tool_call 块。**空内容是合法的**：模型返回空
    字符串时写下的就是一条 content 为空的记录 —— 那是它真实的行为，
    把它丢掉等于把"模型什么都没说"这件事从历史里抹掉。

    ⚠️ 推理块**不进这里**，它走 `reasoning_content` 那一列
    ---------------------------------------------------
    `content` 是"可见内容"，推理是模型的草稿纸：混在一起前端要么渲染两遍、
    要么得自己按块类型区分，而两处都在同一列里时"这一列到底是什么"就
    说不清了（memory 的 L0 索引与 L1 抽取都按 `type == text` 读它，靠的是
    这一列里只有可见内容）。

    这也让"纯推理消息"的形状变得确定：`content == []` 且
    `reasoning_content` 有值 —— 而不是 content 里塞着一个 reasoning 块。

    存量行是例外：过滤是在这次改动里加的，之前把推理放进 content 块的
    provider 写下的行里还留着那些块（前端要容忍，见类 docstring）。
    """

    blocks: list[dict[str, Any]] = []

    content = getattr(message, "content", None)

    if isinstance(content, str):
        if content:
            blocks.append({"type": "text", "text": content})

    elif isinstance(content, Mapping):
        if not is_reasoning_block(content):
            blocks.append(dict(content))

    elif isinstance(content, Sequence):
        for item in content:
            if isinstance(item, str):
                blocks.append({"type": "text", "text": item})
            elif isinstance(item, Mapping):
                # 已经带 type 的块（多模态）原样保留 —— 契约的 ContentBlock
                # 允许额外键，正是为了不必在这里穷举种类。
                # 唯一的例外是推理块：它另有列，见本函数 docstring。
                if not is_reasoning_block(item):
                    blocks.append(dict(item))

    # 工具调用附在最后。langchain 把它放在消息的 tool_calls 属性上而不是
    # content 里，但契约的 content 是"这条消息的全部内容块"，因此这里合并
    # —— 否则前端只看 content 就看不到"它调了哪个工具"。
    #
    # ⚠️ 按 id 去重，不能无条件追加
    # ---------------------------
    # 经 v3 协议流聚合出来的 AIMessage，`content` 里**已经**带着
    # `tool_call` 块（实测：文本块 + tool_call 块），与属性上的
    # `tool_calls` 是同一批调用。无条件追加会让每次工具调用在库里
    # 出现两遍（同一 id），前端按 content 渲染就是两张工具卡片。
    present = {
        str(block.get("id") or "")
        for block in blocks
        if isinstance(block, Mapping)
        and block.get("type") == "tool_call"
    }

    for call in tool_calls(message):
        call_id = str(call.get("id") or "")

        if call_id and call_id in present:
            continue

        blocks.append(
            {
                "type": "tool_call",
                "id": call_id,
                "name": str(call.get("name") or ""),
                "args": call.get("args") or {},
            }
        )

        if call_id:
            present.add(call_id)

    return blocks
