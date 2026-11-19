"""
实体 -> 对外契约的组装。

纯函数、不碰数据库。放在这里而不是各服务里各写一份：会话有“按 id 取”与
“列表”两个读点，但"契约长什么样"只该有一处 —— 各拼一份必然会漂移
（一个带 status 一个不带那种）。列表的分页壳同样只在这里拼。

消息那一组已经印证了这条：读路径在 MessageService（列表）与将来的
"按 id 取单条"之间共享同一个 build_message，改字段只改这里。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from app.agent.event.tool_presentation import (
    build_display,
    parse_tool_output,
    shape_tool_output,
)

from ..domain.entities import AgentMessage as AgentMessageEntity
from ..domain.entities import AgentRun as AgentRunEntity
from ..domain.entities import AgentSession as AgentSessionEntity
from ..domain.schemas import (
    AgentMessage,
    MessagePage,
    Pagination,
    Run,
    Session,
    SessionPage,
    encode_cursor,
)

__all__ = [
    "build_message",
    "build_message_page",
    "build_run",
    "build_session",
    "build_session_page",
]


def build_session(entity: AgentSessionEntity) -> Session:
    """
    会话实体 -> 契约。

    ⚠️ 这里**丢掉**了 meta 与 created_at —— 不是漏写，是契约里没有它们
    （session.yaml 的 Session 只有 session_id / agent_id / status /
    user_id / updated_at）。契约本身的不对称记在
    domain/schemas/session.py 里。
    """

    return Session(
        session_id=entity.session_id,
        agent_id=entity.agent_id,
        status=entity.status,
        user_id=entity.user_id,
        updated_at=entity.updated_at,
    )


def build_session_page(
        entities: Sequence[AgentSessionEntity],
        total: int,
) -> SessionPage:
    """
    一页会话实体 -> 契约。

    排序由仓储保证（updated_at 倒序、session_id 倒序），这里**不重排** ——
    组装层再排一次会让“仓储的排序”与“接口的排序”变成两份，迟早漂移。
    """

    return SessionPage(
        items=[build_session(entity) for entity in entities],
        total=total,
    )


def build_run(entity: AgentRunEntity) -> Run:
    """
    运行实体 -> 契约。同样丢掉 input / meta / 时间戳 / caller_identity，见上。

    ⚠️ `caller_identity` 是**刻意丢掉**的，不是漏写：它是服务端鉴权用的
    身份快照（groups / roles），契约里没有这个字段。逐字段构造让"新增
    一个实体字段"默认不会外泄 —— 别为了省事改成 `**entity.model_dump()`，
    那会把内部字段一起送出去。
    """

    return Run(
        run_id=entity.run_id,
        session_id=entity.session_id,
        status=entity.status,
        # 客户端自己传过的值；回显它是为了让"这次运行有没有开 trace"
        # 在运行详情里可见，不引入新信息。
        trace_id=entity.trace_id,
    )


#: 存量行里推理块的类型名（新行只有 `reasoning_content` 那一列）。
_REASONING_BLOCK_TYPE = "reasoning"

#: 工具块。`tool_result` 是本次改动新增的；`text` 是 ToolMessage 的原文。
_TOOL_CALL_BLOCK = "tool_call"
_TOOL_RESULT_BLOCK = "tool_result"
_TEXT_BLOCK = "text"

#: 哨兵：这一条消息里没有可解析的工具结果原文。
_MISSING = object()


def _raw_output_from_text(content: Sequence[Any]) -> Any:
    """
    从 ToolMessage 的文本块里还原工具结果。

    ToolMessage 的正文就是原始结果（成功结果带 `<ref:tool>` 包装）。用它
    可以在**读**的时候重算 output / display，让旧行（写的时候还没解开
    包装的行）自愈，也让摘要/列始终跟着当前配置走。
    """

    text = "\n".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, Mapping)
        and block.get("type") == _TEXT_BLOCK
    ).strip()

    if not text:
        return _MISSING

    parsed = parse_tool_output(text)

    return _MISSING if parsed is None else parsed


def _filter_tool_blocks(
        content: list[dict[str, Any]],
        presentations: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    按 presenter 过滤 / 补全 content 里的工具块。

    `presentations is None` = **不启用**（老调用方 / 测试）：原样返回。
    `presentations == {}` = 启用但没有工具配置展示 -> 工具块全部摘掉，
    与实时流的默认隐藏同一条口径。

        tool_call    没有 presenter -> 摘块；
                     有但没 display（本次改动前的旧行）-> 用当前配置补
                     （summary 只用 input，completed 的那份在结果里）。
        tool_result  没有 presenter -> 摘块，**并摘掉同一消息的文本块**
                     —— 那条文本是工具结果原文，前端虽不渲染，接口不该回。
        其它块        旁白 / 引用照旧保留（模型的话不能因为旁边有工具调用就丢）。

    过滤只发生在**输出**层：`agent_message` 里的原文不动，排障仍可查库。
    """

    if presentations is None:
        return content

    filtered: list[dict[str, Any]] = []
    hidden_result = False

    raw_output = _raw_output_from_text(content)

    for block in content:

        if not isinstance(block, Mapping):
            filtered.append(block)  # type: ignore[arg-type]
            continue

        block_type = block.get("type")

        if block_type == _TOOL_CALL_BLOCK:

            presentation = presentations.get(
                str(block.get("name") or "")
            )

            if presentation is None:
                continue

            enriched = dict(block)

            # 工具名统一成 tool_key：老行里 tool_call 的 name 是 wire
            # name（AIMessage 的形态），而事件与 tool_result 块用 tool_key。
            # 读时归一，历史与实时就有了同一种写法。
            canonical = getattr(presentation, "tool_key", None)

            if isinstance(canonical, str) and canonical:
                enriched["name"] = canonical

            if "display" not in enriched:
                enriched["display"] = build_display(
                    presentation,
                    input_value=enriched.get("args"),
                )

            filtered.append(enriched)
            continue

        if block_type == _TOOL_RESULT_BLOCK:

            presentation = presentations.get(
                str(block.get("name") or "")
            )

            if presentation is None:
                hidden_result = True
                continue

            enriched = dict(block)

            canonical = getattr(presentation, "tool_key", None)

            if isinstance(canonical, str) and canonical:
                enriched["name"] = canonical

            # 失败结果自愈：旧行把错误文本当 output、summary 渲染成
            # 「共  条」。实时侧已经改成"失败只有 error"，这里按同一
            # 口径重算，旧行刷新后也一致。
            if str(enriched.get("status") or "") == "error":
                enriched.pop("output", None)
                enriched["display"] = build_display(
                    presentation,
                    output_value=None,
                )

                filtered.append(enriched)
                continue

            if (
                raw_output is not _MISSING
                and isinstance(raw_output, (Mapping, list))
            ):
                # 读时重算：旧行（写的时候还没解 `<ref:tool>`）自愈；
                # 新行重算结果与落库值相同；改配置后历史也跟着新配置走。
                # 与实时/落库共用一个组合点。
                shaped = shape_tool_output(presentation, raw_output)

                if shaped.has_output:
                    enriched["output"] = shaped.output

                enriched["display"] = shaped.display

            elif "display" not in enriched:
                enriched["display"] = build_display(
                    presentation,
                    output_value=enriched.get("output"),
                )

            filtered.append(enriched)
            continue

        filtered.append(dict(block))

    if hidden_result:
        filtered = [
            block
            for block in filtered
            if not (
                isinstance(block, Mapping)
                and block.get("type") == _TEXT_BLOCK
            )
        ]

    return filtered


def build_message(
        entity: AgentMessageEntity,
        *,
        include_reasoning: bool = True,
        presentations: Mapping[str, Any] | None = None,
) -> AgentMessage:
    """
    消息实体 -> 契约。

    ⚠️ 这里做了两次改名，方向相反，别搞混：

        session_id   实体有、契约**没有**   -> 丢掉（见下）
        meta         实体有、契约叫 metadata -> 改名

    丢掉 session_id 与上面两个 builder 丢掉 meta / 时间戳是同一条规矩：契约
    里没有的东西不硬塞（messages.yaml 的 Message 不含它）。它是查询的输入
    —— 一整页里每条都相同，逐条回显只是噪声。那一份在 MessagePage 上。

    `include_reasoning=False`（`sinks.chat.reasoning_enabled=false`）
    时**两个容器一起清**：
        · `reasoning_content` 那一列（新行）；
        · content 里 `type=reasoning` 的块（存量行，那一列是后加的）。
    只清一个，等于换一种方式把推理漏出去。库里照旧保留原文 —— 这是
    **输出**开关，不是存储开关。

    `presentations` 是 `tool_key / wire name -> 展示描述符` 的查找表。
    `None` = 不做工具过滤（老调用方）；传表（哪怕是空表）就启用默认隐藏：
    没有 presenter 的工具调用块不出现在历史里，与实时流一致。见
    `_filter_tool_blocks`。
    """

    content = _filter_tool_blocks(entity.content, presentations)
    reasoning_content = entity.reasoning_content

    if not include_reasoning:

        content = [
            block
            for block in content
            if not (
                isinstance(block, dict)
                and block.get("type") == _REASONING_BLOCK_TYPE
            )
        ]
        reasoning_content = None

    return AgentMessage(
        message_id=entity.message_id,
        seq=entity.seq,
        run_id=entity.run_id,
        role=entity.role,
        kind=entity.kind,
        business_agent_id=entity.business_agent_id,
        content=content,
        # 逐字段构造（不是 **entity.model_dump()）的代价：漏一行就是
        # "库里有、接口没有"，而且不报错。见模块 docstring。
        reasoning_content=reasoning_content,
        status=entity.status,
        reply_to_message_id=entity.reply_to_message_id,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
        metadata=dict(entity.meta),
    )


def build_message_page(
        *,
        session_id: UUID,
        entities: Sequence[AgentMessageEntity],
        limit: int,
        has_more_before: bool,
        has_more_after: bool,
        include_reasoning: bool = True,
        presentations: Mapping[str, Any] | None = None,
) -> MessagePage:
    """
    一页实体 -> 契约。

    两个游标由**本页的首尾**编出来（`entities` 按 seq 升序，仓储保证）：

        before_cursor -> 最旧那条   拿它往旧翻
        after_cursor  -> 最新那条   拿它往新翻

    空页时两个都是 None（连同两个 false）—— 客户端据此收手，而不是拿着
    null 再请求一次。这与仓储那条"空页两个标志都回 False"是配套的。
    """

    return MessagePage(
        session_id=session_id,
        messages=[
            build_message(
                entity,
                include_reasoning=include_reasoning,
                presentations=presentations,
            )
            for entity in entities
        ],
        pagination=Pagination(
            limit=limit,
            has_more_before=has_more_before,
            before_cursor=(
                encode_cursor(entities[0].seq) if entities else None
            ),
            has_more_after=has_more_after,
            after_cursor=(
                encode_cursor(entities[-1].seq) if entities else None
            ),
        ),
    )
