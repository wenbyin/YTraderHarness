"""
MEMORY_RETRIEVER（阶段 5）的冒烟测试：四层注入 / 保活 / 降级 / 跳过。

    .venv/Scripts/python.exe -m app.agent.context.processor.memory_smoke

**不需要数据库，也不需要模型** —— 这一层的全部输入都是假的：

    MemoryQueryService.retrieval_preview   换成一台按剧本应答的假服务
                                           （注入点是 processors.memory 的
                                           模块级 `get_service_factory`）
    ContextBuildContext                    直构，不跑 FastAPI 也不跑图

于是本脚本验的是**这一层自己的取舍**（四段怎么铺、INITIAL 与 CONTINUATION
的分工、保活 payload 的形状、哪些情况下什么都不做），而不是召回本身好不好。

⚠️ 为什么值得单独跑一条
------------------
"一次运行只召回一次"这件事**没有别的守卫**：它靠的是
`process()` 里对 `run.phase` 的一次比较，而写错的表现是"每过一个工具往返
就多付一次向量检索" —— 功能全对、只是变慢，不会有人发现。所以这里把
"CONTINUATION 一次都不许碰服务"写成硬断言。

⚠️ 与 `retrieval_preview` 那一侧的分工
-----------------------------------
本脚本只喂**假的** preview，因此它证明不了"预览返回的四段与真实召回一致"。
那一侧见 app/services/memory/memory_retrieval_smoke.py。
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from typing import Any
from uuid import UUID

from .context import BuildPhase, ContextBuildContext
from .dependencies import ContextDependencies

import app.agent.context.processor.processors.memory as memory_module
from app.agent.context.processor.processors.memory import MemoryProcessor

__all__: list[str] = []

_FAILURES: list[str] = []

SESSION = UUID("2f1a6b1e-1111-4111-8111-111111111111")

MESSAGE_ID = UUID("11111111-1111-4111-8111-111111111111")
RECORD_ID = UUID("33333333-3333-4333-8333-333333333333")

UNIT = {"l0": True, "l1": True, "agent_id": "general"}

#: 场景与记忆那两行渲染的是 `创建 → 更新`，要两个**不同**的时间才走那条
#: 分支（相等时只打一个）。这里给一对固定的值，断言里不能依赖"现在"。
OLDER = datetime(2026, 10, 10, 3, 20, tzinfo=timezone.utc)
NOW = datetime(2026, 10, 12, 2, 41, tzinfo=timezone.utc)


def check(label: str, condition: bool, detail: str = "") -> None:
    """断言并记账。失败**不中断**：一次跑完能看见所有问题。"""

    mark = "OK  " if condition else "FAIL"
    line = f"  [{mark}] {label}"

    if detail and not condition:
        line += f"  <- {detail}"

    print(line, flush=True)

    if not condition:
        _FAILURES.append(label)


# ============================================================
# 假服务
# ============================================================


def _preview(
        *,
        persona: bool = True,
        scenes: int = 2,
        scenes_total: int = 7,
        conversations: int = 2,
        facts: int = 1,
        degraded: tuple[tuple[str, str], ...] = (),
) -> NS:
    """一台按参数拼出来的 preview。形状对齐 MemoryRetrievalPrompt。"""

    sections: list[NS] = []

    if persona:
        sections.append(
            NS(
                block="persona",
                layer="l3",
                items=[NS(content="我是老张，做量化。", filename="persona.md",
                           version=3)],
            )
        )

    if scenes:
        sections.append(
            NS(
                block="scenes",
                layer="l2",
                scenes_total=scenes_total,
                items=[
                    NS(filename=f"s{index}.md", summary=f"聊过第 {index} 件事",
                       heat=index, created_at=OLDER, updated_at=NOW)
                    for index in range(scenes)
                ],
            )
        )

    if conversations:
        sections.append(
            NS(
                block="conversations",
                layer="l0",
                items=[
                    NS(role="user", text="不要自动下单", message_id=MESSAGE_ID,
                       created_at=NOW)
                ][:conversations],
            )
        )

    if facts:
        sections.append(
            NS(
                block="facts",
                layer="l1",
                items=[
                    NS(content="用户不允许自动下单", type="rule",
                       record_id=RECORD_ID, created_at=OLDER, updated_at=NOW)
                ][:facts],
            )
        )

    return NS(
        sections=sections,
        degraded=[NS(layer=layer, reason=reason) for layer, reason in degraded],
        # ⚠️ 这一格是管理端那条路用的整串。本层**不该**读它（读了会渲染出
        # 两层标题），所以故意填一个显眼的哨兵：谁改错了会立刻看见。
        prompt=NS(user="<<不该被读到>>"),
    )


class _FakeService:
    """按剧本应答的 retrieval_preview，并记下收到的请求。"""

    def __init__(self, result: Any = None, boom: Exception | None = None):
        self.requests: list[Any] = []
        self._result = result
        self._boom = boom

    async def retrieval_preview(self, request: Any) -> Any:
        self.requests.append(request)

        if self._boom is not None:
            raise self._boom

        return self._result


class _FakeFactory:
    def __init__(self, service: Any):
        self._service = service

    def scope(self, contract: type, *, begin: bool = True) -> Any:
        service = self._service

        class _Scope:
            async def __aenter__(self) -> Any:
                return service

            async def __aexit__(self, *exc: Any) -> bool:
                return False

        return _Scope()


def _install(service: Any) -> None:
    """把进程级服务工厂换成假的（只在 processors.memory 这个取用点上）。"""

    memory_module.get_service_factory = lambda: _FakeFactory(service)


def _context(phase: str = "initial") -> ContextBuildContext:
    return ContextBuildContext(
        base_state={},
        dependencies=ContextDependencies(),
        policy=None,
        phase=BuildPhase(phase),
    )


def _inputs(**over: Any) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "retrieve": UNIT,
        "phase": "initial",
        "user_id": "u-1",
        # ⚠️ 没有 `conversation_id` —— 处理器不再绑它，L0 恒跨会话
        # （见 processors/memory.py）。多给一格会让下一个读到这里的人
        # 以为会话维度还在。
        "query": "我上次说过什么",
        "persisted": None,
    }
    inputs.update(over)
    return inputs


async def main() -> int:
    processor = MemoryProcessor()

    # ========================================================
    # 1. INITIAL：真召回 -> 四段 -> 保活
    # ========================================================

    print("\n[1] INITIAL", flush=True)

    service = _FakeService(
        result=_preview(degraded=(("l1", "向量后端不可用"),))
    )
    _install(service)

    context = _context()
    await processor.process(_inputs(), context)

    check("只调一次服务", len(service.requests) == 1,
          f"实际 {len(service.requests)} 次")

    request = service.requests[0] if service.requests else None

    check("查询串就是原话", getattr(request, "input", None) == "我上次说过什么")
    check("L0/L1 按决策打开",
          request is not None and request.retrieve.l0 and request.retrieve.l1)
    check("top-k 是 5",
          request is not None and request.limits.l0 == 5
          and request.limits.l1 == 5)
    # ⚠️ **不传 session**：L0 与 L1 都跨会话。
    #
    # 传了它，L0 就被收窄到当前会话 —— 而分类里判 l0 的线索是「上次」
    # 「我之前说过」「我们聊过」这类，**多数发生在别的会话**。收窄之后
    # 那些线索判出的 l0=true 会去搜一个注定搜不到的范围。
    #
    # 实测：用户在新会话里问「你上次推荐我练什么曲子」-> 预召回搜本会话
    # （只有这一句问话）-> 空 -> 上下文里没有 L0 段 -> 主模型只好去调
    # 记忆检索**工具**（那个工具走 `scope=user`，跨会话）—— 同一个问题，
    # 两条路给出了不同的范围。
    check("L0 不带会话（跨会话召回）",
          request is not None and request.session_id is None,
          f"实际 {getattr(request, 'session_id', None)}")
    check("target 的身份来自 state",
          request is not None and request.target.agent_id == "general"
          and request.target.user_id == "u-1")

    labels = [section.label for section in context.sections]

    check("注入四段", len(context.sections) == 4, f"实际 {labels}")
    check("段序：画像 -> 场景 -> 原话 -> 记忆",
          # 画像那一格的标题带版本（`用户画像（v3）`）—— 版本是提示词正文
          # 的一部分（见 memory_retrieval 的 render_blocks），所以这里按
          # 完整标题断言，而不是只判前缀。
          labels[0] == "用户画像（v3）" and labels[1].startswith("记忆场景索引")
          and labels[2].startswith("相关原话") and labels[3].startswith("相关记忆"),
          f"实际 {labels}")
    check("四段共用 section 名（快照的 memory_ids 靠它）",
          {section.name for section in context.sections} == {"memory"},
          f"实际 {[s.name for s in context.sections]}")

    check("场景那段的计数写的是总数",
          "共 7 篇" in labels[1] and "以下 2 篇" in labels[1],
          f"实际 {labels[1]}")

    check("正文里不带 ## 标题（否则 system_message 会套两层）",
          all(not section.text.lstrip().startswith("## ")
              for section in context.sections))

    check("没有哪一段漏读了 prompt.user 那个哨兵",
          all("<<不该被读到>>" not in section.text
              for section in context.sections))

    check("source_ids：画像与场景用文件名，L0 用 message_id，L1 用 record_id",
          list(context.sections[0].source_ids) == ["persona.md"]
          and list(context.sections[1].source_ids) == ["s0.md", "s1.md"]
          and list(context.sections[2].source_ids) == [str(MESSAGE_ID)]
          and list(context.sections[3].source_ids) == [str(RECORD_ID)],
          f"实际 {[list(s.source_ids) for s in context.sections]}")

    check("每段都算了 token",
          all(section.token_estimate > 0 for section in context.sections))

    saved = context.metadata.get("context_state_updates", {}).get("memory")

    check("INITIAL 把分节交给 state 保活", isinstance(saved, dict))
    check("保活的是渲染好的四段",
          isinstance(saved, dict) and len(saved.get("blocks", [])) == 4)
    check("保活的 payload JSON 可序列化（要落 checkpoint / state）",
          _is_json_safe(saved))
    check("分层降级也记下来了",
          saved is not None and saved.get("degraded") == [
              {"layer": "l1", "reason": "向量后端不可用"}
          ],
          f"实际 {saved.get('degraded') if saved else None}")

    # ========================================================
    # 2. CONTINUATION：复用，且**一次都不碰服务**
    # ========================================================

    print("\n[2] CONTINUATION", flush=True)

    service.requests.clear()

    continued = _context("continuation")
    await processor.process(
        _inputs(phase="continuation", persisted=saved), continued
    )

    check("CONTINUATION 不调服务（一个运行只召回一次）",
          service.requests == [], f"实际 {len(service.requests)} 次")
    check("四段逐段一致（label 与正文都不变）",
          [s.label for s in continued.sections]
          == [s.label for s in context.sections]
          and [s.text for s in continued.sections]
          == [s.text for s in context.sections])
    check("CONTINUATION 不再写保活（state 通道不许每轮变）",
          "memory" not in (continued.metadata.get(
              "context_state_updates") or {}))

    # ========================================================
    # 3. 只开 L2/L3（决策说不用召回）
    # ========================================================

    print("\n[3] 不召回那一轮", flush=True)

    service = _FakeService(result=_preview(conversations=0, facts=0))
    _install(service)

    quiet = _context()
    await processor.process(
        _inputs(retrieve={"l0": False, "l1": False, "agent_id": "general"}),
        quiet,
    )

    check("只注入画像 + 场景索引两段", len(quiet.sections) == 2,
          f"实际 {[s.label for s in quiet.sections]}")
    check("L0/L1 的开关如实传给了服务",
          service.requests and not service.requests[0].retrieve.l0
          and not service.requests[0].retrieve.l1)

    # ========================================================
    # 4. 降级与跳过
    # ========================================================

    print("\n[4] 降级与跳过", flush=True)

    # ⚠️ 这一条钉的是"降级要**看得见**"：连服务都取不到时**不许**自己吞掉
    # 异常 —— 吞了就丢掉流水线那条 FAILED 与 warning，于是"记忆整段不可用"
    # 只剩一行日志，而快照才是排查时看的地方。
    # （层**内**的失败不走这里：那是 retrieval_preview 收进 degraded 的。）
    _install(_FakeService(boom=RuntimeError("工厂没装配")))
    broken = _context()

    raised: Exception | None = None

    try:
        await processor.process(_inputs(), broken)

    except RuntimeError as exc:
        raised = exc

    check("连服务都取不到 -> 抛出去（交给 optional 记 FAILED + warning）",
          raised is not None, "实际被吞掉了")
    check("抛之前什么都没注入", broken.sections == [],
          f"实际 {[s.label for s in broken.sections]}")
    check("也不留保活（免得下一轮拿一份空的分节）",
          "memory" not in (broken.metadata.get(
              "context_state_updates") or {}))

    # ⚠️ 这几个用例都**在碰服务之前**就返回了，所以上一步那台会抛的假服务
    # 不影响它们 —— 反过来说，正因如此它们证明不了"服务坏了会怎样"
    # （那一条已经在上面单独验过）。
    for label, over in (
        ("没有 user_id", {"user_id": ""}),
        ("没有 agent_id（duty_key 为空）",
         {"retrieve": {"l0": True, "l1": True}}),
        ("没有决策（builder 节点那条早返回）", {"retrieve": None}),
        ("没有查询串", {"query": ""}),
    ):
        skippable = _context()
        await processor.process(_inputs(**over), skippable)

        check(f"{label} -> 不注入", skippable.sections == [],
              f"实际 {[s.label for s in skippable.sections]}")

    # ⚠️ 这里原本还有一条"会话 id 不是 UUID -> 回落成跨会话"的用例，
    # 随 session 维度一起去掉了。
    #
    # 处理器现在**根本不读 `conversation_id`**（L0 恒跨会话，见
    # processors/memory.py 那一段），因此"解不成 UUID"这个情形结构上
    # 不可能发生。留一条永远为真的断言比没有更糟 —— 它看起来像覆盖。

    # ========================================================
    # 5. 排版只有一处（管理端那条串与这里的四段同源）
    # ========================================================

    print("\n[5] 排版同源", flush=True)

    from app.agent.prompt.prompts.memory_retrieval import (
        render_blocks,
        render_user_message,
    )

    preview = _preview()
    persona_section = preview.sections[0]
    scenes_section = preview.sections[1]

    blocks = render_blocks(
        persona=persona_section.items[0],
        scenes=scenes_section.items,
        scenes_total=scenes_section.scenes_total,
        conversations=preview.sections[2].items,
        facts=preview.sections[3].items,
    )

    joined = render_user_message(
        persona=persona_section.items[0],
        scenes=scenes_section.items,
        scenes_total=scenes_section.scenes_total,
        conversations=preview.sections[2].items,
        facts=preview.sections[3].items,
    )

    check("管理端的整串 = 四段按同一排版拼回去",
          joined == "\n\n".join(f"## {title}\n{body}"
                                for _, title, body in blocks))
    check("段标识与契约的 block 取值同词",
          [block for block, _, _ in blocks]
          == ["persona", "scenes", "conversations", "facts"])

    # ========================================================
    # 6. 整条流水线：槽 -> metadata -> 白名单
    # ========================================================
    #
    # ⚠️ 这一段才是"保活到底通没通"的答案
    # ---------------------------------
    # 上面 [1] 只证明处理器把分节放进了 `context.metadata` 那个**槽**里。
    # 槽到图 state 之间还有两跳：`_delegate` 合并进
    # `ModelRequest.metadata`，再由 agent_node 按白名单落到 state。
    # 那两跳写错的表现是"每轮都重新召回（或者干脆丢失）" —— 功能看着对，
    # 只是慢或者少，不会有人发现。所以这里把流水线整条跑一遍。

    print("\n[6] 槽 -> metadata -> 白名单", flush=True)

    from langchain_core.messages import HumanMessage

    from app.agent.context.builder.agent_invoke import (
        AgentInvokeProcessorContextBuilder,
    )
    from app.agent.context.builder.contracts import ContextBuildRequest
    from app.agent.graph.react.agent_node import _context_state_updates
    from app.common.context import ContextBuilderType

    service = _FakeService(result=_preview())
    _install(service)

    model_request = await AgentInvokeProcessorContextBuilder().build(
        ContextBuildRequest(
            builder_type=ContextBuilderType.AGENT_INVOKE,
            thread_id="t-1",
            messages=(HumanMessage(content="我上次说过什么", id="m-1"),),
            graph_state={
                "messages": (),
                "user_id": "u-1",
                "conversation_id": str(SESSION),
                "context_policy": {
                    "include_memory": True,
                    "memory_retrieve": UNIT,
                },
            },
            agent_type="react",
            model_role="reasoning",
            dependencies=ContextDependencies(),
        )
    )

    updates = (model_request.metadata or {}).get("context_state_updates", {})

    check("槽被 _delegate 收进了 ModelRequest.metadata",
          "memory" in updates, f"实际键 {sorted(updates)}")
    check("保活 payload 完整地过了流水线",
          len(updates.get("memory", {}).get("blocks", [])) == 4,
          f"实际 {updates.get('memory')}")

    system = next(
        (m for m in model_request.messages if m.type == "system"), None
    )

    system_text = getattr(system, "content", "") if system else ""

    check("四段确实进了 system 消息（各带一个 ## 标题）",
          all(f"## {label}" in system_text
              for label in ("用户画像", "记忆场景索引", "相关原话", "相关记忆")),
          f"实际 {system_text[:200]!r}")
    check("system 里没有两层标题（`## 用户画像` 只出现一次）",
          system_text.count("## 用户画像") == 1)

    applied = _context_state_updates(model_request)

    check("agent_node 的白名单放行了 memory",
          set(applied) == {"memory"}, f"实际 {sorted(applied)}")
    check("放行的那份就是处理器写的那份",
          applied.get("memory") == updates.get("memory"))

    # 白名单真的在挡：混进来的敏感键必须被丢掉。
    poisoned = NS(
        metadata={
            "context_state_updates": {
                "memory": {"blocks": []},
                "messages": [{"id": "伪造的消息"}],
                "context_result": {"伪造": "快照"},
                "summary": {"text": "合法"},
                "unknown": 1,
            }
        }
    )

    filtered = _context_state_updates(poisoned)

    check("白名单丢掉 messages / context_result / unknown（处理器不许篡改对话）",
          set(filtered) == {"memory", "summary"}, f"实际 {sorted(filtered)}")
    check("非 Mapping 的值也丢掉（state 那三个通道都是 dict）",
          _context_state_updates(
              NS(metadata={"context_state_updates": {"memory": ["不是 dict"]}})
          ) == {})

    # ========================================================

    print("\n" + "=" * 60)

    if _FAILURES:
        print(f"失败 {len(_FAILURES)} 项：")
        for label in _FAILURES:
            print(f"  - {label}")
        return 1

    print("全部通过。")
    return 0


def _is_json_safe(value: Any) -> bool:
    """payload 要落 checkpoint —— 里头不能有 UUID / datetime / pydantic。"""

    import json

    try:
        json.dumps(value, ensure_ascii=False)

    except (TypeError, ValueError):
        return False

    return True


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
