"""
记忆上下文组装的冒烟测试：四层怎么拼 + L0 的 scope 怎么选。

    .venv/Scripts/python.exe -m app.services.memory.memory_retrieval_smoke

**不需要数据库，也不需要模型** —— 三次注入都是假的：

    五个仓储       换成只记调用、按剧本应答的假仓储
    两个召回服务   **显式传进构造函数**（`l0_recall` / `l1_recall`），
                   于是 `_recall_service` 走 override 那条路，
                   `get_service_factory()` 一次都不会被碰到

这条测试填的是一个真实的空白：`retrieval_preview` 在此之前**没有任何**
测试或冒烟覆盖（全仓 grep 只有契约、实现、路由三处）。

⚠️ 它验的是"组装"，不是"召回"
------------------------
召回本身（两路 + 融合 + 降级）在 memory_smoke.py / 那两个服务自己的测试里。
这里要钉住的是另外三件事：

    组装顺序与截断               画像 -> 场景 -> L0 -> L1，计数写总数
    L0 的 scope 由 session_id 决定   给了收窄到本会话，没给就跨会话
    L1 **恒**跨会话               哪怕请求带了 session_id 也不收窄

第三条是最容易写错的一条：它没有编译期保护，写错的表现是"L1 记忆莫名少
了一大半"，而那是静默的。
"""

from __future__ import annotations

import asyncio
import sys
import datetime as dt
from types import SimpleNamespace as NS
from typing import Any
from uuid import UUID, uuid4

from app.services.memory.application.memory_query_service_impl import (
    MemoryQueryServiceImpl,
)
from app.services.memory.domain.schemas.memory import MemorySearchScope
from app.services.memory.domain.schemas.memory_atom import (
    MemoryAtomSearchScope,
)
from app.services.memory.domain.schemas.query import (
    MemoryPreviewLayers,
    MemoryPreviewLimits,
    MemoryRetrievalRequest,
    RetrievalTarget,
)
from app.services.session.domain.schemas.enum import MessageRole

__all__: list[str] = []

_FAILURES: list[str] = []

SESSION = uuid4()

OTHER_SESSION = uuid4()

NOW = dt.datetime.now(dt.timezone.utc)

#: 比 `NOW` 早的一个时刻。场景与记忆那两行渲染的是 `创建 → 更新`，
#: 需要两个**不同**的时间才看得出那条分支 —— 相等时只会打一个。
OLDER = NOW - dt.timedelta(days=2)

#: 一份带 META 块的场景正文（格式见 memory_scene_consolidate 的 parse_meta）。
SCENE_META = """\
----- META-START -----
summary: {summary}
heat: {heat}
----- META-END -----

正文不重要。
"""


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
# 假仓储
# ============================================================


class _FakePersonas:
    def __init__(self, content: str | None):
        self.content = content
        self.scopes: list[Any] = []

    async def get_persona(self, scope: Any) -> Any | None:
        self.scopes.append(scope)

        if self.content is None:
            return None

        # `version` 进提示词（画像标题写成 `用户画像（v3）`），
        # `created_at` 是后加的列 —— 两处都要给，否则渲染时会 AttributeError。
        return NS(content=self.content, filename="persona.md", version=3,
                  created_at=OLDER, updated_at=NOW)


class _FakeScenes:
    def __init__(self, total: int):
        self.rows = [
            NS(
                filename=f"s{index:02d}.md",
                content=SCENE_META.format(summary=f"第 {index} 件事", heat=index),
                version=1,
                # created != updated：场景那行该渲染成 `创建 → 更新`。
                created_at=OLDER,
                updated_at=NOW,
            )
            for index in range(total)
        ]
        self.scopes: list[Any] = []

    async def list_scenes(self, scope: Any) -> list[Any]:
        self.scopes.append(scope)

        return list(self.rows)


# ============================================================
# 假召回服务
# ============================================================


class _FakeRecall:
    """L0 / L1 共用的假服务：记调用，按剧本回结果。"""

    def __init__(self, hits: list[Any], degraded: list[Any] | None = None):
        self.hits = hits
        self.degraded = degraded or []
        self.calls: list[dict[str, Any]] = []

    async def search(self, request: Any, **kwargs: Any) -> Any:
        self.calls.append({"request": request, **kwargs})

        return NS(results=list(self.hits), degraded_routes=list(self.degraded))


def _conversation_hit() -> Any:
    return NS(
        message_id=uuid4(),
        session_id=OTHER_SESSION,
        seq=12,
        role=MessageRole.USER,
        text="不要自动下单",
        created_at=NOW,
        score=0.5,
        matched_by=[NS(value="keyword")],
    )


def _atom_hit() -> Any:
    return NS(
        record_id=uuid4(),
        content="用户不允许自动下单",
        type="rule",
        priority=80,
        scene_name="风控",
        score=0.4,
        matched_by=[NS(value="vector")],
        # 两个时间来自 `memory_atom.created_time` / `updated_time`
        # （源列就叫这两个名字），段 DTO 里归一成 `*_at`。
        created_time=OLDER,
        updated_time=NOW,
    )


# ============================================================
# 组装
# ============================================================


def _service(
        *,
        persona: str | None = "我是老张，做量化。",
        scenes: int = 3,
        l0: _FakeRecall | None = None,
        l1: _FakeRecall | None = None,
) -> tuple[MemoryQueryServiceImpl, _FakePersonas, _FakeScenes]:
    personas = _FakePersonas(persona)
    scene_rows = _FakeScenes(scenes)

    service = MemoryQueryServiceImpl(
        atoms=None,          # type: ignore[arg-type] —— 预览用不到
        scenes=scene_rows,   # type: ignore[arg-type]
        personas=personas,   # type: ignore[arg-type]
        messages=None,       # type: ignore[arg-type]
        sessions=None,       # type: ignore[arg-type]
        # ⚠️ 显式传进来 -> `_recall_service` 走 override，**不碰服务工厂**。
        # 传 None 的话它会去 `get_service_factory()`，而脚本里没有装配。
        l0_recall=l0,
        l1_recall=l1,
    )

    return service, personas, scene_rows


def _request(
        *,
        target: RetrievalTarget,
        l0: bool = False,
        l1: bool = False,
        l2_limit: int = 12,
        session_id: UUID | None = None,
) -> MemoryRetrievalRequest:
    return MemoryRetrievalRequest(
        target=target,
        input="我上次说过什么",
        retrieve=MemoryPreviewLayers(l0=l0, l1=l1),
        limits=MemoryPreviewLimits(l0=5, l1=5, l2=l2_limit),
        session_id=session_id,
    )


def _blocks(prompt: Any) -> list[str]:
    return [section.block for section in prompt.sections]


async def main() -> int:
    user_target = RetrievalTarget(agent_id="general", user_id="u-1")

    # ========================================================
    # 1. 默认：只组装 L2/L3
    # ========================================================

    print("\n[1] 默认（不勾 L0/L1）", flush=True)

    l0 = _FakeRecall([_conversation_hit()])
    l1 = _FakeRecall([_atom_hit()])

    service, personas, scenes = _service(l0=l0, l1=l1)

    prompt = await service.retrieval_preview(
        _request(target=user_target)
    )

    check("只出现画像与场景两段",
          _blocks(prompt) == ["persona", "scenes"], f"实际 {_blocks(prompt)}")
    check("两层召回**一次都没跑**",
          l0.calls == [] and l1.calls == [],
          f"L0 {len(l0.calls)} 次 / L1 {len(l1.calls)} 次")
    check("画像读到了", prompt.prompt.user.startswith("## 用户画像"))
    check("L0/L1 的段也**不出现**在 prompt.user 里（不是空段）",
          "## 相关原话" not in prompt.prompt.user
          and "## 相关记忆" not in prompt.prompt.user)
    check("scope 是按 target 折出来的", len(personas.scopes) == 1)

    # ========================================================
    # 2. 勾上 L0/L1，且**不传** session_id
    # ========================================================

    print("\n[2] 勾 L0/L1，不传 session_id", flush=True)

    l0 = _FakeRecall([_conversation_hit()])
    l1 = _FakeRecall([_atom_hit()])

    service, _, _ = _service(l0=l0, l1=l1)

    prompt = await service.retrieval_preview(
        _request(target=user_target, l0=True, l1=True)
    )

    check("四段按契约顺序排",
          _blocks(prompt) == ["persona", "scenes", "conversations", "facts"],
          f"实际 {_blocks(prompt)}")

    l0_call = l0.calls[0]

    check("没给 session_id -> L0 跨会话（scope=user）",
          l0_call["request"].scope == MemorySearchScope.USER,
          f"实际 {l0_call['request'].scope}")
    check("没给 session_id -> L0 也不带 session 参数",
          l0_call["session_id"] is None)
    check("L0 的条数用 limits.l0", l0_call["request"].limit == 5)
    check("L0 的身份是显式入参的 user_id",
          l0_call["user_id"] == "u-1")
    check("L1 恒跨会话（scope=user）",
          l1.calls[0]["request"].scope == MemoryAtomSearchScope.USER)
    # ⚠️ 按**完整行**断言，不是子串
    # --------------------------
    # 这两行现在都带时间戳（L0 是 `- [user] <创建时间> 正文`，L1 是
    # `[类型] <创建 → 更新> 正文`），而子串断言在"时间是空的"那种回归下
    # 照样会绿 —— 那正是这次改动要盯的东西之一。
    check("L0 命中的片段带上创建时间进了 相关原话 那段",
          f"- [user] {NOW.isoformat(timespec='minutes')} 不要自动下单"
          in prompt.prompt.user)
    check("L1 命中带上了类型标记与创建→更新时间",
          f"[rule] {OLDER.isoformat(timespec='minutes')}"
          f" → {NOW.isoformat(timespec='minutes')} 用户不允许自动下单"
          in prompt.prompt.user)

    # ========================================================
    # 3. 勾上 L0/L1，**传** session_id —— 只有 L0 收窄
    # ========================================================

    print("\n[3] 勾 L0/L1，传 session_id", flush=True)

    l0 = _FakeRecall([_conversation_hit()])
    l1 = _FakeRecall([_atom_hit()])

    service, _, _ = _service(l0=l0, l1=l1)

    await service.retrieval_preview(
        _request(target=user_target, l0=True, l1=True, session_id=SESSION)
    )

    l0_call = l0.calls[0]
    l1_call = l1.calls[0]

    check("给了 session_id -> L0 收窄到本会话（scope=session）",
          l0_call["request"].scope == MemorySearchScope.SESSION,
          f"实际 {l0_call['request'].scope}")
    check("会话 id 如实传给了 L0 那条路",
          l0_call["session_id"] == SESSION, f"实际 {l0_call['session_id']}")
    check("L1 **仍然**跨会话（scope=user）",
          l1_call["request"].scope == MemoryAtomSearchScope.USER,
          f"实际 {l1_call['request'].scope}")
    check("L1 **连参数都不带** session_id（带了就是把它削成本轮对话）",
          "session_id" not in l1_call, f"实际传了 {l1_call}")

    # ========================================================
    # 4. 团队 target：L0/L1 开不起来，但不整单失败
    # ========================================================

    print("\n[4] 团队 target", flush=True)

    l0 = _FakeRecall([_conversation_hit()])
    l1 = _FakeRecall([_atom_hit()])

    service, _, _ = _service(l0=l0, l1=l1)

    prompt = await service.retrieval_preview(
        _request(
            target=RetrievalTarget(agent_id="general", team_id="t-1"),
            l0=True, l1=True,
        )
    )

    check("团队 scope 下两层召回都没跑",
          l0.calls == [] and l1.calls == [],
          f"L0 {len(l0.calls)} 次 / L1 {len(l1.calls)} 次")
    check("两层进 degraded 并说明原因",
          {item.layer for item in prompt.degraded} == {"l0", "l1"},
          f"实际 {[item.layer for item in prompt.degraded]}")
    check("L2/L3 照常给（部分降级，不是整单失败）",
          _blocks(prompt) == ["persona", "scenes"], f"实际 {_blocks(prompt)}")
    check("团队 scope 下没有 conversation/fact 段",
          "## 相关原话" not in prompt.prompt.user
          and "## 相关记忆" not in prompt.prompt.user)

    # ========================================================
    # 5. L2 的截断必须看得见
    # ========================================================

    print("\n[5] L2 截断", flush=True)

    service, _, scenes = _service(scenes=30)

    prompt = await service.retrieval_preview(
        _request(target=user_target, l2_limit=12)
    )

    scenes_section = next(
        section for section in prompt.sections if section.block == "scenes"
    )

    check("给了 12 篇", scenes_section.returned == 12,
          f"实际 {scenes_section.returned}")
    check("总数写 30（不是 12）", scenes_section.scenes_total == 30,
          f"实际 {scenes_section.scenes_total}")
    check("truncated 为真", scenes_section.truncated)
    check("提示词里的计数是**总数**，条数是实际给的",
          "共 30 篇" in prompt.prompt.user and "以下 12 篇" in prompt.prompt.user,
          prompt.prompt.user.splitlines()[3:4])

    # ========================================================

    print("\n" + "=" * 60)

    if _FAILURES:
        print(f"失败 {len(_FAILURES)} 项：")
        for label in _FAILURES:
            print(f"  - {label}")
        return 1

    print("全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
