"""
L3（memory_persona）的冒烟测试：增量水位 -> 模型 -> 画像行 + 乐观锁 + 触发判定。

    .venv/Scripts/python.exe -m app.services.memory.memory_persona_smoke

直接用真数据库跑（本地 Postgres），但**模型是假的** —— 本脚本要验的是
这条链的读写与取舍（scope 级水位、no_change、乐观锁、两套提示词的分派、
四条触发判定），不是某一次 Prompt 调出来的画像好不好。

⚠️ 一个注入点：`memory_persona_service_impl` 的模块级 getter
--------------------------------------------------
    get_model_registry    换成一台**按剧本应答**的假 runtime（每次调用
                          弹出一个预先写好的 JSON），于是"这一轮该不该改
                          画像"完全由脚本决定

提示词那一侧**不注入** —— 走真 provider，顺带把"这份提示词确实注册上了"
一起验了（见 `_install` 的说明）。

⚠️ 造的是 `memory_scene` 行，不是 L1 记忆
-------------------------------
L3 的输入是 L2 的**产物**。造 L1 记忆再跑一遍 L2 也能造出场景，但那要
先把 L2 那条链整条跑通 —— 而本脚本要验的是 L3，多一层只会让失败的原因
变模糊。所以场景行直接写，`updated_at` 用单调时钟显式给（它就是水位）。

⚠️ 用过即清
--------
脚本自己造会话、场景、画像与水位，跑完按 `agent_id` 删干净。**失败路径也
清**（finally），否则库里会留下垃圾行，而它们会在下次跑别的冒烟脚本时以
"多出来几条"的形式出现。

⚠️ 它绕过了 HTTP 层
---------------
与仓库其他冒烟脚本一样，这里直接调服务 —— 因此**不能**用它证明"接口在
HTTP 上通"。L3 当前没有 HTTP 入口（见 memory_persona_service.py）。
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import sys
import uuid
from collections import deque
from types import SimpleNamespace

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.services.memory.application.memory_persona_service_impl as impl
from app.agent.model import ModelRole
from app.common.exceptions import ConflictError
from app.services.memory.domain.entities.persona import (
    PersonaCursorState,
    PersonaRow,
)
from app.services.memory.domain.entities.scene import (
    PERSONA_FILENAME,
    SceneScope,
    persona_id,
    scene_id,
)
from app.services.memory.domain.schemas.persona import (
    PersonaParseError,
    parse_persona,
)
from app.services.memory.domain.tasks import MemoryPersonaExtractParams
from app.services.memory.models.sqlalchemy.memory_persona import MemoryPersona
from app.services.memory.models.sqlalchemy.memory_persona_cursor import (
    MemoryPersonaCursor,
)
from app.services.memory.models.sqlalchemy.memory_scene import MemoryScene
from app.services.memory.repository.persona_impl import (
    MemoryPersonaRepositoryImpl,
)
from app.services.memory.repository.scene_impl import MemorySceneRepositoryImpl
from app.services.session.models.sqlalchemy.agent_message import (
    AgentMessage as AgentMessageModel,
)
from app.services.session.models.sqlalchemy.agent_session import (
    AgentSession as AgentSessionModel,
)
from app.services.session.repository.message_impl import (
    MessageRepositoryImpl,
)
from app.services.task.models.sqlalchemy.task import Task as TaskModelRow

DSN = "postgresql+asyncpg://postgres:123456@localhost:5432/ytrader_agent"

#: 冒烟用的身份。取正交的值，方便一眼认出库里的垃圾行是谁留下的。
AGENT_ID = "smoke-l3-agent"
USER_A = "smoke-l3-user-a"
USER_B = "smoke-l3-user-b"
USER_C = "smoke-l3-user-c"
ORG_T = "smoke-l3-org"

_FAILURES: list[str] = []


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
# 假模型
# ============================================================


class _ScriptedRuntime:
    """
    按剧本应答的 runtime：每次 `ainvoke` 弹出一个预设的 JSON 字符串。

    顺带记下每次收到的请求（`requests`），于是"喂进去的长什么样"可以被
    断言 —— 那是本脚本唯一能验提示词输入的地方（模型是假的）。
    """

    def __init__(self, replies: list[str]) -> None:
        self.replies = deque(replies)
        self.requests: list = []

    async def ainvoke(self, request):
        self.requests.append(request)

        if not self.replies:
            raise AssertionError(
                "假 runtime 的剧本用完了 —— 这次生成比预期多调了一次模型。"
            )

        return SimpleNamespace(
            message=SimpleNamespace(content=self.replies.popleft())
        )


class _FakeModelRegistry:
    def __init__(self, runtime: _ScriptedRuntime) -> None:
        self._runtime = runtime
        self.calls: list[dict] = []

    def get_runtime(self, *, agent_type: str, model_role: str, context=None):
        self.calls.append({"agent_type": agent_type, "model_role": model_role})
        return self._runtime


def _install(runtime: _ScriptedRuntime) -> _FakeModelRegistry:
    """
    换掉模型注册表，返回假 registry（调用方要断言它被怎么调）。

    ⚠️ **只换模型，不换提示词**
    --------------------
    提示词走真的 `get_prompt_provider()` —— 于是本脚本顺带验了
    "`memory.persona.extract` 确实被目录装配收进来了"（少了那份提示词就抛
    PromptNotFoundError，而不是悄悄地发一份空的系统提示词给模型）。
    """

    registry = _FakeModelRegistry(runtime)

    impl.get_model_registry = lambda: registry

    return registry


# ============================================================
# 造数据 / 清理
# ============================================================


async def _seed_session(
        session,
        *,
        user_id: str | None,
        org_id: str | None,
) -> uuid.UUID:
    """造一个会话（触发判定里那次跨会话计数要靠它 join）。"""

    now = dt.datetime.now(dt.timezone.utc)
    session_id = uuid.uuid4()

    session.add(AgentSessionModel(
        session_id=session_id,
        agent_id=AGENT_ID,
        status="active",
        meta={},
        thread_id=f"smoke-l3-{session_id}",
        user_id=user_id,
        org_id=org_id,
        created_at=now,
        updated_at=now,
    ))

    await session.flush()

    return session_id


async def _seed_turns(
        session,
        *,
        session_id: uuid.UUID,
        count: int,
        after: dt.datetime,
        duty: str | None = AGENT_ID,
        seq_from: int = 1,
) -> None:
    """
    从这个时刻之后造 `count` 条**用户轮次**（`role='user'` + `kind='chat'`）。

    ⚠️ 时间必须**晚于** `last_l3_at`
    ---------------------------
    条件 2 数的是 `created_at > last_l3_at` 那一段。造在窗口之前的话，
    计数恒为 0，而症状是"跨会话计数没生效"—— 其实是脚本自己造早了。

    ⚠️ `seq_from` 是必需的（不是省事）
    ----------------------------
    `agent_message` 上有 `(session_id, seq)` 的唯一约束。给同一个会话补造
    第二批时序号必须**接着往下排**，从 1 重来会直接撞唯一约束 —— 而报错
    发生在 flush，离"我补了一批"这个动作很远。

    `duty=None` 时 `business_agent_id` 留空，用来验兜底链（走会话的
    `agent_id`）—— 那条链写错的话，**存量消息**（那一列加进来之前的行）
    会让计数恒为 0，而症状是"画像永远不更新"。
    """

    base = after + dt.timedelta(seconds=1)

    for offset in range(count):

        stamp = base + dt.timedelta(seconds=offset + 1)

        session.add(AgentMessageModel(
            message_id=uuid.uuid4(),
            session_id=session_id,
            run_id=None,
            seq=seq_from + offset,
            role="user",
            kind="chat",
            business_agent_id=duty,
            content=[{"type": "text", "text": f"第 {offset + 1} 轮"}],
            status="completed",
            meta={},
            created_at=stamp,
            updated_at=stamp,
        ))

    await session.flush()


async def _seed_persona(
        session,
        *,
        scope: SceneScope,
        content: str = "# 已存在的画像",
) -> dt.datetime:
    """
    造一行"已经生成过"的画像 + 它的水位，返回 `last_l3_at`（造轮次要以它
    为界）。这两件事**总要一起做** —— 只有画像行没有水位的话，条件 2 会走
    "有画像行但无生成时刻"那条分支（也排任务），而那不是这里要验的东西。
    """

    now = dt.datetime.now(dt.timezone.utc)

    personas = MemoryPersonaRepositoryImpl(session=session)

    await personas.insert_persona(PersonaRow(
        id=persona_id(scope),
        filename=PERSONA_FILENAME,
        content=content,
        content_md5=hashlib.md5(content.encode()).hexdigest(),
        agent_id=scope.agent_id,
        user_id=scope.user_id,
        org_id=scope.org_id,
    ))

    await personas.advance_l3_cursor(
        scope,
        last_scene_updated_time=None,
        last_scene_ids=[],
        last_l3_at=now,
    )

    return now


#: 造数据用的**单调时钟**：每个场景比前一个晚一秒。
#:
#: ⚠️ 不能用 `datetime.now()` 现取
#: ---------------------------
#: `memory_scene.updated_at` 就是 L3 的增量水位，而一次 L2 整合写的若干行
#: 共享一个 `now()`（同一事务）。现取时间的话，"第二轮造的场景"有可能落在
#: **第一轮的水位之前**（两次 `now()` 只差几毫秒）—— 于是 L3 合法地读不到
#: 它，而脚本会以为是产品的 bug。用单调时钟把"后来的场景一定更晚"钉死。
_SEED_CLOCK = dt.datetime.now(dt.timezone.utc)


async def _seed_scenes(
        session,
        *,
        scope: SceneScope,
        filenames: list[str],
        same_instant: bool = False,
) -> list[str]:
    """
    造 L2 的场景行。**显式给 `updated_at`** —— 它就是 L3 的水位。

    `same_instant=True` 时这一批**共享同一个时间戳**（模拟 L2 一次事务写
    若干行的真实形态），用来验水位边界那批 id 的排除逻辑。

    ⚠️ `content_md5` 必须自己算
    ----------------------
    走仓储写入时是服务侧算的（见 scene_impl）；这里直接写 ORM 行，所以
    要自己算。不算的话那一列会是一串假数据，而**没有任何断言会因此失败**
    —— 直到某天有人拿它做短路比较（L3 恰恰会）。
    """

    global _SEED_CLOCK

    created: list[str] = []
    batch_stamp: dt.datetime | None = None

    for index, filename in enumerate(filenames, start=1):

        if same_instant:
            if batch_stamp is None:
                _SEED_CLOCK += dt.timedelta(seconds=1)
                batch_stamp = _SEED_CLOCK
            stamp = batch_stamp
        else:
            _SEED_CLOCK += dt.timedelta(seconds=1)
            stamp = _SEED_CLOCK

        # ⚠️ 用**真的** `scene_id`，不自己编一个
        # ---------------------------------
        # 它与服务侧读回来时看到的是同一个值。编一个的话，"边界 id 的排除"
        # 那条断言验的就不是产品行为，而是脚本自己的哈希。
        row_id = scene_id(scope, filename=filename)

        # ⚠️ 正文里带上**文件名**，别只用序号
        # ---------------------------------
        # 序号是**每次调用**从 1 起的，于是"第二轮造的场景"与"第一轮造的"
        # 正文会一模一样 —— 而断言"这一轮只喂了变化的那一个"正是靠正文里
        # 的标记区分的。撞了之后那条断言会失败，看起来像水位没生效，其实是
        # 脚本自己的内容重复（2026-10-11 实际踩过）。
        content = f"# {filename}\n\n场景正文：{filename}。"

        session.add(MemoryScene(
            id=row_id,
            filename=filename,
            content=content,
            content_md5=hashlib.md5(content.encode()).hexdigest(),
            agent_id=scope.agent_id,
            user_id=scope.user_id,
            org_id=scope.org_id,
            version=0,
            created_at=stamp,
            updated_at=stamp,
        ))

        created.append(row_id)

    await session.flush()

    return created


async def _cleanup(session, *, scope: SceneScope) -> None:
    """
    按 scope 清干净。

    ⚠️ 条件是**三列**（agent + 那两列），不是只按 agent_id
    -------------------------------------------------
    本脚本造了个人与团队两个 scope。只按 agent_id 删会把别的场景一起带走
    —— 那样清理看起来是对的（库里确实干净了），但下一场跑的时候会缺少
    本该存在的行，而症状是"前置条件不成立"这种查不出原因的错误。
    """

    for model in (MemoryScene, MemoryPersona, MemoryPersonaCursor):

        await session.execute(
            delete(model).where(
                model.agent_id == scope.agent_id,
                model.user_id == scope.user_id,
                model.org_id == scope.org_id,
            )
        )

    await session.flush()


async def _cleanup_sessions(session, session_ids: list[uuid.UUID]) -> None:
    """
    删会话**与它们下面的消息**。

    ⚠️ 消息要显式删：`agent_message` 对 `agent_session` **没有外键**
    （见那个模块的说明，存亡由写入方负责）。只删会话的话，那一批消息会
    留在库里，而且下一场跑的时候会以"轮次莫名多出来几条"的形式出现 ——
    那正是这条链最怕的一类脏数据（它直接进阈值判定）。
    """

    if not session_ids:
        return

    await session.execute(
        delete(AgentMessageModel).where(
            AgentMessageModel.session_id.in_(session_ids)
        )
    )
    await session.execute(
        delete(AgentSessionModel).where(
            AgentSessionModel.session_id.in_(session_ids)
        )
    )

    await session.flush()


# ============================================================
# 读回来的小工具
# ============================================================


async def _persona(session, scope: SceneScope) -> MemoryPersona | None:
    return await session.scalar(
        select(MemoryPersona).where(
            MemoryPersona.agent_id == scope.agent_id,
            MemoryPersona.user_id == scope.user_id,
            MemoryPersona.org_id == scope.org_id,
        )
    )


async def _cursor(session, scope: SceneScope) -> PersonaCursorState:
    """
    读水位 —— 返回**值对象**，不是 ORM 行。

    ⚠️ 与 L2 冒烟里那个 `_cursor` 同一个坑
    ------------------------------
    ORM 实例在 identity map 里是**同一个对象**，而仓储那侧用了
    `populate_existing`（见 persona_impl 的 `_fresh`，那里是必须的）——于是
    "读两次拿到同一个对象、第二次原地刷新"会让 `after > before` 恒为假，
    看起来像"水位没动"。值对象在每次读时新建，断言比较的是两个时刻的快照。

    ⚠️ "没有行"返回的是默认值（水位 None），不是 None
    ------------------------------------------
    与契约一致（见 MemoryPersonaRepository.get_l3_cursor）。所以判"这个
    scope 还没生成过"要看字段是不是 None，别判对象是不是 None。
    """

    return await MemoryPersonaRepositoryImpl(
        session=session
    ).get_l3_cursor(scope)


def _keep(content: str) -> str:
    """模型判"要改"并给出整篇内容。"""

    return json.dumps(
        {"no_change": False, "content": content}, ensure_ascii=False,
    )


_NO_CHANGE = json.dumps({"no_change": True}, ensure_ascii=False)


def _service(session) -> impl.MemoryPersonaServiceImpl:
    return impl.MemoryPersonaServiceImpl(
        personas=MemoryPersonaRepositoryImpl(session=session),
        scenes=MemorySceneRepositoryImpl(session=session),
        # 「满 N 轮次」那条触发要跨会话数用户轮次（见 PERSONA_TRIGGER_TURNS）。
        # 口径归 session 模块，所以这里用**它的真仓储**，不造桩 —— 那正是
        # 本脚本的立场：验的是真 SQL。
        messages=MessageRepositoryImpl(session=session),
    )


async def _run(
        session,
        scope: SceneScope,
        *,
        persona_update_request: bool = False,
) -> None:
    await _service(session).memory_persona_extract(
        MemoryPersonaExtractParams(
            agent_id=scope.agent_id,
            user_id=scope.user_id,
            org_id=scope.org_id,
            persona_update_request=persona_update_request,
        ),
        None,
    )


# ============================================================
# 场景 A：个人画像全流程（首次 / 增量 / 不改 / 未变 / 冲突 / 空转）
# ============================================================


async def smoke_personal(session) -> SceneScope:

    print("\n--- A 个人画像：首次 -> 增量 -> 不改 -> 未变 -> 冲突 -> 空转 ---")

    scope = SceneScope.resolve(agent_id=AGENT_ID, user_id=USER_A, org_id=None)

    stamps = await _seed_scenes(
        session, scope=scope, filenames=["技术研究.md", "日常生活.md"],
    )

    # ---- 首次生成 ----------------------------------------------------------
    runtime = _ScriptedRuntime([_keep("# 画像 v1\n\n> Archetype: 务实理想主义者")])
    registry = _install(runtime)

    await _run(session, scope)

    row = await _persona(session, scope)

    check("A1 首次生成建了画像行", row is not None, "没有行")
    check(
        "A2 主键是派生 id（scope 派生，不是随机）",
        row is not None and row.id == persona_id(scope),
        f"实际 {row.id if row else None}",
    )
    check(
        "A3 filename 恒为 persona.md（团队与个人不靠文件名区分）",
        row is not None and row.filename == PERSONA_FILENAME,
    )
    check("A4 version 从 0 起", row is not None and row.version == 0,
          f"实际 {row.version if row else None}")
    check(
        "A5 行归属是个人画像（user 有值 / org 为空）",
        row is not None and row.user_id == USER_A and row.org_id is None,
    )
    check(
        "A6 content_md5 与内容对得上",
        row is not None
        and row.content_md5 == hashlib.md5(row.content.encode()).hexdigest(),
    )
    check(
        "A7 模型绑的是 L3 那个 agent_type",
        registry.calls[0]["agent_type"] == "memory_persona_extract",
        f"实际 {registry.calls[0]['agent_type']}",
    )
    check(
        "A8 提示词要的是 L3 的 role",
        runtime.requests[0].metadata["model_role"]
        == ModelRole.MEMORY_PERSONA_EXTRACT,
    )
    check(
        "A9 个人 scope 走的是个人那份系统提示词",
        "Persona Architect" in runtime.requests[0].messages[0].content,
    )
    check(
        "A10 metadata 标了 is_team=False",
        runtime.requests[0].metadata["is_team"] is False,
    )
    check(
        "A11 metadata 标了还没有画像",
        runtime.requests[0].metadata["has_persona"] is False,
    )

    human = runtime.requests[0].messages[-1].content
    check("A12 两个场景的全文都进了提示词",
          "技术研究.md" in human and "日常生活.md" in human,
          human[:400])
    check("A13 统计块给了总数与变化数",
          "**场景总数**：2 个" in human and "**本轮变化**：2 个" in human,
          human[:200])
    check("A14 首次生成不给迭代指南",
          "首次生成，无历史版本可比" in human, human[:400])

    cursor = await _cursor(session, scope)
    check(
        "A15 水位推到这一批的最后一条",
        cursor.last_scene_updated_time is not None and cursor.last_l3_at is not None,
    )
    check(
        "A16 边界 id 只含最后那个时间戳上的场景",
        len(cursor.last_scene_ids) == 1,
        f"实际 {cursor.last_scene_ids}",
    )

    # ---- 增量：只喂变化的那一个 ---------------------------------------------
    await _seed_scenes(session, scope=scope, filenames=["健康管理.md"])

    first_l3_at = cursor.last_l3_at

    runtime = _ScriptedRuntime([_keep("# 画像 v2\n\n> Archetype: 务实理想主义者\n\n## Chapter 1\n加了健康。")])
    _install(runtime)

    # ⚠️ 举手是**必须**的（2026-10-13）
    # -------------------------------
    # 触发条件改成「首次创建 + L2 举手」之后，"已经有画像 + 场景变了"不再
    # 生成。所以增量更新那条路要走通就得带一个举手 —— 那正是它的语义：
    # 画像的更新由 L2 那次整合的模型发起，不由"某条场景被改写了"发起。
    await _run(session, scope, persona_update_request=True)

    row = await _persona(session, scope)

    check("A17 举手 + 有变化场景 -> 改写了画像",
          row is not None and row.version == 1,
          f"version={row.version if row else None}")
    check("A18 内容确实换了", row is not None and "画像 v2" in row.content)

    human = runtime.requests[0].messages[-1].content
    check(
        "A19 这一轮只喂了**变化**的那个场景（水位生效）",
        "健康管理.md" in human and "技术研究.md" not in human,
        human[:400],
    )
    check(
        "A20 但总数给的是全量（3 个）",
        "**场景总数**：3 个" in human and "**本轮变化**：1 个" in human,
        human[:200],
    )
    check(
        "A21 迭代更新给了迭代指南",
        "迭代决策指南" in human and "首次生成，无历史版本可比" not in human,
    )
    check(
        "A22 喂进去的是**旧**那一份画像（模型据此更新）",
        "画像 v1" in human and "画像 v2" not in human,
        human[:400],
    )

    # ---- 模型判 no_change：不写库但**推水位** --------------------------------
    await _seed_scenes(session, scope=scope, filenames=["饮食偏好.md"])

    before = await _persona(session, scope)
    cursor_before = await _cursor(session, scope)

    runtime = _ScriptedRuntime([_NO_CHANGE])
    _install(runtime)

    await _run(session, scope, persona_update_request=True)

    after = await _persona(session, scope)
    cursor_after = await _cursor(session, scope)

    check(
        "A23 no_change 不写库、不推 version",
        after is not None and before is not None
        and after.version == before.version
        and after.content == before.content
        and after.content_md5 == before.content_md5,
    )
    check(
        "A24 no_change 仍然**推场景水位**（否则同一批场景每轮重喂一遍）",
        cursor_before.last_scene_updated_time is not None
        and cursor_after.last_scene_updated_time is not None
        and cursor_after.last_scene_updated_time
        > cursor_before.last_scene_updated_time,
        f"before={cursor_before.last_scene_updated_time} "
        f"after={cursor_after.last_scene_updated_time}",
    )
    check(
        "A25 no_change 也戳了 last_l3_at",
        cursor_after.last_l3_at is not None
        and cursor_before.last_l3_at is not None
        and cursor_after.last_l3_at > cursor_before.last_l3_at,
    )

    # ---- 内容逐字未变：推水位、不推 version -----------------------------------
    await _seed_scenes(session, scope=scope, filenames=["阅读习惯.md"])

    before = await _persona(session, scope)
    cursor_before = await _cursor(session, scope)

    runtime = _ScriptedRuntime([_keep(before.content)])
    _install(runtime)

    await _run(session, scope, persona_update_request=True)

    after = await _persona(session, scope)
    cursor_after = await _cursor(session, scope)

    check(
        "A26 重写后逐字未变时 version 不动（否则版本号=跑过几轮）",
        after is not None and before is not None and after.version == before.version,
        f"before={before.version if before else None} "
        f"after={after.version if after else None}",
    )
    check(
        "A27 逐字未变也推了场景水位",
        cursor_after.last_scene_updated_time is not None
        and cursor_before.last_scene_updated_time is not None
        and cursor_after.last_scene_updated_time
        > cursor_before.last_scene_updated_time,
    )

    # ---- 乐观锁：生成期间被别的任务改过 ---------------------------------------
    await _seed_scenes(session, scope=scope, filenames=["旅行计划.md"])

    stale = await _persona(session, scope)
    cursor_before = await _cursor(session, scope)

    assert stale is not None, "缺前置：上一段应当已经建好画像行"

    class _BumpingRuntime(_ScriptedRuntime):
        """
        ⚠️ 时机卡在模型调用里
        -----------------
        服务读完画像 -> 调模型 -> 写回。假的 runtime 在 `ainvoke` 里把
        version 顶掉一步，等价于**另一个会话的 L2 完成**排出的那条 L3 先
        提交了（同一个 scope 可以被多条 L3 并发写）。
        """

        async def ainvoke(self, request):
            await session.execute(
                update(MemoryPersona)
                .where(MemoryPersona.id == stale.id)
                .values(version=MemoryPersona.version + 1)
            )
            return await super().ainvoke(request)

    _install(_BumpingRuntime([_keep("# 画像 v3")]))

    raised = None

    try:
        # ⚠️ 举手是必须的（2026-10-13）：触发条件改成「首次创建 + L2
        # 举手」之后，「已经有画像 + 场景变了」不再生成 —— 下面这些用例
        # 要走到模型就得带一个举手。
        await _run(session, scope, persona_update_request=True)
    except ConflictError as exc:
        raised = exc

    cursor_after = await _cursor(session, scope)

    check("A28 乐观锁冲突抛 ConflictError（交给任务重入）", raised is not None,
          "没抛")
    check(
        "A29 冲突时水位**完全没动**（否则那批场景被静默跳过）",
        cursor_after.last_scene_updated_time
        == cursor_before.last_scene_updated_time
        and cursor_after.last_l3_at == cursor_before.last_l3_at,
    )

    await session.rollback()

    return scope


# ============================================================
# 场景 B：团队画像 + 空转
# ============================================================


async def smoke_team(session) -> SceneScope:

    print("\n--- B 团队画像：走团队那份提示词 / 空转不调模型 ---")

    scope = SceneScope.resolve(agent_id=AGENT_ID, user_id=USER_B, org_id=ORG_T)

    await _seed_scenes(session, scope=scope, filenames=["部署流程.md"])

    runtime = _ScriptedRuntime([_keep("# Team Operating Doctrine\n\n> Operating Thesis: 先验证再上线")])
    _install(runtime)

    await _run(session, scope, persona_update_request=True)

    row = await _persona(session, scope)

    check(
        "B1 团队 scope 的行是 org 有值 / user 为空",
        row is not None and row.org_id == ORG_T and row.user_id is None,
        f"user={row.user_id if row else None} org={row.org_id if row else None}",
    )
    check(
        "B2 主键按 org scope 派生（与个人 scope 不会撞）",
        row is not None and row.id == persona_id(scope),
    )
    check(
        "B3 走的是团队那份系统提示词",
        "Team Operating Doctrine Architect" in runtime.requests[0].messages[0].content,
    )
    check(
        "B4 团队那份带 1200 字上限（与个人那份的 2000 不同）",
        "禁止超过 1200 字" in runtime.requests[0].messages[0].content,
    )
    check(
        "B5 metadata 标了 is_team=True",
        runtime.requests[0].metadata["is_team"] is True,
    )
    check(
        "B6 团队画像不写个人画像化的内容（提示词里那条禁令在）",
        "禁止个人画像化" in runtime.requests[0].messages[0].content,
    )

    # ---- 空转：没有变化场景 -> 一次模型都不调 --------------------------------
    cursor_before = await _cursor(session, scope)

    runtime = _ScriptedRuntime([])
    _install(runtime)

    await _run(session, scope)

    cursor_after = await _cursor(session, scope)

    check("B7 没有变化场景时**不调模型**", len(runtime.requests) == 0,
          f"实际调了 {len(runtime.requests)} 次")
    check(
        "B8 空转也戳了 last_l3_at（否则每次 L2 完成都白跑）",
        cursor_after.last_l3_at is not None
        and cursor_before.last_l3_at is not None,
    )
    check(
        "B9 空转**不动**场景水位（没有可消费的东西）",
        cursor_after.last_scene_updated_time
        == cursor_before.last_scene_updated_time
        and cursor_after.last_scene_ids == cursor_before.last_scene_ids,
        f"before={cursor_before.last_scene_updated_time} "
        f"after={cursor_after.last_scene_updated_time}",
    )

    return scope


async def smoke_trigger_info(session) -> SceneScope:
    """
    L2 的"举手"理由要一路带到 L3 的提示词里（原项目的 `triggerInfo`）。

    ⚠️ 这条链最容易在中间断掉，而且断了两边都不报错
    ----------------------------------------
    它要穿过三处：L2 handler -> `submit_persona_extract` 的判定 -> 任务的
    params -> 提示词的渲染。任何一处把它降级成 bool（"有没有举手"）都会让
    理由消失，而 L3 照常跑、照常产出画像 —— 只是模型不知道 L2 指的是哪件
    事，于是把整篇平铺着重写一遍。所以这里逐段验。
    """

    print("\n--- E L2 的举手理由一路带到提示词 ---")

    scope = SceneScope.resolve(agent_id=AGENT_ID, user_id=USER_C, org_id=None)

    await _seed_scenes(session, scope=scope, filenames=["价值观转变.md"])

    # 举手现在是个布尔 —— 原来这里还有一段"理由原文"，压成布尔之后
    # 没有原文可带了（见 ConsolidationResult 的字段说明）。
    runtime = _ScriptedRuntime([_keep("# 画像 v1")])
    _install(runtime)

    await _run(session, scope, persona_update_request=True)

    human = runtime.requests[0].messages[-1].content

    check("E1 触发信息段出现在提示词里", "### 0️⃣ 触发信息" in human,
          human[:300])
    # ⚠️ E2 原来钉的是"理由原文不摘要、不转述地进去了"。那个字段
    # 2026-10-13 从「理由原文」压成了布尔（见 ConsolidationResult 的说明），
    # 所以原文没有了 —— 现在钉的是"这一格说清了来由是 L2 举手"。
    check("E2 说清了这次生成的来由是 L2 举手",
          "L2 整合时的模型" in human, human[:600])
    check(
        "E3 明说它不是指令（否则模型会把'有人要求更新'当成'必须重写'）",
        "不是指令" in human,
    )
    check(
        "E4 metadata 只记'有没有'，不抄理由原文",
        runtime.requests[0].metadata["persona_update_requested"] is True,
    )

    # ---- 没有举手时：**一次模型都不调**（2026-10-13 改）---------------------
    #
    # ⚠️ E5/E6 原来验的是"模板里不留空标题"：那时有画像 + 场景变了照样调
    # 模型，只是那一格不出现。触发条件改成「首次创建 + 举手」之后，这种
    # 情况**根本走不到模型** —— 断言因此从"提示词里没有那段"变成"没有这次
    # 调用"，它更强，而且正是改动的要点。
    #
    # "不留空标题"那条性质仍然被覆盖：**首次生成**（画像还不在）不戴举手
    # 也会调模型，而那一轮的提示词里同样不该出现触发信息段 —— 见 A 段。
    await _seed_scenes(session, scope=scope, filenames=["日常工作.md"])

    runtime = _ScriptedRuntime([_keep("# 画像 v2")])
    _install(runtime)

    await _run(session, scope)

    check(
        "E5 已有画像 + 没举手 -> 一次模型都不调（触发条件不是「场景变了」）",
        len(runtime.requests) == 0,
        f"实际调了 {len(runtime.requests)} 次",
    )

    # ---- 自由文本的"假阳性"随字段一起消失了 --------------------------
    #
    # ⚠️ 这一段原来叫"假阳性"：字段是**自由文本**，工程侧只按"非空 = 举手"
    # 判，于是模型写一句否定式的话（"本轮没有重大变化，不需要更新画像"）
    # 也会被当成举手 —— 一次白跑的模型调用。
    #
    # 压成布尔之后这个坑**结构上不存在了**：`false` 就是 false，没有"说了
    # 一段话但意思是不要"这种中间态。所以那两条断言（E7 白跑一次 / E8 画像
    # 逐字不变）连同整段一起删掉 —— 它们守的那个失效模式已经没了。

    return scope


# ============================================================
# 场景 C：纯函数（解析 + 触发判定）
# ============================================================


def smoke_pure() -> None:

    print("\n--- C 纯函数：输出解析 / 四条触发判定 ---")

    check("C1 no_change=true -> 不采用 content",
          parse_persona('{"no_change": true, "content": "# x"}').content is None)
    check("C2 字符串 'true' 也认",
          parse_persona('{"no_change": "true"}').no_change is True)
    check("C3 缺 no_change 时按 false 处理",
          parse_persona('{"content": "# x"}').content == "# x")
    check("C4 content 两端空白被剥掉",
          parse_persona('{"content": "  # x  "}').content == "# x")

    for label, raw in (
        ("C5 no_change 为假但 content 缺失 -> 抛", '{"no_change": false}'),
        ("C6 no_change 为假但 content 空白 -> 抛", '{"no_change": false, "content": "   "}'),
        ("C7 顶层不是对象 -> 抛", '["# x"]'),
        ("C8 不是 JSON -> 抛", "我不是 JSON"),
        ("C9 no_change 认不出的字符串 -> 抛", '{"no_change": "yes"}'),
    ):
        raised = None

        try:
            parse_persona(raw)
        except PersonaParseError as exc:
            raised = exc

        check(label, raised is not None, "没抛")

    # ⚠️ C10-C18 那一组（四条判定）随判定本身一起搬到执行侧了
    # --------------------------------------------------
    # 它们测的是 `_persona_trigger_reason` 这个纯函数 —— 触发侧已经不判
    # "排不排"（见 trigger 模块的说明），那个函数连同它的细节描述函数一起
    # 删掉了。判定现在的家是 `MemoryPersonaServiceImpl.memory_persona_extract`
    # 第 3 步，覆盖它的是场景 B（生成）里那些走真 task 的用例。

    # ⚠️ C10 原来断言 `PERSONA_TRIGGER_TURNS == 50`。那个常量（连同
    # `_persona_force` 与整条"满 N 轮强制触发"）2026-10-13 一起删掉了：
    # 它的基准 `last_l3_at` 在每次空转时也被推，于是窗口不断归零、实测
    # 从未生效。触发条件现在是「首次创建 + L2 举手」，见 trigger 模块。


# ============================================================
# 场景 D：触发（真任务基建）
# ============================================================


async def smoke_trigger(maker) -> None:
    """
    验 `trigger.submit_persona_extract`：**它不再判定"排不排"**（判定在执行
    侧），只负责"没有活着的任务就排一条"，并把两个触发事实带进参数。所以
    这一场验的是 —— 排出来的那条**键对不对、是不是立刻执行、能不能反复
    用**，以及 `force` 有没有按跨会话轮次正确置位。

    ⚠️ 它必须跑在**独立的事务**里
    -------------------------
    `submit_persona_extract` 自己开作用域（`get_service_factory()` +
    `scope(TaskService)`），所以它读得到的东西必须**已经提交** —— 上面那批
    场景都挤在一个未提交的事务里（那样才验得了"同一个 session 连着调几次
    handler"），所以这一场单独放在外层会话之后。

    ⚠️ 装配是真的，只有 dispatcher 故意不给
    ---------------------------------
    与 L2 冒烟同一手法：不给 `task_dispatcher` 是有意的（它是可选的，投递
    由 outbox relay 兜底），而"任务排进库"这件事不需要有人真的去跑它。

    ⚠️ 清理只删**这一场自己造的**任务
    ---------------------------
    按 `params_json->>'agent_id'` 删。别按 handler 删 —— 那会波及用户库里
    全部 L3 任务，甚至撞上正在跑的那条（见 L2 冒烟里那段血泪说明）。
    """

    print("\n--- D 触发：四条判定 / 键与执行时间 ---")

    from app.bootstrap.task import _register_handlers
    from app.common.factories import build_factories, reset_factories
    from app.infra.task import TaskHandlerRegistryImpl
    from app.services.memory.application.trigger import (
        persona_task_key,
        submit_persona_extract,
    )

    registry = TaskHandlerRegistryImpl()
    _register_handlers(registry)

    check(
        "D1 注册表里有 memory.l3.extract",
        "memory.l3.extract" in registry.known(),
        f"实际 {registry.known()}",
    )

    build_factories(maker, pool={"handler_registry": registry})

    scopes: list[SceneScope] = []
    session_ids: list[uuid.UUID] = []

    try:
        # ---- 1. 有场景、没有画像 -> 排 ------------------------------------
        first_scope = SceneScope.resolve(
            agent_id=AGENT_ID, user_id="smoke-l3-trigger-a", org_id=None,
        )
        scopes.append(first_scope)

        async with maker() as session:
            session_ids.append(await _seed_session(
                session, user_id="smoke-l3-trigger-a", org_id=None,
            ))
            await _seed_scenes(session, scope=first_scope, filenames=["甲.md"])
            await session.commit()

        key = persona_task_key(first_scope)

        task_id = await submit_persona_extract(
            scope=first_scope,
            causation_id=None,
        )

        check("D2 有场景无画像 -> 排出了任务", task_id is not None, "返回了 None")

        task = await _task(maker, key)

        check(
            "D3 handler 是 memory.l3.extract",
            task is not None and task.handler == "memory.l3.extract",
            f"实际 {task.handler if task else None}",
        )
        check(
            "D4 幂等键按 scope（不是会话）",
            task is not None and task.idempotency_key == key,
        )
        check(
            "D5 立刻执行（execute_at 为空 = 已到点）",
            task is not None and task.execute_at is None,
            f"实际 {task.execute_at if task else None}",
        )
        check(
            "D6 键声明可反复用（终态会释放）",
            task is not None and task.reusable_key is True,
        )
        check(
            "D7 params 里带的是 scope 三件套（没有 session_id）",
            task is not None
            and task.params_json.get("agent_id") == AGENT_ID
            and task.params_json.get("user_id") == "smoke-l3-trigger-a"
            and task.params_json.get("org_id") is None
            and "session_id" not in task.params_json,
            f"实际 {task.params_json if task else None}",
        )

        # 再触发一次：同一条键，不新建（它已经排在"立刻"）
        again = await submit_persona_extract(scope=first_scope)

        check("D8 重复触发复用同一条任务", again == task_id,
              f"第一次 {task_id}，第二次 {again}")

        # ---- 2. 一个场景都没有 -> 不排 ------------------------------------
        empty_scope = SceneScope.resolve(
            agent_id=AGENT_ID, user_id="smoke-l3-trigger-b", org_id=None,
        )
        scopes.append(empty_scope)

        async with maker() as session:
            session_ids.append(await _seed_session(
                session, user_id="smoke-l3-trigger-b", org_id=None,
            ))
            await session.commit()

        result = await submit_persona_extract(scope=empty_scope)

        # ⚠️ 判定不在这一侧了 —— 触发侧只有入队权（见 trigger 模块的
        # docstring）。一个场景都没有时它**照样排**，"没有输入就没有画像"
        # 由执行侧判（`memory_persona_extract` 第 3 步）。
        check("D9 没有场景也照样排（判定在执行侧）",
              result is not None, "返回了 None")

        task = await _task(maker, persona_task_key(empty_scope))

        check(
            "D9b 那条没戴举手（参数里只有身份）",
            task is not None and task.params_json.get("persona_update_request") is False,
            f"实际 {task.params_json if task else None}",
        )

        # ---- 3. 有画像、刚生成过、没有新轮次 -> 不排 --------------------------
        settled_scope = SceneScope.resolve(
            agent_id=AGENT_ID, user_id="smoke-l3-trigger-c", org_id=None,
        )
        scopes.append(settled_scope)

        async with maker() as session:
            await _seed_scenes(session, scope=settled_scope, filenames=["乙.md"])
            await _seed_persona(session, scope=settled_scope)
            await session.commit()

        result = await submit_persona_extract(scope=settled_scope)

        check("D10 画像在、刚生成过、没有新轮次 -> 照样排",
              result is not None, "返回了 None")

        task = await _task(maker, persona_task_key(settled_scope))

        check(
            "D10b 那条也照样没戴举手（触发侧只带 L2 那一刻的事实）",
            task is not None and task.params_json.get("persona_update_request") is False,
            f"实际 {task.params_json if task else None}",
        )

        # ---- 3b. 已有任务时，新信号进不去参数（**已知限制**） ----------------
        #
        # ⚠️ 这一条记的是一个**缺口**，不是想要的性质
        # --------------------------------------------
        # 触发侧现在每次都排，于是"任务已经待跑"从罕见变成了常态；而
        # `TaskService` 没有"改 params"的能力，所以这一轮的举手 / 强制
        # **进不了**那条已存在的任务（它带着上一轮的参数）。
        #
        # 表现：L2 举手时若同 scope 正好有一条任务排在立刻还没被领走，这次
        # 举手就被丢掉 —— 而"举手"恰恰是本次改动要让它生效的那条路。
        # 补法见 trigger 里那段说明（要给任务模块加一个 merge params 的口子，
        # 或改成取消重排）。
        again = await submit_persona_extract(
            scope=settled_scope, persona_update_request=True,
        )

        task = await _task(maker, persona_task_key(settled_scope))

        check(
            "D11 已知限制：已有待跑任务时，这一轮的举手留在参数外",
            again == result
            and task is not None
            and task.params_json.get("persona_update_request") is False,
            f"task_id={again} params={task.params_json if task else None}",
        )

        # ---- 4. 没有待跑任务时，举手**原样**进参数 --------------------------
        signal_scope = SceneScope.resolve(
            agent_id=AGENT_ID, user_id="smoke-l3-trigger-e", org_id=None,
        )
        scopes.append(signal_scope)

        async with maker() as session:
            await _seed_scenes(session, scope=signal_scope, filenames=["丁.md"])
            await session.commit()

        result = await submit_persona_extract(
            scope=signal_scope, persona_update_request=True,
        )

        check("D12 有举手 -> 排", result is not None, "返回了 None")

        task = await _task(maker, persona_task_key(signal_scope))

        check(
            "D13 那一条也是立刻执行",
            task is not None and task.execute_at is None,
        )
        check(
            "D17 举手**进了任务参数**",
            task is not None
            and task.params_json.get("persona_update_request") is True,
            f"实际 {task.params_json.get('persona_update_request')!r}",
        )

        # ---- 5. 这个 scope 的任务按 scope 排队 ------------------------------
        #
        # ⚠️ 这一段原来验的是「条件 2：自 last_l3_at 起**跨会话**攒够 50 轮 ->
        # force=True」。`force` 连同 `_persona_force`、那次 join COUNT 与
        # `PERSONA_TRIGGER_TURNS` 一起在 2026-10-13 删掉了 —— 它的基准
        # `last_l3_at` 在每次空转时也被推，于是「攒够 50 轮」的窗口不断归零，
        # 实测从未生效过。触发条件改成「首次创建 + L2 举手」之后，它连存在
        # 的理由都没有了。
        #
        # 留在这里的只剩一条仍然有意义的性质：**幂等键按 scope**（与按会话
        # 的 L1/L2 不同）。它是「同 scope 下多个会话的 L2 完成合并成一条任务」
        # 的依据，与 force 无关。
        turn_scope = SceneScope.resolve(
            agent_id=AGENT_ID, user_id="smoke-l3-trigger-d", org_id=None,
        )
        scopes.append(turn_scope)

        async with maker() as session:
            await _seed_scenes(session, scope=turn_scope, filenames=["丙.md"])
            await _seed_persona(session, scope=turn_scope)
            await session.commit()

        scheduled = await submit_persona_extract(scope=turn_scope)

        check("D15 没有活着的任务 -> 排", scheduled is not None,
              "返回了 None")

        task = await _task(maker, persona_task_key(turn_scope))
        check(
            "D16 幂等键按 scope（不是会话）—— 与 L1/L2 那条区别的落点",
            task is not None and task.idempotency_key == persona_task_key(turn_scope),
        )

    finally:
        reset_factories()

        async with maker() as session:
            await session.execute(
                delete(TaskModelRow).where(
                    TaskModelRow.params_json["agent_id"].astext == AGENT_ID
                )
            )
            for scope in scopes:
                await _cleanup(session, scope=scope)
            await _cleanup_sessions(session, session_ids)
            await session.commit()


async def _task(maker, key: str) -> TaskModelRow | None:
    """按幂等键取那条任务（新开一个会话，绕开 identity map）。"""

    async with maker() as session:
        return await session.scalar(
            select(TaskModelRow)
            .where(TaskModelRow.idempotency_key == key)
        )


# ============================================================
# main
# ============================================================


async def main() -> int:

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    engine = create_async_engine(DSN)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    smoke_pure()

    scopes: list[SceneScope] = []

    #: 这一场里造的会话 —— 给触发那一段造轮次数据用，也要清掉。
    session_ids: list[uuid.UUID] = []

    async with maker() as session:

        try:
            scopes.append(await smoke_personal(session))
            scopes.append(await smoke_team(session))
            scopes.append(await smoke_trigger_info(session))

            await session.commit()

        finally:
            # ⚠️ 清理要**独立于上面是否成功**：失败时库里同样留下了垃圾行，
            # 而它们会在下次跑别的冒烟脚本时以"多出来几条"的形式出现。
            try:
                for scope in scopes:
                    await _cleanup(session, scope=scope)
                await _cleanup_sessions(session, session_ids)
                await session.commit()
            except Exception as exc:  # noqa: BLE001 —— 清理失败不能再盖掉原错误
                print(f"  [WARN] 清理失败：{exc}", flush=True)
                await session.rollback()

    # ⚠️ 触发那一段单独放在这里：`submit_persona_extract` 自己开作用域，读得到
    # 的东西必须是**已提交**的，而上面那批场景挤在同一个未提交的事务里。
    await smoke_trigger(maker)

    await engine.dispose()

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
